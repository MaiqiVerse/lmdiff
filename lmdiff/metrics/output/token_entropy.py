from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lmdiff.metrics.base import BaseMetric, MetricLevel, MetricResult
from lmdiff.metrics.output._slicing import probe_predicting_logits, safe_probe_slice
from lmdiff.tokenizer_utils import tokenizers_equivalent

if TYPE_CHECKING:
    from lmdiff.engine import InferenceEngine


class TokenEntropy(BaseMetric):
    """Per-token entropy of next-token distributions, compared across two engines."""

    name = "token_entropy"
    level = MetricLevel.OUTPUT

    @classmethod
    def is_applicable(cls, config_a: Any, config_b: Any) -> bool:
        same = config_a.shares_tokenizer_with(config_b)
        return same is not False

    def compute(
        self,
        engine_a: InferenceEngine,
        engine_b: InferenceEngine,
        probes: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        if not tokenizers_equivalent(engine_a.tokenizer, engine_b.tokenizer):
            raise ValueError(
                "TokenEntropy requires matching tokenizers; got "
                f"{type(engine_a.tokenizer).__name__} vs "
                f"{type(engine_b.tokenizer).__name__}"
            )

        result_a = engine_a.get_logits(probes, topk=0)
        result_b = engine_b.get_logits(probes, topk=0)

        # Compute entropy only over probe-predicting positions so prefix text
        # (system_prompt, context) does not leak into the entropy delta.
        entropies_a: list[np.ndarray] = []
        entropies_b: list[np.ndarray] = []
        for i in range(len(probes)):
            slice_a = safe_probe_slice(result_a, i)
            slice_b = safe_probe_slice(result_b, i)
            la = probe_predicting_logits(result_a.logits[i], slice_a)
            lb = probe_predicting_logits(result_b.logits[i], slice_b)
            # Tail-align so A and B cover the same probe positions when one
            # engine has no prefix (P=0) and the other has one (P>=1).
            min_len = min(la.shape[0], lb.shape[0])
            if min_len == 0:
                entropies_a.append(np.array([], dtype=np.float32))
                entropies_b.append(np.array([], dtype=np.float32))
                continue
            la = la[-min_len:]
            lb = lb[-min_len:]
            entropies_a.append(_entropy_from_logits(la))
            entropies_b.append(_entropy_from_logits(lb))

        valid_a = [e for e in entropies_a if e.size > 0]
        valid_b = [e for e in entropies_b if e.size > 0]
        if not valid_a:
            raise ValueError("all probes yielded zero-length probe spans; cannot compute entropy")
        mean_a = np.mean([e.mean() for e in valid_a])
        mean_b = np.mean([e.mean() for e in valid_b])
        delta = float(mean_b - mean_a)

        per_prompt: list[dict] = []
        for i, probe in enumerate(probes):
            if entropies_a[i].size == 0:
                per_prompt.append({
                    "probe": probe,
                    "entropy_a": float("nan"),
                    "entropy_b": float("nan"),
                    "delta": float("nan"),
                    "skipped": True,
                })
                continue
            per_prompt.append({
                "probe": probe,
                "entropy_a": float(entropies_a[i].mean()),
                "entropy_b": float(entropies_b[i].mean()),
                "delta": float(entropies_b[i].mean() - entropies_a[i].mean()),
            })

        return MetricResult(
            name=self.name,
            level=self.level,
            value=delta,
            details={
                "mean_entropy_a": float(mean_a),
                "mean_entropy_b": float(mean_b),
                "per_prompt": per_prompt,
            },
        )

    @classmethod
    def requirements(cls) -> dict[str, bool]:
        return {"logits": True, "hidden_states": False, "generations": False}


def _entropy_from_logits(logits: torch.Tensor) -> np.ndarray:
    """Compute per-position entropy from logits. Shape: (seq_len, vocab)."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.numpy()
