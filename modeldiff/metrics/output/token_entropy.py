from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from modeldiff.metrics.base import BaseMetric, MetricLevel, MetricResult

if TYPE_CHECKING:
    from modeldiff.engine import InferenceEngine


class TokenEntropy(BaseMetric):
    """Per-token entropy of next-token distributions, compared across two engines."""

    name = "token_entropy"
    level = MetricLevel.OUTPUT

    def compute(
        self,
        engine_a: InferenceEngine,
        engine_b: InferenceEngine,
        probes: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        topk = kwargs.get("topk", 0)

        result_a = engine_a.get_logits(probes, topk=topk)
        result_b = engine_b.get_logits(probes, topk=topk)

        entropies_a = [_entropy_from_logits(lg) for lg in result_a.logits]
        entropies_b = [_entropy_from_logits(lg) for lg in result_b.logits]

        mean_a = np.mean([e.mean() for e in entropies_a])
        mean_b = np.mean([e.mean() for e in entropies_b])
        delta = float(mean_b - mean_a)

        per_prompt: list[dict] = []
        for i, probe in enumerate(probes):
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
