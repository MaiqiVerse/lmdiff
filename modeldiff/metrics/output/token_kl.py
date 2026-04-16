from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from modeldiff.metrics.base import BaseMetric, MetricLevel, MetricResult
from modeldiff.tokenizer_utils import tokenizers_equivalent

if TYPE_CHECKING:
    from modeldiff.engine import InferenceEngine


class TokenKL(BaseMetric):
    """Per-token KL divergence: KL(P_A || P_B) averaged over probes."""

    name = "token_kl"
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
                "TokenKL requires matching tokenizers; got "
                f"{type(engine_a.tokenizer).__name__} vs "
                f"{type(engine_b.tokenizer).__name__}"
            )

        # KL must use full vocab: top-k subsets from A and B have different
        # token indices, so positional alignment would be meaningless.
        result_a = engine_a.get_logits(probes, topk=0)
        result_b = engine_b.get_logits(probes, topk=0)

        kl_ab_list: list[np.ndarray] = []
        kl_ba_list: list[np.ndarray] = []

        per_prompt: list[dict] = []
        for i, probe in enumerate(probes):
            logits_a = result_a.logits[i].float()
            logits_b = result_b.logits[i].float()

            kl_ab = _kl_divergence(logits_a, logits_b)
            kl_ba = _kl_divergence(logits_b, logits_a)
            kl_ab_list.append(kl_ab)
            kl_ba_list.append(kl_ba)

            per_prompt.append({
                "probe": probe,
                "kl_ab": float(kl_ab.mean()),
                "kl_ba": float(kl_ba.mean()),
                "symmetric": float((kl_ab.mean() + kl_ba.mean()) / 2),
            })

        mean_kl_ab = float(np.mean([kl.mean() for kl in kl_ab_list]))
        mean_kl_ba = float(np.mean([kl.mean() for kl in kl_ba_list]))
        symmetric = (mean_kl_ab + mean_kl_ba) / 2

        return MetricResult(
            name=self.name,
            level=self.level,
            value=symmetric,
            details={
                "kl_ab": mean_kl_ab,
                "kl_ba": mean_kl_ba,
                "per_prompt": per_prompt,
            },
        )

    @classmethod
    def requirements(cls) -> dict[str, bool]:
        return {"logits": True, "hidden_states": False, "generations": False}


def _kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> np.ndarray:
    """KL(P || Q) per position. Shape: (seq_len, vocab) -> (seq_len,)."""
    log_p = torch.log_softmax(logits_p, dim=-1)
    log_q = torch.log_softmax(logits_q, dim=-1)
    p = torch.exp(log_p)
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return kl.numpy()
