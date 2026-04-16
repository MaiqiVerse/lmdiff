from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modeldiff.metrics.base import BaseMetric, MetricLevel, MetricResult
from modeldiff.tokenizer_utils import bpb_from_ce, tokenizers_equivalent

if TYPE_CHECKING:
    from modeldiff.engine import InferenceEngine


class BehavioralDistance(BaseMetric):
    """Symmetric behavioral distance with self-entropy baseline subtraction.

    BD(A, B) = ½[CE(A,B) − CE(B,B)] + ½[CE(B,A) − CE(A,A)]
    Asymmetry = [CE(B,A) − CE(A,A)] − [CE(A,B) − CE(B,B)]

    Code variable naming: ce_XY where X = scoring engine, Y = output owner.
    So ce_ab = engine_b scores A's output = CE(B,A) in the formula above.

    Assumes greedy decoding (n_samples=1). When tokenizers differ, all CE
    values are BPB-normalized (using continuation byte count) before the formula.
    """

    name = "behavioral_distance"
    level = MetricLevel.OUTPUT

    def compute(
        self,
        engine_a: InferenceEngine,
        engine_b: InferenceEngine,
        probes: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        max_new_tokens = kwargs.get("max_new_tokens", 64)

        same_tok = engine_a.config.shares_tokenizer_with(engine_b.config)
        if same_tok is None:
            same_tok = tokenizers_equivalent(engine_a.tokenizer, engine_b.tokenizer)
        use_bpb = not same_tok

        gen_a = engine_a.generate(probes, n_samples=1, max_new_tokens=max_new_tokens)
        gen_b = engine_b.generate(probes, n_samples=1, max_new_tokens=max_new_tokens)

        outputs_a = [comps[0] for comps in gen_a.completions]
        outputs_b = [comps[0] for comps in gen_b.completions]

        score_aa = engine_a.score(probes, outputs_a)
        score_ab = engine_b.score(probes, outputs_a)
        score_ba = engine_a.score(probes, outputs_b)
        score_bb = engine_b.score(probes, outputs_b)

        per_prompt: list[dict] = []
        bd_sum = 0.0
        asym_sum = 0.0

        for i, probe in enumerate(probes):
            ce_aa = score_aa.cross_entropies[i]
            ce_ab = score_ab.cross_entropies[i]
            ce_ba = score_ba.cross_entropies[i]
            ce_bb = score_bb.cross_entropies[i]

            if use_bpb:
                text_a = outputs_a[i]
                text_b = outputs_b[i]
                n_tok_aa = len(score_aa.token_ids[i])
                n_tok_ab = len(score_ab.token_ids[i])
                n_tok_ba = len(score_ba.token_ids[i])
                n_tok_bb = len(score_bb.token_ids[i])

                ce_aa = bpb_from_ce(ce_aa, n_tok_aa, text_a)
                ce_ab = bpb_from_ce(ce_ab, n_tok_ab, text_a)
                ce_ba = bpb_from_ce(ce_ba, n_tok_ba, text_b)
                ce_bb = bpb_from_ce(ce_bb, n_tok_bb, text_b)

            bd_i = 0.5 * (ce_ab - ce_bb) + 0.5 * (ce_ba - ce_aa)
            asym_i = (ce_ab - ce_aa) - (ce_ba - ce_bb)

            bd_sum += bd_i
            asym_sum += asym_i

            per_prompt.append({
                "probe": probe,
                "ce_aa": ce_aa,
                "ce_ab": ce_ab,
                "ce_ba": ce_ba,
                "ce_bb": ce_bb,
                "bd": bd_i,
                "asymmetry": asym_i,
            })

        n = len(probes)
        bd = bd_sum / n
        asymmetry = asym_sum / n

        mean_ce_aa = sum(p["ce_aa"] for p in per_prompt) / n
        mean_ce_ab = sum(p["ce_ab"] for p in per_prompt) / n
        mean_ce_ba = sum(p["ce_ba"] for p in per_prompt) / n
        mean_ce_bb = sum(p["ce_bb"] for p in per_prompt) / n

        return MetricResult(
            name=self.name,
            level=self.level,
            value=bd,
            details={
                "ce_aa": mean_ce_aa,
                "ce_ab": mean_ce_ab,
                "ce_ba": mean_ce_ba,
                "ce_bb": mean_ce_bb,
                "asymmetry": asymmetry,
                "bpb_normalized": use_bpb,
                "per_prompt": per_prompt,
            },
        )

    @classmethod
    def is_applicable(cls, config_a: Any, config_b: Any) -> bool:
        return True

    @classmethod
    def requirements(cls) -> dict[str, bool]:
        return {"logits": False, "hidden_states": False, "generations": True}
