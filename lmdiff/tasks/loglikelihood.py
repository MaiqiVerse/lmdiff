"""Loglikelihood-based multiple-choice accuracy.

For each probe, score every choice as a continuation, pick the one with
the lowest cross-entropy (optionally byte-length normalized, per lm-eval's
acc_norm convention), and compare to the gold correct_index.

Requires each probe to have:
    - metadata["choices"]: list[str]  (populated by from_lm_eval for MC tasks)
    - metadata["correct_index"]: int

Zero-coupled with metrics. Uses engine.score() directly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from lmdiff.tasks.base import EvalResult, TaskResult

if TYPE_CHECKING:
    from lmdiff.engine import InferenceEngine
    from lmdiff.probes.loader import ProbeSet


def loglikelihood_accuracy(
    probes: "ProbeSet",
    engine: "InferenceEngine",
    task_name: str = "loglikelihood_choice",
    normalize: bool = True,
) -> TaskResult:
    """Score each probe's choices via CE; pick argmin; compare to gold.

    Args:
        probes: ProbeSet where every probe has metadata["choices"] (list[str])
                and metadata["correct_index"] (int). Raises ValueError otherwise.
        engine: InferenceEngine to score with.
        task_name: name to tag the TaskResult with.
        normalize: if True, divide per-choice CE by the UTF-8 byte length of
                   that choice — matches lm-eval's acc_norm. If False, uses
                   raw per-token CE (matches acc).

    Returns:
        TaskResult with per-probe EvalResult. Each EvalResult's output is
        the model's predicted choice text; score is 1.0/0.0 for correct/wrong.
    """
    per_probe: list[EvalResult] = []
    for probe in probes:
        choices = probe.metadata.get("choices")
        correct_idx = probe.metadata.get("correct_index")
        if not isinstance(choices, list) or not isinstance(correct_idx, int):
            raise ValueError(
                f"probe {probe.id}: loglikelihood_accuracy requires both "
                f"metadata['choices'] (list[str]) and metadata['correct_index'] (int); "
                f"got choices={type(choices).__name__}, correct_index={type(correct_idx).__name__}"
            )
        if not 0 <= correct_idx < len(choices):
            raise ValueError(
                f"probe {probe.id}: correct_index={correct_idx} out of range "
                f"for {len(choices)} choices"
            )

        prompts = [probe.text] * len(choices)
        score_result = engine.score(prompts, continuations=list(choices))
        ces = list(score_result.cross_entropies)

        if normalize:
            # acc_norm: divide by UTF-8 byte length of each choice
            scored: list[float] = []
            for ce, choice in zip(ces, choices):
                if ce != ce:  # NaN check (NaN != NaN)
                    scored.append(float("inf"))
                    continue
                nb = max(1, len(choice.encode("utf-8")))
                scored.append(ce / nb)
        else:
            scored = [ce if ce == ce else float("inf") for ce in ces]

        predicted_idx = min(range(len(scored)), key=lambda k: scored[k])
        correct = predicted_idx == correct_idx
        per_probe.append(EvalResult(
            probe_id=probe.id,
            output=choices[predicted_idx],
            expected=choices[correct_idx],
            correct=correct,
            score=1.0 if correct else 0.0,
            metadata={
                "predicted_index": predicted_idx,
                "correct_index": correct_idx,
                "per_choice_ce": list(ces),
                "per_choice_score": list(scored),
                "normalize": normalize,
            },
        ))

    n_probes = len(per_probe)
    n_correct = sum(r.correct for r in per_probe)

    domain_groups: dict[str, list[EvalResult]] = {}
    for r, probe in zip(per_probe, probes):
        d = probe.domain or "unknown"
        domain_groups.setdefault(d, []).append(r)
    per_domain: dict[str, dict] = {}
    for d, results in domain_groups.items():
        dc = sum(r.correct for r in results)
        per_domain[d] = {
            "n": len(results),
            "correct": dc,
            "accuracy": dc / len(results) if results else 0.0,
        }

    return TaskResult(
        task_name=task_name,
        engine_name=engine.model_name,
        probe_set_name=probes.name,
        n_probes=n_probes,
        n_correct=n_correct,
        accuracy=n_correct / n_probes if n_probes > 0 else 0.0,
        per_probe=per_probe,
        per_domain=per_domain,
        metadata={"normalize": normalize},
    )
