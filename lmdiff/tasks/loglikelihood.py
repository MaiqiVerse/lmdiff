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
    from lmdiff._engine import Engine
    from lmdiff.probes.loader import ProbeSet


def loglikelihood_accuracy(
    probes: "ProbeSet",
    engine: "Engine",
    task_name: str = "loglikelihood_choice",
    normalize: bool = True,
    *,
    prefix_text: str = "",
) -> TaskResult:
    """Score each probe's choices via CE; pick argmin; compare to gold.

    Args:
        probes: ProbeSet where every probe has metadata["choices"] (list[str])
                and metadata["correct_index"] (int). Raises ValueError otherwise.
        engine: Engine (lmdiff._engine Protocol) to score with.
        task_name: name to tag the TaskResult with.
        prefix_text: system_prompt / context material preceding each
                     probe. Must be supplied -- the Protocol's engines are
                     stateless, so an omitted prefix silently scores a
                     different configuration (v0.4.5).
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

        # v0.4.5: one Protocol `score` call per choice, replacing the
        # v0.2.x batch form. `ScoreResult.avg_logprob` is mean(logprobs),
        # so `ce = -avg_logprob` reproduces the legacy
        # `-lp.sum() / n_tokens` exactly; an empty continuation stays NaN
        # rather than becoming avg_logprob's 0.0, which would read as a
        # perfect score.
        ces: list[float] = []
        for choice in choices:
            try:
                sr = engine.score(probe.text, choice, prefix_text=prefix_text)
            except TypeError:
                # Engine without the prefix_text kwarg (mock engines in
                # unit tests). Concatenate instead; real backends take
                # the kwarg and split-tokenize, which is what keeps the
                # probe span byte-aligned (L-030).
                sr = engine.score(prefix_text + probe.text, choice)
            ces.append(
                float("nan") if not sr.tokens else -float(sr.avg_logprob)
            )

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
        engine_name=engine.name,
        probe_set_name=probes.name,
        n_probes=n_probes,
        n_correct=n_correct,
        accuracy=n_correct / n_probes if n_probes > 0 else 0.0,
        per_probe=per_probe,
        per_domain=per_domain,
        metadata={"normalize": normalize},
    )
