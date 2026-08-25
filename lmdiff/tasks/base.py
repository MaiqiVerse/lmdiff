from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lmdiff.engine import InferenceEngine
    from lmdiff.probes.loader import ProbeSet


# ── Unscorable probes (v0.4.4) ───────────────────────────────────────
#
# Two things an evaluator can report, and conflating them is what made
# `MultipleChoice` on `v01` render as a clean 0.0 across all 90 probes:
#
#   * **the evaluator could not apply to this probe at all** — the probe
#     lacks what the rule needs, whatever the model produced. A property
#     of (probe, evaluator), carrying no information about the model.
#   * **the evaluator applied and the output failed** — a bad parse, a
#     failed extraction, an empty completion. A fact about the model,
#     and correctly counted as wrong.
#
# Only the first is *unscorable*. Such probes are excluded from the
# accuracy denominator rather than counted wrong, because counting them
# wrong makes the model look worse for a defect in the probe set —
# "excluded cells are removed, not nulled", applied one layer down.
#
# The vocabulary lives here and evaluators import the constants, so a
# typo is a NameError rather than a silently unmatched string.

UNSCORABLE_NO_EXPECTED = "no_expected"
"""The probe has no ``expected``, and the rule compares against it."""

UNSCORABLE_EMPTY_EXPECTED = "empty_expected"
"""``expected`` is the empty string, which every output contains."""

UNSCORABLE_MISSING_MC_METADATA = "missing_mc_metadata"
"""No ``metadata['correct_index']``, so there is no gold to compare to."""

UNSCORABLE_REASONS: frozenset[str] = frozenset({
    UNSCORABLE_NO_EXPECTED,
    UNSCORABLE_EMPTY_EXPECTED,
    UNSCORABLE_MISSING_MC_METADATA,
})
"""Reason codes meaning *this evaluator cannot judge this probe*.

Deliberately excludes ``no_choice_parsed`` and ``extraction_failed``:
the evaluator did apply, and the output is what failed.
"""


@dataclass(frozen=True)
class EvalResult:
    """Single-probe evaluation outcome."""
    probe_id: str
    output: str
    expected: str | None
    correct: bool
    score: float
    metadata: dict = field(default_factory=dict)
    scorable: bool = True
    """False when the evaluator reported it could not apply (v0.4.4).

    Such probes are excluded from ``TaskResult.accuracy`` rather than
    counted as incorrect. ``correct`` stays ``False`` so existing
    consumers see no new value; read ``scorable`` to tell "wrong" from
    "unjudgeable"."""
    evaluator: str | None = None
    """Which evaluator judged this probe (v0.4.4). ``None`` for results
    constructed directly rather than by ``Task.run``."""


class BaseEvaluator(ABC):
    """Decides whether a model output matches expectation."""
    name: str

    @abstractmethod
    def evaluate(
        self,
        output: str,
        expected: str | None,
        probe_metadata: dict | None = None,
    ) -> tuple[bool, float, dict]:
        """Return (correct, score, extra_metadata).

        To report that this probe cannot be judged at all — as opposed
        to judged and wrong — put one of ``UNSCORABLE_REASONS`` in the
        returned metadata under ``"reason"``.
        """


@dataclass
class TaskResult:
    """Result of running a Task on one engine against one ProbeSet."""
    task_name: str
    engine_name: str
    probe_set_name: str | None
    n_probes: int
    n_correct: int
    accuracy: float | None
    per_probe: list[EvalResult]
    per_domain: dict[str, dict]
    metadata: dict = field(default_factory=dict)
    n_unscorable: int = 0
    """Probes the evaluator could not judge (v0.4.4). Excluded from
    ``accuracy``'s denominator."""

    @property
    def n_scorable(self) -> int:
        """The denominator ``accuracy`` actually rests on."""
        return self.n_probes - self.n_unscorable

    def get(self, probe_id: str) -> EvalResult | None:
        for r in self.per_probe:
            if r.probe_id == probe_id:
                return r
        return None


class Task:
    """Pairs a ProbeSet with an evaluator and a generation config.

    Tasks use engines directly (generate + evaluate). They do NOT call
    metrics — the comparison between configs is the caller's job.

    .. versionchanged:: 0.4.4
       The evaluator is chosen **per probe** from ``probe.scoring``, and
       the constructor's ``evaluator`` is the fallback for probes that
       do not name one. Before this, one evaluator judged an entire
       ProbeSet, which meant a mixed-format set could not be scored
       correctly by any single choice — see PHASE_PLAN §5.1.
    """

    def __init__(
        self,
        name: str,
        probes: ProbeSet,
        evaluator: BaseEvaluator,
        max_new_tokens: int = 32,
    ) -> None:
        self.name = name
        self.probes = probes
        self.evaluator = evaluator
        self.max_new_tokens = max_new_tokens

    def _evaluator_for(self, probe: Any) -> BaseEvaluator:
        """``probe.scoring`` via the registry, else the fallback.

        An unrecognised ``scoring`` falls back rather than raising: the
        registry lookup is total by construction, and a probe set that
        names an evaluator this version does not have is still worth
        scoring by the caller's rule. ``ProbeSet.from_json`` is where an
        unknown value gets warned about, once, at load.
        """
        from lmdiff.tasks.registry import get_evaluator

        scoring = getattr(probe, "scoring", None)
        if not scoring:
            return self.evaluator
        return get_evaluator(scoring) or self.evaluator

    def run(
        self,
        engine: InferenceEngine,
        pre_generated: Any = None,
    ) -> TaskResult:
        """Generate on each probe, evaluate, aggregate.

        Pass pre_generated (a GenerationResult) to reuse outputs from a
        prior engine.generate() call — required when pairing task accuracy
        with BD under sampling decode, so both views share the same samples.
        """
        if pre_generated is not None:
            gen = pre_generated
        else:
            gen = engine.generate(
                self.probes.texts, n_samples=1, max_new_tokens=self.max_new_tokens,
            )

        per_probe: list[EvalResult] = []
        for i, probe in enumerate(self.probes):
            output = gen.completions[i][0]
            evaluator = self._evaluator_for(probe)

            meta: dict[str, Any] = {}
            if not output.strip():
                # An empty completion is a fact about the model, not a
                # defect in the probe — scorable, and wrong.
                meta["empty_output"] = True
                correct, score = False, 0.0
                eval_meta: dict = {}
            else:
                correct, score, eval_meta = evaluator.evaluate(
                    output, probe.expected, probe.metadata,
                )

            per_probe.append(EvalResult(
                probe_id=probe.id,
                output=output,
                expected=probe.expected,
                correct=correct,
                score=score,
                metadata={**meta, **eval_meta},
                scorable=eval_meta.get("reason") not in UNSCORABLE_REASONS,
                evaluator=evaluator.name,
            ))

        n_correct = sum(r.correct for r in per_probe)
        n_probes = len(per_probe)
        n_unscorable = sum(not r.scorable for r in per_probe)

        domain_groups: dict[str, list[EvalResult]] = {}
        for r, probe in zip(per_probe, self.probes):
            d = probe.domain or "unknown"
            domain_groups.setdefault(d, []).append(r)

        per_domain: dict[str, dict] = {}
        for d, results in domain_groups.items():
            dc = sum(r.correct for r in results)
            d_unscorable = sum(not r.scorable for r in results)
            d_scorable = len(results) - d_unscorable
            per_domain[d] = {
                "n": len(results),
                "correct": dc,
                "n_scorable": d_scorable,
                "n_unscorable": d_unscorable,
                "accuracy": (dc / d_scorable) if d_scorable > 0 else None,
            }

        n_scorable = n_probes - n_unscorable
        return TaskResult(
            task_name=self.name,
            engine_name=engine.model_name,
            probe_set_name=self.probes.name,
            n_probes=n_probes,
            n_correct=n_correct,
            # None rather than 0.0 when nothing could be judged. This is
            # a division guard, not a threshold: no minimum-fraction
            # floor is introduced here, and if one is ever wanted it
            # belongs beside `min_valid_fraction` in `_validity`.
            accuracy=(n_correct / n_scorable) if n_scorable > 0 else None,
            per_probe=per_probe,
            per_domain=per_domain,
            n_unscorable=n_unscorable,
        )
