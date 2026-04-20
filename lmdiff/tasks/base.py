from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lmdiff.engine import InferenceEngine
    from lmdiff.probes.loader import ProbeSet


@dataclass(frozen=True)
class EvalResult:
    """Single-probe evaluation outcome."""
    probe_id: str
    output: str
    expected: str | None
    correct: bool
    score: float
    metadata: dict = field(default_factory=dict)


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
        """Return (correct, score, extra_metadata)."""


@dataclass
class TaskResult:
    """Result of running a Task on one engine against one ProbeSet."""
    task_name: str
    engine_name: str
    probe_set_name: str | None
    n_probes: int
    n_correct: int
    accuracy: float
    per_probe: list[EvalResult]
    per_domain: dict[str, dict]
    metadata: dict = field(default_factory=dict)

    def get(self, probe_id: str) -> EvalResult | None:
        for r in self.per_probe:
            if r.probe_id == probe_id:
                return r
        return None


class Task:
    """Pairs a ProbeSet with an evaluator and a generation config.

    Tasks use engines directly (generate + evaluate). They do NOT call
    metrics — the comparison between configs is the caller's job.
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

            meta: dict[str, Any] = {}
            if not output.strip():
                meta["empty_output"] = True
                correct, score = False, 0.0
                eval_meta: dict = {}
            else:
                correct, score, eval_meta = self.evaluator.evaluate(
                    output, probe.expected, probe.metadata,
                )

            per_probe.append(EvalResult(
                probe_id=probe.id,
                output=output,
                expected=probe.expected,
                correct=correct,
                score=score,
                metadata={**meta, **eval_meta},
            ))

        n_correct = sum(r.correct for r in per_probe)
        n_probes = len(per_probe)

        domain_groups: dict[str, list[EvalResult]] = {}
        for r, probe in zip(per_probe, self.probes):
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
            task_name=self.name,
            engine_name=engine.model_name,
            probe_set_name=self.probes.name,
            n_probes=n_probes,
            n_correct=n_correct,
            accuracy=n_correct / n_probes if n_probes > 0 else 0.0,
            per_probe=per_probe,
            per_domain=per_domain,
        )
