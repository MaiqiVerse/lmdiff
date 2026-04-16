from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lmdiff.tasks.base import BaseEvaluator, Task, TaskResult
from lmdiff.tasks.evaluators import ContainsAnswer

if TYPE_CHECKING:
    from lmdiff.engine import InferenceEngine
    from lmdiff.probes.loader import ProbeSet


@dataclass(frozen=True)
class DomainRadarResult:
    """Per-domain measurement for one engine."""
    domain: str
    n_probes: int
    accuracy: float
    bd_vs_baseline: float | None = None


@dataclass
class RadarResult:
    """Capability radar result for a pair of configs (or single config)."""
    engine_a_name: str
    engine_b_name: str | None
    domains: list[str]
    a_by_domain: dict[str, DomainRadarResult]
    b_by_domain: dict[str, DomainRadarResult] | None
    bd_by_domain: dict[str, float] | None
    bd_healthy_by_domain: dict[str, float | None] | None
    degeneracy_rates: dict[str, dict[str, float]] | None
    metadata: dict = field(default_factory=dict)

    def summary_table(self) -> list[dict]:
        """Flat list of rows for report rendering."""
        rows: list[dict] = []
        for d in self.domains:
            row: dict[str, Any] = {
                "domain": d,
                "n_probes": self.a_by_domain[d].n_probes,
                "accuracy_a": self.a_by_domain[d].accuracy,
            }
            if self.b_by_domain is not None:
                row["accuracy_b"] = self.b_by_domain[d].accuracy
                row["delta_acc"] = row["accuracy_b"] - row["accuracy_a"]
            if self.bd_by_domain is not None:
                row["bd"] = self.bd_by_domain[d]
            if self.bd_healthy_by_domain is not None:
                row["bd_healthy"] = self.bd_healthy_by_domain[d]
            if self.degeneracy_rates is not None:
                row["degen_a"] = self.degeneracy_rates[d]["a"]
                row["degen_b"] = self.degeneracy_rates[d]["b"]
            rows.append(row)
        return rows


class CapabilityRadar:
    """Multi-domain capability + distribution comparison.

    Runs task evaluation (accuracy) AND behavioral distance (BD)
    per domain for a pair of configs. Two views of the same underlying
    generations.
    """

    def __init__(
        self,
        probes: ProbeSet,
        evaluator: BaseEvaluator | None = None,
        max_new_tokens: int = 16,
    ) -> None:
        self.probes = probes
        self.evaluator = evaluator or ContainsAnswer()
        self.max_new_tokens = max_new_tokens

        domains = probes.domains
        if len(domains) < 2:
            raise ValueError(
                f"CapabilityRadar requires at least 2 domains, got {len(domains)}: {domains}"
            )

    def _run_task_for_domain(
        self, domain: str, domain_probes: ProbeSet, engine: InferenceEngine,
    ) -> TaskResult:
        task = Task(
            name=f"radar_{domain}",
            probes=domain_probes,
            evaluator=self.evaluator,
            max_new_tokens=self.max_new_tokens,
        )
        return task.run(engine)

    def run_single(self, engine: InferenceEngine) -> RadarResult:
        """Accuracy-only radar for one engine."""
        by_domain = self.probes.by_domain()
        domains = sorted(by_domain.keys())

        a_results: dict[str, DomainRadarResult] = {}
        for d in domains:
            tr = self._run_task_for_domain(d, by_domain[d], engine)
            a_results[d] = DomainRadarResult(
                domain=d,
                n_probes=tr.n_probes,
                accuracy=tr.accuracy,
            )

        return RadarResult(
            engine_a_name=engine.model_name,
            engine_b_name=None,
            domains=domains,
            a_by_domain=a_results,
            b_by_domain=None,
            bd_by_domain=None,
            bd_healthy_by_domain=None,
            degeneracy_rates=None,
        )

    def run_pair(
        self, engine_a: InferenceEngine, engine_b: InferenceEngine,
    ) -> RadarResult:
        """Full radar: accuracy per engine + BD per domain."""
        from lmdiff.metrics.output.behavioral_distance import BehavioralDistance

        by_domain = self.probes.by_domain()
        domains = sorted(by_domain.keys())
        bd_metric = BehavioralDistance()

        a_results: dict[str, DomainRadarResult] = {}
        b_results: dict[str, DomainRadarResult] = {}
        bd_by_domain: dict[str, float] = {}
        bd_healthy_by_domain: dict[str, float | None] = {}
        degeneracy_rates: dict[str, dict[str, float]] = {}

        for d in domains:
            domain_probes = by_domain[d]

            tr_a = self._run_task_for_domain(d, domain_probes, engine_a)
            tr_b = self._run_task_for_domain(d, domain_probes, engine_b)

            bd_result = bd_metric.compute(
                engine_a, engine_b, domain_probes.texts,
                max_new_tokens=self.max_new_tokens,
            )

            a_results[d] = DomainRadarResult(
                domain=d,
                n_probes=tr_a.n_probes,
                accuracy=tr_a.accuracy,
                # bd_vs_baseline deliberately None: BD is symmetric,
                # lives in top-level bd_by_domain only.
            )
            b_results[d] = DomainRadarResult(
                domain=d,
                n_probes=tr_b.n_probes,
                accuracy=tr_b.accuracy,
            )

            bd_by_domain[d] = bd_result.value
            bd_healthy_by_domain[d] = bd_result.details.get("bd_healthy")
            degeneracy_rates[d] = {
                "a": bd_result.details.get("degeneracy_rate_a", 0.0),
                "b": bd_result.details.get("degeneracy_rate_b", 0.0),
            }

        return RadarResult(
            engine_a_name=engine_a.model_name,
            engine_b_name=engine_b.model_name,
            domains=domains,
            a_by_domain=a_results,
            b_by_domain=b_results,
            bd_by_domain=bd_by_domain,
            bd_healthy_by_domain=bd_healthy_by_domain,
            degeneracy_rates=degeneracy_rates,
        )
