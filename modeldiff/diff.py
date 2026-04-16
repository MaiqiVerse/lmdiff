from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from modeldiff.config import Config
from modeldiff.engine import InferenceEngine
from modeldiff.metrics.base import BaseMetric, MetricLevel, MetricResult
from modeldiff.metrics.output.behavioral_distance import BehavioralDistance
from modeldiff.metrics.output.token_entropy import TokenEntropy
from modeldiff.metrics.output.token_kl import TokenKL
from modeldiff.probes.loader import ProbeSet

if TYPE_CHECKING:
    from modeldiff.tasks.base import Task, TaskResult
    from modeldiff.tasks.capability_radar import RadarResult

_OUTPUT_METRICS: list[type[BaseMetric]] = [BehavioralDistance, TokenEntropy, TokenKL]

_METRICS_BY_LEVEL: dict[str, list[type[BaseMetric]]] = {
    "output": _OUTPUT_METRICS,
}


@dataclass
class DiffReport:
    config_a: Config
    config_b: Config
    results: list[MetricResult]
    metadata: dict = field(default_factory=dict)

    def get(self, name: str) -> MetricResult | None:
        for r in self.results:
            if r.name == name:
                return r
        return None


@dataclass
class PairTaskResult:
    """Result of running one Task on both engines."""
    task_name: str
    result_a: TaskResult
    result_b: TaskResult
    delta_accuracy: float
    per_domain_delta: dict[str, float]
    metadata: dict = field(default_factory=dict)


@dataclass
class FullReport:
    """Combines metric-level DiffReport with task-level results."""
    config_a: Config
    config_b: Config
    diff_report: DiffReport | None
    task_results: list[PairTaskResult]
    radar_result: RadarResult | None = None
    metadata: dict = field(default_factory=dict)


class ModelDiff:
    """Compare two model configurations across metrics."""

    def __init__(
        self,
        config_a: Config,
        config_b: Config,
        prompts: list[str] | ProbeSet,
        n_samples: int = 5,
    ) -> None:
        self.config_a = config_a
        self.config_b = config_b

        if isinstance(prompts, ProbeSet):
            self.probe_set = prompts
        else:
            self.probe_set = ProbeSet.from_list(prompts)

        self.prompts = self.probe_set.texts
        self.n_samples = n_samples
        self._engine_a: InferenceEngine | None = None
        self._engine_b: InferenceEngine | None = None

    @property
    def engine_a(self) -> InferenceEngine:
        if self._engine_a is None:
            self._engine_a = InferenceEngine(self.config_a)
        return self._engine_a

    @property
    def engine_b(self) -> InferenceEngine:
        if self._engine_b is None:
            self._engine_b = InferenceEngine(self.config_b)
        return self._engine_b

    def run(
        self,
        level: str = "output",
        metrics: list[type[BaseMetric]] | None = None,
        **kwargs: Any,
    ) -> DiffReport:
        """Run all applicable metrics at the given level."""
        if metrics is None:
            metric_classes = _METRICS_BY_LEVEL.get(level, [])
        else:
            metric_classes = metrics

        results: list[MetricResult] = []
        for cls in metric_classes:
            if not cls.is_applicable(self.config_a, self.config_b):
                continue
            metric = cls()
            result = metric.compute(
                self.engine_a, self.engine_b, self.prompts, **kwargs,
            )
            results.append(result)

        meta: dict[str, Any] = {
            "level": level,
            "n_probes": len(self.prompts),
            "name_a": self.config_a.display_name,
            "name_b": self.config_b.display_name,
        }
        if self.probe_set.name:
            meta["probe_set_name"] = self.probe_set.name
        if self.probe_set.version:
            meta["probe_set_version"] = self.probe_set.version

        return DiffReport(
            config_a=self.config_a,
            config_b=self.config_b,
            results=results,
            metadata=meta,
        )

    def run_task(self, task: Task) -> PairTaskResult:
        """Run a single Task on both engines and return paired results."""
        result_a = task.run(self.engine_a)
        result_b = task.run(self.engine_b)

        per_domain_delta: dict[str, float] = {}
        for d in result_a.per_domain:
            acc_a = result_a.per_domain[d]["accuracy"]
            acc_b = result_b.per_domain.get(d, {}).get("accuracy", 0.0)
            per_domain_delta[d] = acc_b - acc_a

        return PairTaskResult(
            task_name=task.name,
            result_a=result_a,
            result_b=result_b,
            delta_accuracy=result_b.accuracy - result_a.accuracy,
            per_domain_delta=per_domain_delta,
        )

    def run_tasks(self, tasks: list[Task]) -> FullReport:
        """Run multiple tasks on both engines."""
        task_results = [self.run_task(t) for t in tasks]
        return FullReport(
            config_a=self.config_a,
            config_b=self.config_b,
            diff_report=None,
            task_results=task_results,
            metadata={
                "name_a": self.config_a.display_name,
                "name_b": self.config_b.display_name,
                "n_tasks": len(tasks),
            },
        )

    def run_radar(
        self,
        probes: ProbeSet | None = None,
        evaluator: Any = None,
        max_new_tokens: int = 16,
    ) -> RadarResult:
        """Convenience: run CapabilityRadar on both engines."""
        from modeldiff.tasks.capability_radar import CapabilityRadar

        radar = CapabilityRadar(
            probes=probes or self.probe_set,
            evaluator=evaluator,
            max_new_tokens=max_new_tokens,
        )
        return radar.run_pair(self.engine_a, self.engine_b)
