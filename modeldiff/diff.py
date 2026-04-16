from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modeldiff.config import Config
from modeldiff.engine import InferenceEngine
from modeldiff.metrics.base import BaseMetric, MetricLevel, MetricResult
from modeldiff.metrics.output.behavioral_distance import BehavioralDistance
from modeldiff.metrics.output.token_entropy import TokenEntropy
from modeldiff.metrics.output.token_kl import TokenKL

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


class ModelDiff:
    """Compare two model configurations across metrics."""

    def __init__(
        self,
        config_a: Config,
        config_b: Config,
        prompts: list[str],
        n_samples: int = 5,
    ) -> None:
        self.config_a = config_a
        self.config_b = config_b
        self.prompts = prompts
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

        return DiffReport(
            config_a=self.config_a,
            config_b=self.config_b,
            results=results,
            metadata={
                "level": level,
                "n_probes": len(self.prompts),
                "name_a": self.config_a.display_name,
                "name_b": self.config_b.display_name,
            },
        )
