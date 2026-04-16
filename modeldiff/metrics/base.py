from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from modeldiff.engine import InferenceEngine


class MetricLevel(Enum):
    OUTPUT = "output"
    CALIBRATION = "calibration"
    REPRESENTATION = "representation"
    TRAJECTORY = "trajectory"
    CAUSAL = "causal"


@dataclass
class MetricResult:
    name: str
    level: MetricLevel
    value: float | dict | np.ndarray
    details: dict | None = None
    metadata: dict | None = None


class BaseMetric(ABC):
    name: str
    level: MetricLevel

    @abstractmethod
    def compute(
        self,
        engine_a: InferenceEngine,
        engine_b: InferenceEngine,
        probes: list[str],
        **kwargs: Any,
    ) -> MetricResult: ...

    @classmethod
    def is_applicable(cls, config_a: Any, config_b: Any) -> bool:
        return True

    @classmethod
    def requirements(cls) -> dict[str, bool]:
        return {"logits": False, "hidden_states": False, "generations": False}
