"""lmdiff: compare language model configurations."""

from lmdiff.config import Config
from lmdiff.diff import DiffReport, FullReport, ModelDiff, PairTaskResult
from lmdiff.engine import (
    ForwardResult,
    GenerationResult,
    HiddenStatesResult,
    InferenceEngine,
)
from lmdiff.geometry import ChangeGeometry, GeoResult
from lmdiff.probes.loader import Probe, ProbeSet
from lmdiff.tasks.base import BaseEvaluator, EvalResult, Task, TaskResult
from lmdiff.tasks.evaluators import (
    ContainsAnswer,
    ExactMatch,
    F1,
    Gsm8kNumberMatch,
    MultipleChoice,
)

__all__ = [
    "Config",
    "ModelDiff",
    "DiffReport",
    "PairTaskResult",
    "FullReport",
    "ChangeGeometry",
    "GeoResult",
    "InferenceEngine",
    "GenerationResult",
    "ForwardResult",
    "HiddenStatesResult",
    "Probe",
    "ProbeSet",
    "Task",
    "TaskResult",
    "EvalResult",
    "BaseEvaluator",
    "ExactMatch",
    "ContainsAnswer",
    "MultipleChoice",
    "F1",
    "Gsm8kNumberMatch",
]
