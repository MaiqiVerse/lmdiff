"""lmdiff: compare language model configurations.

Top-level symbols are resolved lazily via :pep:`562`. ``import lmdiff``
does not import torch / transformers / matplotlib; each heavy dependency
is pulled in the first time you reference a symbol that needs it
(``from lmdiff import ModelDiff``, ``lmdiff.ChangeGeometry(...)``, ...).

This keeps ``lmdiff --help`` and ``lmdiff list-metrics`` fast and lets
the package import succeed on torch-less environments — they only fail
at the moment you reach for a torch-using symbol.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.3.1"

# Map every public symbol to the submodule that defines it. __getattr__
# below will resolve the target lazily on first access and then cache it
# in the module globals() so repeated access is free.
_LAZY: dict[str, str] = {
    # v0.3.0 Config + sub-specs (lightweight, no torch). The pre-v0.3.0
    # Config in lmdiff.config remains importable as `from lmdiff.config
    # import Config` until the deprecation shim in commit 1.1.
    "Config": "lmdiff._config",
    "AdapterSpec": "lmdiff._config",
    "QuantSpec": "lmdiff._config",
    "PruneSpec": "lmdiff._config",
    "DecodeSpec": "lmdiff._config",
    "ICLExample": "lmdiff._config",
    "Message": "lmdiff._config",
    "KVCacheSpec": "lmdiff._config",
    "SteeringSpec": "lmdiff._config",
    # v0.3.0 Engine protocol + canonical HFEngine + result/error classes.
    # All lightweight at import; HFEngine instantiation pulls torch lazily.
    "Engine": "lmdiff._engine",
    "HFEngine": "lmdiff._engine",
    "ScoreResult": "lmdiff._engine",
    "GenerateResult": "lmdiff._engine",
    "HiddenStatesResult": "lmdiff._engine",
    "AttentionWeightsResult": "lmdiff._engine",
    "CapabilityError": "lmdiff._engine",
    "CrossTokenizerError": "lmdiff._engine",
    "MinimalEngine": "lmdiff.engines.minimal",
    # v0.3.0 user-facing API
    "compare": "lmdiff._api",
    "family": "lmdiff._api",
    "load_result": "lmdiff.report.json_report",
    # v0.3.0 commit 1.6 findings (eight types + extractor)
    "Finding": "lmdiff._findings",
    "MostLikeBaseFinding": "lmdiff._findings",
    "BiggestMoveFinding": "lmdiff._findings",
    "DirectionClusterFinding": "lmdiff._findings",
    "DirectionOutlierFinding": "lmdiff._findings",
    "SpecializationPeakFinding": "lmdiff._findings",
    "AccuracyArtifactFinding": "lmdiff._findings",
    "TokenizerMismatchFinding": "lmdiff._findings",
    "BaseAccuracyMissingFinding": "lmdiff._findings",
    "extract_findings": "lmdiff._findings",
    # diff / engine / geometry (pull torch + transformers)
    "DiffReport": "lmdiff.diff",
    "FullReport": "lmdiff.diff",
    "ModelDiff": "lmdiff.diff",
    "PairTaskResult": "lmdiff.diff",
    "ForwardResult": "lmdiff.engine",
    "GenerationResult": "lmdiff.engine",
    "HiddenStatesResult": "lmdiff.engine",
    "InferenceEngine": "lmdiff.engine",
    "ChangeGeometry": "lmdiff.geometry",
    "GeoResult": "lmdiff.geometry",
    # probes
    "Probe": "lmdiff.probes.loader",
    "ProbeSet": "lmdiff.probes.loader",
    # tasks
    "BaseEvaluator": "lmdiff.tasks.base",
    "EvalResult": "lmdiff.tasks.base",
    "Task": "lmdiff.tasks.base",
    "TaskResult": "lmdiff.tasks.base",
    "ContainsAnswer": "lmdiff.tasks.evaluators",
    "ExactMatch": "lmdiff.tasks.evaluators",
    "F1": "lmdiff.tasks.evaluators",
    "Gsm8kNumberMatch": "lmdiff.tasks.evaluators",
    "MultipleChoice": "lmdiff.tasks.evaluators",
    "loglikelihood_accuracy": "lmdiff.tasks.loglikelihood",
    # experiments (transitively pulls engine → torch)
    "DEFAULT_DOMAIN_ORDER": "lmdiff.experiments.family",
    "DEFAULT_MAX_NEW_TOKENS": "lmdiff.experiments.family",
    "DEFAULT_TASKS": "lmdiff.experiments.family",
    "FamilyExperimentResult": "lmdiff.experiments.family",
    "TASK_MAX_NEW_TOKENS": "lmdiff.experiments.family",
    "TASK_TO_DOMAIN": "lmdiff.experiments.family",
    "plot_family_geometry": "lmdiff.experiments.family",
    "resolve_max_new_tokens": "lmdiff.experiments.family",
    "run_family_experiment": "lmdiff.experiments.family",
}


def __getattr__(name: str):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'lmdiff' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY.keys()))


__all__ = [
    # v0.3.0 Config + sub-specs (lmdiff._config)
    "Config",
    "AdapterSpec",
    "QuantSpec",
    "PruneSpec",
    "DecodeSpec",
    "ICLExample",
    "Message",
    "KVCacheSpec",
    "SteeringSpec",
    # v0.3.0 Engine protocol + impls + result / error types
    "Engine",
    "HFEngine",
    "MinimalEngine",
    "ScoreResult",
    "GenerateResult",
    "HiddenStatesResult",
    "AttentionWeightsResult",
    "CapabilityError",
    "CrossTokenizerError",
    # v0.3.0 user-facing API
    "compare",
    "family",
    "load_result",
    # v0.3.0 commit 1.6 findings
    "Finding",
    "MostLikeBaseFinding",
    "BiggestMoveFinding",
    "DirectionClusterFinding",
    "DirectionOutlierFinding",
    "SpecializationPeakFinding",
    "AccuracyArtifactFinding",
    "TokenizerMismatchFinding",
    "BaseAccuracyMissingFinding",
    "extract_findings",
    # v0.2.x carry-over (deprecation shim emits warning at __init__ time)
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
    "loglikelihood_accuracy",
    "DEFAULT_TASKS",
    "DEFAULT_DOMAIN_ORDER",
    "DEFAULT_MAX_NEW_TOKENS",
    "TASK_MAX_NEW_TOKENS",
    "TASK_TO_DOMAIN",
    "FamilyExperimentResult",
    "resolve_max_new_tokens",
    "run_family_experiment",
    "plot_family_geometry",
]

# Type-checkers need eager names; keep this block under TYPE_CHECKING so
# it never runs at runtime.
if TYPE_CHECKING:  # pragma: no cover
    from lmdiff._config import (  # noqa: F401
        AdapterSpec,
        Config,
        DecodeSpec,
        ICLExample,
        KVCacheSpec,
        Message,
        PruneSpec,
        QuantSpec,
        SteeringSpec,
    )
    from lmdiff._engine import (  # noqa: F401
        AttentionWeightsResult,
        CapabilityError,
        CrossTokenizerError,
        Engine,
        GenerateResult,
        HFEngine,
        HiddenStatesResult,
        ScoreResult,
    )
    from lmdiff.engines.minimal import MinimalEngine  # noqa: F401
    from lmdiff._api import compare, family  # noqa: F401
    from lmdiff.diff import (  # noqa: F401
        DiffReport,
        FullReport,
        ModelDiff,
        PairTaskResult,
    )
    from lmdiff.engine import (  # noqa: F401
        ForwardResult,
        GenerationResult,
        HiddenStatesResult,
        InferenceEngine,
    )
    from lmdiff.experiments.family import (  # noqa: F401
        DEFAULT_DOMAIN_ORDER,
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_TASKS,
        FamilyExperimentResult,
        TASK_MAX_NEW_TOKENS,
        TASK_TO_DOMAIN,
        plot_family_geometry,
        resolve_max_new_tokens,
        run_family_experiment,
    )
    from lmdiff.geometry import ChangeGeometry, GeoResult  # noqa: F401
    from lmdiff.probes.loader import Probe, ProbeSet  # noqa: F401
    from lmdiff.tasks.base import (  # noqa: F401
        BaseEvaluator,
        EvalResult,
        Task,
        TaskResult,
    )
    from lmdiff.tasks.evaluators import (  # noqa: F401
        ContainsAnswer,
        ExactMatch,
        F1,
        Gsm8kNumberMatch,
        MultipleChoice,
    )
    from lmdiff.tasks.loglikelihood import loglikelihood_accuracy  # noqa: F401
