"""High-level experiment workflows that compose lmdiff primitives.

Library equivalents of the runnable scripts under ``scripts/``. Use these
when you want to drive an experiment from Python (or from the CLI) without
re-implementing the load/geometry/accuracy/plot wiring.
"""
from lmdiff.experiments.family import (
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

__all__ = [
    "DEFAULT_DOMAIN_ORDER",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_TASKS",
    "FamilyExperimentResult",
    "TASK_MAX_NEW_TOKENS",
    "TASK_TO_DOMAIN",
    "plot_family_geometry",
    "resolve_max_new_tokens",
    "run_family_experiment",
]
