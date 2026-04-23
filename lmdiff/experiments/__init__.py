"""High-level experiment workflows that compose lmdiff primitives.

Library equivalents of the runnable scripts under ``scripts/``. Use these
when you want to drive an experiment from Python (or from the CLI) without
re-implementing the load/geometry/accuracy/plot wiring.
"""
from lmdiff.experiments.family import (
    DEFAULT_TASKS,
    FamilyExperimentResult,
    plot_family_geometry,
    run_family_experiment,
)

__all__ = [
    "DEFAULT_TASKS",
    "FamilyExperimentResult",
    "plot_family_geometry",
    "run_family_experiment",
]
