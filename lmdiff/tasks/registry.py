"""The evaluator registry — one place that maps a name to a rule.

Before v0.4.4 this existed twice, partially, in two modules: `cli.py`'s
`_EVALUATOR_MAP` (three of the five) and `experiments/family.py`'s
`GENERATE_EVALUATORS` (eight task names onto two classes). Neither
listed all five, neither knew about the other, and nothing could answer
"which evaluators are there" — which is why PHASE_PLAN §5.4 specified a
registry dispatching on metrics that do not exist. There was nothing
addressable to dispatch on.

Keys are each evaluator's own ``name`` attribute rather than a
hand-written list, so the registry cannot drift from the classes it
indexes. Adding an evaluator with a ``name`` registers it; there is no
second place to update.
"""
from __future__ import annotations

from lmdiff.tasks.base import BaseEvaluator
from lmdiff.tasks.evaluators import (
    F1,
    ContainsAnswer,
    ExactMatch,
    Gsm8kNumberMatch,
    MultipleChoice,
)

__all__ = [
    "EVALUATOR_REGISTRY",
    "KNOWN_SCORINGS",
    "get_evaluator",
]


_EVALUATOR_CLASSES: tuple[type[BaseEvaluator], ...] = (
    ExactMatch,
    ContainsAnswer,
    MultipleChoice,
    F1,
    Gsm8kNumberMatch,
)

EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {
    cls.name: cls for cls in _EVALUATOR_CLASSES
}
"""``scoring`` value → evaluator class. Derived from ``cls.name``."""

KNOWN_SCORINGS: frozenset[str] = frozenset(EVALUATOR_REGISTRY)
"""The vocabulary a probe's ``scoring`` field is checked against.

``scoring`` is deliberately **not** a closed enum. It names an evaluator,
and the registry is the vocabulary — a probe set naming one this version
does not ship is still worth running under the caller's fallback rule.
``ProbeSet.from_json`` warns once at load; nothing raises.
"""


def get_evaluator(scoring: str | None) -> BaseEvaluator | None:
    """Instantiate the evaluator named by ``scoring``, or ``None``.

    ``None`` for an unknown name is the whole contract: callers fall
    back rather than branching on membership, which keeps the lookup
    total and means an unrecognised value degrades instead of failing.
    """
    if not scoring:
        return None
    cls = EVALUATOR_REGISTRY.get(scoring)
    return cls() if cls is not None else None
