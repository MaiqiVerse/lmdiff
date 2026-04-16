from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modeldiff.probes.loader import ProbeSet


def from_hf_dataset(
    dataset_name: str,
    split: str = "test",
    text_field: str = "question",
    expected_field: str | None = None,
    domain: str | None = None,
    limit: int | None = None,
) -> ProbeSet:
    """Load probes from a HuggingFace dataset. Phase 2."""
    raise NotImplementedError("from_hf_dataset is Phase 2")


def from_lm_eval(task_name: str, limit: int | None = None) -> ProbeSet:
    """Load probes from an lm-eval-harness task. Phase 2."""
    raise NotImplementedError("from_lm_eval is Phase 2")
