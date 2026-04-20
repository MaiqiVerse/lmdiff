from __future__ import annotations

from typing import Any

import torch


def safe_probe_slice(result: Any, i: int) -> slice | None:
    """Extract probe_slice at index i from a ForwardResult, robust to mocks."""
    ps = getattr(result, "probe_slices", None)
    if isinstance(ps, list) and i < len(ps) and isinstance(ps[i], slice):
        return ps[i]
    return None


def probe_predicting_logits(
    logits: torch.Tensor, probe_slice: slice | None,
) -> torch.Tensor:
    """Return the logit positions that predict probe tokens.

    input_ids = [prefix..., probe_0, ..., probe_{L-1}]; probe_slice marks
    the probe span in input_ids. Causal LM: logits[t] predicts input_ids[t+1].
    So predictions of probe tokens live at logits[start-1 : stop-1] when
    start >= 1, or logits[0 : stop-1] when start == 0 (no prior token to
    predict probe_0 from; only L-1 predictions available).

    probe_slice=None → return full logits (mocks or legacy callers).
    """
    if probe_slice is None:
        return logits
    start = probe_slice.start
    stop = probe_slice.stop
    if start >= 1:
        return logits[start - 1 : stop - 1]
    return logits[0 : stop - 1]
