"""Shared style tokens for the family-figure plot suite.

Centralizes color / marker assignments so every cross-variant figure
identifies the same variant identically. Falls back to ``tab10`` for
unknown variant names.
"""
from __future__ import annotations

VARIANT_COLORS: dict[str, str] = {
    "yarn": "#d62728",
    "long": "#2ca02c",
    "code": "#1f77b4",
    "math": "#9467bd",
}

VARIANT_MARKERS: dict[str, str] = {
    "yarn": "o",
    "long": "s",
    "code": "^",
    "math": "D",
}

FALLBACK_CMAP: str = "tab10"
DEFAULT_DPI: int = 200

BASE_MARKER: dict = dict(
    marker="*", s=300, c="gold", edgecolor="black", linewidth=1.2, zorder=5,
)


def _tab10_color(idx: int) -> str:
    """Hex color from the tab10 palette without importing matplotlib here."""
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    return palette[idx % len(palette)]


def variant_color(name: str, idx: int = 0) -> str:
    """Look up VARIANT_COLORS; fall back to tab10[idx % 10] for unknown variants."""
    return VARIANT_COLORS.get(name, _tab10_color(idx))


def variant_marker(name: str, idx: int = 0) -> str:
    """Look up VARIANT_MARKERS; fall back to a small marker rotation."""
    if name in VARIANT_MARKERS:
        return VARIANT_MARKERS[name]
    fallback = ["o", "s", "^", "D", "v", "<", ">", "P", "X", "h"]
    return fallback[idx % len(fallback)]


def domain_to_tasks_map(metadata: dict | None) -> dict[str, list[str]]:
    """Build a {domain: [task_name, ...]} mapping from a result's metadata.

    Used by figure renderers to surface the domain↔dataset relationship
    in a small note (so a viewer reading "commonsense" knows it came
    from ``hellaswag``, not some generic synonym). Reads
    ``metadata['tasks']`` (the list populated by the v0.3.2 lm_eval
    multi-task probe loader) and groups by ``KNOWN_TASK_DOMAINS[t].domain``.

    Returns an empty dict when ``tasks`` is missing or empty (legacy
    runs, single-domain runs, non-lm_eval probe sources). Renderers
    should treat the empty dict as "no note to render".
    """
    if not metadata:
        return {}
    tasks = metadata.get("tasks") or []
    if not tasks:
        return {}
    # Lazy import — keeps the viz layer torch-free in the import path.
    from lmdiff.probes.adapters import KNOWN_TASK_DOMAINS

    out: dict[str, list[str]] = {}
    for t in tasks:
        info = KNOWN_TASK_DOMAINS.get(t)
        if info is None:
            # Unknown task — bucket it under a placeholder so the user
            # still sees the task name in the legend.
            out.setdefault("unknown", []).append(t)
            continue
        out.setdefault(info.domain, []).append(t)
    return out


__all__ = [
    "VARIANT_COLORS",
    "VARIANT_MARKERS",
    "FALLBACK_CMAP",
    "DEFAULT_DPI",
    "BASE_MARKER",
    "variant_color",
    "variant_marker",
    "domain_to_tasks_map",
]
