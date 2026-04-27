"""Figures renderer (commit 1.5).

Delegates to the existing v0.2.x ``lmdiff.viz.plot_family_figures`` 7-figure
paper-grade suite. Application-tier dual-view figures
(drift+share, direction agreement, change size) arrive in commits
1.8-1.9 — they will live alongside this delegator, both reachable from
the pipeline.

matplotlib is imported only when this renderer is invoked, not at
module load. ``import lmdiff.report`` stays matplotlib-free.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lmdiff.geometry import GeoResult


def render(
    result: "GeoResult",
    *,
    findings: tuple = (),  # noqa: ARG001
    tables: dict | None = None,  # noqa: ARG001
    out_dir: str | Path,
    which: list[str] | None = None,
    variant_order: list[str] | None = None,
    domain_order: list[str] | None = None,
    dpi: int = 200,
    **_unused,
) -> dict[str, Path]:
    """Render the v0.2.x 7-figure paper suite for ``result`` into ``out_dir``.

    Returns ``{key: path}`` for every figure that rendered. Plots whose
    required GeoResult fields are missing skip with a stderr warning, not
    an exception.
    """
    from lmdiff.viz.family_figures import plot_family_figures

    return plot_family_figures(
        result,
        Path(out_dir),
        which=which,
        variant_order=variant_order,
        domain_order=domain_order,
        dpi=dpi,
    )


__all__ = ["render"]
