"""Figures renderer (commits 1.5 + 1.9).

The pipeline channel ``"figures"`` lands here. v0.3.0 supports two tiers:

  - ``tier="applied"`` (default in commit 1.9): renders the three
    application-tier figures — drift+share, direction agreement,
    change size — into ``out_dir`` and returns ``[Path, Path, Path]``.
  - ``tier="paper"``: delegates to the existing v0.2.x
    ``lmdiff.viz.plot_family_figures`` 7-figure paper suite.

matplotlib is imported only when this renderer is invoked, not at module
load. ``import lmdiff.report`` stays matplotlib-free.
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
    tier: str = "applied",
    which: list[str] | None = None,
    variant_order: list[str] | None = None,
    domain_order: list[str] | None = None,
    dpi: int = 200,
    **_unused,
) -> Any:
    """Render the figure suite for ``result`` into ``out_dir``.

    Parameters
    ----------
    tier : str
        ``"applied"`` (default): the three v0.3.0 application figures
        (drift_share_dual, direction_agreement, change_size_bars).
        ``"paper"``: the v0.2.x 7-figure paper suite.
    which, variant_order, domain_order, dpi
        Forwarded to the underlying renderer (``which`` only meaningful
        for ``tier="paper"``).

    Returns
    -------
    ``list[Path]`` for ``tier="applied"`` (in render order: drift_share,
    direction, change_size).

    ``dict[str, Path]`` for ``tier="paper"`` (key → file path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if tier == "applied":
        from lmdiff.viz.change_size import render_change_size
        from lmdiff.viz.direction import render_direction
        from lmdiff.viz.drift_share import render_drift_share

        return [
            render_drift_share(
                result, out_dir / "drift_share_dual.png",
                variant_order=variant_order, domain_order=domain_order, dpi=dpi,
            ),
            render_direction(
                result, out_dir / "direction_agreement.png",
                variant_order=variant_order, dpi=dpi,
            ),
            render_change_size(
                result, out_dir / "change_size_bars.png",
                variant_order=variant_order, dpi=dpi,
            ),
        ]

    if tier == "paper":
        from lmdiff.viz.family_figures import plot_family_figures
        return plot_family_figures(
            result,
            out_dir,
            which=which,
            variant_order=variant_order,
            domain_order=domain_order,
            dpi=dpi,
        )

    raise ValueError(
        f"Unknown tier {tier!r}; expected 'applied' or 'paper'"
    )


__all__ = ["render"]
