"""Orchestrator: render the v0.2.3 paper-grade family-experiment figure set.

Default output is 7 figures named ``01_…`` through ``07_…``. Pass
``which=`` (a list of registry keys) to render a subset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from lmdiff.geometry import GeoResult
from lmdiff.viz._loaders import _load_geo
from lmdiff.viz._style import DEFAULT_DPI
from lmdiff.viz.cosine import (
    plot_cosine_heatmap_raw,
    plot_cosine_heatmap_selective,
)
from lmdiff.viz.normalization_effect import plot_normalization_effect
from lmdiff.viz.normalized_magnitude import plot_normalized_magnitude_heatmap
from lmdiff.viz.pca import plot_pca_scatter_normalized, plot_pca_scatter_raw
from lmdiff.viz.specialization import plot_specialization_heatmap

FIGURE_REGISTRY: dict[str, tuple[str, Callable]] = {
    "cosine_raw":           ("01_cosine_raw.png",             plot_cosine_heatmap_raw),
    "cosine_selective":     ("02_cosine_selective.png",       plot_cosine_heatmap_selective),
    "normalized_magnitude": ("03_normalized_magnitude.png",   plot_normalized_magnitude_heatmap),
    "specialization":       ("04_specialization_zscore.png",  plot_specialization_heatmap),
    "pca_raw":              ("05_pca_scatter_raw.png",        plot_pca_scatter_raw),
    "pca_normalized":       ("06_pca_scatter_normalized.png", plot_pca_scatter_normalized),
    "normalization_effect": ("07_normalization_effect.png",   plot_normalization_effect),
}


def plot_family_figures(
    georesult: GeoResult | dict | str | Path,
    output_dir: str | Path,
    *,
    which: list[str] | None = None,
    variant_order: list[str] | None = None,
    domain_order: list[str] | None = None,
    dpi: int = DEFAULT_DPI,
) -> dict[str, Path]:
    """Render selected paper figures to ``output_dir``.

    Args:
        georesult: GeoResult instance, parsed dict, or path to GeoResult JSON.
        output_dir: directory for the PNGs (created if missing).
        which: subset of ``FIGURE_REGISTRY`` keys to render. ``None`` (default)
            renders all 7. Unknown keys raise ``ValueError``.
        variant_order: row order for heatmaps / legend order for PCA.
            Defaults to ``geo.variant_names``.
        domain_order: column order for the variant × domain heatmaps.
            Defaults to ``DEFAULT_DOMAIN_ORDER`` filtered to present domains.
        dpi: DPI for all rendered PNGs.

    Returns:
        ``{registry_key: absolute_path}`` for every figure that rendered
        successfully. Skipped figures (e.g. selective heatmap on a v1
        GeoResult) print a stderr warning and do not appear in the result.
    """
    geo = _load_geo(georesult)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if which is None:
        keys = list(FIGURE_REGISTRY.keys())
    else:
        unknown = [k for k in which if k not in FIGURE_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown figure key(s): {unknown}. "
                f"Valid keys: {sorted(FIGURE_REGISTRY)}"
            )
        keys = list(which)

    rendered: dict[str, Path] = {}
    import sys
    for key in keys:
        filename, fn = FIGURE_REGISTRY[key]
        path = out_dir / filename
        kwargs: dict = {"dpi": dpi}
        if variant_order is not None:
            kwargs["variant_order"] = variant_order
        if domain_order is not None and key in (
            "normalized_magnitude", "specialization",
        ):
            kwargs["domain_order"] = domain_order
        try:
            out = fn(geo, path, **kwargs)
        except Exception as exc:  # noqa: BLE001 - per-plot isolation
            print(
                f"  [WARN] {key} skipped: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        rendered[key] = Path(out)

    return rendered


__all__ = ["FIGURE_REGISTRY", "plot_family_figures"]
