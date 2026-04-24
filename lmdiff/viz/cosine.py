"""Direction-similarity heatmap (raw or selective) for a GeoResult.

Wraps ``plot_direction_heatmap`` with a GeoResult-aware signature and the
v0.2.3 paper-grade defaults (DPI, transparent background off for paper
embedding, fixed value range).
"""
from __future__ import annotations

from pathlib import Path

from lmdiff.geometry import GeoResult
from lmdiff.viz._loaders import _load_geo
from lmdiff.viz._style import DEFAULT_DPI


def _plot_cosine_kind(
    georesult: GeoResult | dict | str | Path,
    output_path: str | Path,
    *,
    kind: str,
    figsize: tuple[float, float] | None = None,
    dpi: int = DEFAULT_DPI,
    variant_order: list[str] | None = None,
) -> Path:
    geo = _load_geo(georesult)
    if kind == "raw":
        matrix = geo.cosine_matrix
        title = "Direction similarity (raw cosine)"
        cbar_label = "cos(δ_i, δ_j)"
    elif kind == "selective":
        if not geo.selective_cosine_matrix:
            raise ValueError(
                "selective_cosine_matrix is empty (legacy v1 GeoResult); "
                "regenerate from a v2+ analyze() call."
            )
        matrix = geo.selective_cosine_matrix
        title = "Selective cosine (Pearson r, mean-removed)"
        cbar_label = "Pearson r over (δ_i − ⟨δ_i⟩, δ_j − ⟨δ_j⟩)"
    else:
        raise ValueError(f"kind must be 'raw' or 'selective', got {kind!r}")

    names = list(variant_order) if variant_order else list(geo.variant_names)
    n = len(names)
    if n == 0:
        raise ValueError("no variants to plot")
    for a in names:
        if a not in matrix:
            raise ValueError(f"cosine_matrix missing row {a!r}")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import Normalize
    except ImportError as exc:
        raise ImportError(
            "matplotlib required. Install with: pip install lmdiff-kit[viz]"
        ) from exc

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.full((n, n), np.nan, dtype=float)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            v = matrix[a].get(b)
            arr[i, j] = float("nan") if v is None else float(v)

    if figsize is None:
        figsize = (max(5.5, 0.7 * n + 4), max(4.5, 0.7 * n + 3))

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d0d0d0")
    norm = Normalize(vmin=-1.0, vmax=1.0)
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    vmax_abs = 1.0
    for i in range(n):
        for j in range(n):
            v = arr[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="#555")
                continue
            color = "white" if abs(v) >= vmax_abs * 0.55 else "black"
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=9, color=color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def plot_cosine_heatmap_raw(
    georesult: GeoResult | dict | str | Path,
    output_path: str | Path,
    *,
    figsize: tuple[float, float] | None = None,
    dpi: int = DEFAULT_DPI,
    variant_order: list[str] | None = None,
) -> Path:
    """Render the raw cosine-similarity matrix as a square heatmap."""
    return _plot_cosine_kind(
        georesult, output_path, kind="raw",
        figsize=figsize, dpi=dpi, variant_order=variant_order,
    )


def plot_cosine_heatmap_selective(
    georesult: GeoResult | dict | str | Path,
    output_path: str | Path,
    *,
    figsize: tuple[float, float] | None = None,
    dpi: int = DEFAULT_DPI,
    variant_order: list[str] | None = None,
) -> Path:
    """Render the selective (mean-removed Pearson r) cosine matrix."""
    return _plot_cosine_kind(
        georesult, output_path, kind="selective",
        figsize=figsize, dpi=dpi, variant_order=variant_order,
    )


__all__ = ["plot_cosine_heatmap_raw", "plot_cosine_heatmap_selective"]
