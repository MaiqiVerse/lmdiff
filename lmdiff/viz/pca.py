"""PC1–PC2 scatter of variants in raw and normalized probe space.

Wraps ``GeoResult.pca_map(use_normalized=...)`` with the v0.2.3
shared-style markers and the gold-star base-position annotation.
"""
from __future__ import annotations

from pathlib import Path

from lmdiff.geometry import GeoResult
from lmdiff.viz._loaders import _load_geo
from lmdiff.viz._style import BASE_MARKER, DEFAULT_DPI, variant_color, variant_marker


def _plot_pca_kind(
    georesult: GeoResult | dict | str | Path,
    output_path: str | Path,
    *,
    space: str,
    figsize: tuple[float, float] = (6.0, 5.5),
    dpi: int = DEFAULT_DPI,
    variant_order: list[str] | None = None,
) -> Path:
    geo = _load_geo(georesult)
    if space == "raw":
        use_norm = False
        title_suffix = "raw probe space"
    elif space == "normalized":
        use_norm = True
        title_suffix = "per-token normalized space"
    else:
        raise ValueError(f"space must be 'raw' or 'normalized', got {space!r}")

    if len(geo.variant_names) < 2:
        raise ValueError("PCA requires at least 2 variants")

    pca = geo.pca_map(n_components=2, use_normalized=use_norm)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib required. Install with: pip install lmdiff-kit[viz]"
        ) from exc

    variants = list(variant_order) if variant_order else list(geo.variant_names)

    fig, ax = plt.subplots(figsize=figsize)
    # Base = origin in this projection (δ_base ≡ 0 by construction).
    ax.scatter([0], [0], **BASE_MARKER, label=f"base ({geo.base_name})")
    ax.text(0, 0, "  base", fontsize=9, va="center", color="#555")

    for idx, name in enumerate(variants):
        if name not in pca.coords:
            continue
        x, y = pca.coords[name]
        ax.scatter(
            x, y,
            color=variant_color(name, idx), marker=variant_marker(name, idx),
            s=140, edgecolor="black", linewidth=0.8, zorder=3, label=name,
        )
        ax.text(x, y, f"  {name}", fontsize=9, va="center")

    ax.axhline(0, color="#bbbbbb", linewidth=0.6, zorder=1)
    ax.axvline(0, color="#bbbbbb", linewidth=0.6, zorder=1)

    ev = pca.explained_variance_ratio
    ev1 = ev[0] if len(ev) > 0 else 0.0
    ev2 = ev[1] if len(ev) > 1 else 0.0
    ax.set_xlabel(f"PC1 ({ev1*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({ev2*100:.1f}%)", fontsize=10)
    ax.set_title(f"Change geometry PCA — {title_suffix}", fontsize=11, pad=10)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def plot_pca_scatter_raw(
    georesult: GeoResult | dict | str | Path,
    output_path: str | Path,
    *,
    figsize: tuple[float, float] = (6.0, 5.5),
    dpi: int = DEFAULT_DPI,
    variant_order: list[str] | None = None,
) -> Path:
    """PC1–PC2 of variants in raw probe space."""
    return _plot_pca_kind(
        georesult, output_path, space="raw",
        figsize=figsize, dpi=dpi, variant_order=variant_order,
    )


def plot_pca_scatter_normalized(
    georesult: GeoResult | dict | str | Path,
    output_path: str | Path,
    *,
    figsize: tuple[float, float] = (6.0, 5.5),
    dpi: int = DEFAULT_DPI,
    variant_order: list[str] | None = None,
) -> Path:
    """PC1–PC2 of variants in per-token-normalized probe space."""
    return _plot_pca_kind(
        georesult, output_path, space="normalized",
        figsize=figsize, dpi=dpi, variant_order=variant_order,
    )


__all__ = ["plot_pca_scatter_raw", "plot_pca_scatter_normalized"]
