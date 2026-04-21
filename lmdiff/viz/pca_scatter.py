"""PC-space scatter of variants from a PCAResult.

Base is implicitly at the origin: δ = 0 for base-vs-base by construction
in ChangeGeometry.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from lmdiff.geometry import PCAResult


def plot_pca_scatter(
    pca_result: "PCAResult",
    *,
    out_path: str | Path,
    title: str = "Change geometry (PCA)",
    show_base: bool = True,
    variant_colors: dict[str, str] | None = None,
    annotate_labels: bool = True,
) -> str:
    """2D scatter of variants in PC-space.

    Args:
        pca_result: output of GeoResult.pca_map(). n_components must be >= 2.
        out_path: output PNG path (parent dirs created).
        title: figure title.
        show_base: if True, draw base as a black cross at the origin.
        variant_colors: optional {name: color_str} override.
        annotate_labels: overlay variant name next to each point.

    Returns:
        Absolute path (str) of the written PNG.

    Raises:
        ImportError: matplotlib not installed.
        ValueError: n_components < 2 or no variants.
    """
    if pca_result.n_components < 2:
        raise ValueError(
            f"pca_scatter requires n_components >= 2; "
            f"got pca_result.n_components={pca_result.n_components}"
        )
    if pca_result.n_variants < 1:
        raise ValueError("pca_result has no variants")

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "matplotlib required for pca_scatter. "
            "Install with: pip install lmdiff-kit[viz]"
        ) from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="none")
    ax.set_facecolor("none")

    default_colors = plt.cm.tab10.colors if hasattr(plt.cm.tab10, "colors") else None

    xs: list[float] = []
    ys: list[float] = []
    for i, (name, coord) in enumerate(pca_result.coords.items()):
        x = float(coord[0])
        y = float(coord[1])
        xs.append(x)
        ys.append(y)

        color = None
        if variant_colors is not None and name in variant_colors:
            color = variant_colors[name]
        elif default_colors is not None:
            color = default_colors[i % len(default_colors)]

        ax.scatter([x], [y], s=140, color=color, edgecolor="black",
                   linewidth=0.8, label=name, zorder=3)
        if annotate_labels:
            ax.annotate(
                name, (x, y),
                xytext=(6, 6), textcoords="offset points",
                fontsize=10,
            )

    if show_base:
        ax.scatter([0.0], [0.0], s=120, marker="x", color="black",
                   linewidth=2, label="base", zorder=4)
        if annotate_labels:
            ax.annotate(
                "base", (0.0, 0.0),
                xytext=(6, -12), textcoords="offset points",
                fontsize=10, color="#333",
            )

    ax.axhline(0, color="#888", linewidth=0.6, alpha=0.5)
    ax.axvline(0, color="#888", linewidth=0.6, alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Axis labels include explained variance
    evr = pca_result.explained_variance_ratio
    pc1_pct = evr[0] * 100 if len(evr) > 0 else 0.0
    pc2_pct = evr[1] * 100 if len(evr) > 1 else 0.0
    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}% var)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)

    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return str(out_path.resolve())
