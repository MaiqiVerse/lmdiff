"""N×N direction (cosine) similarity heatmap.

Lazy-imports matplotlib so ``import lmdiff.viz.direction_heatmap`` works
on a machine without matplotlib; only ``plot_direction_heatmap()`` raises.
"""
from __future__ import annotations

import math
from pathlib import Path


def plot_direction_heatmap(
    cosine_matrix: dict[str, dict[str, float]],
    variant_names: list[str],
    *,
    out_path: str | Path,
    title: str = "Direction similarity",
    cmap: str = "RdBu_r",
    value_range: tuple[float, float] = (-1.0, 1.0),
    annotate: bool = True,
) -> str:
    """Render an N×N cosine-similarity heatmap. NaN cells render light gray.

    Args:
        cosine_matrix: {name: {name: float}} — both self-entries (diagonal)
            and off-diagonals included.
        variant_names: row / column order. Must all appear as keys in
            cosine_matrix (both outer and inner level).
        out_path: output PNG path (parent dirs created).
        title: figure title.
        cmap: matplotlib colormap name (default RdBu_r diverges around 0).
        value_range: (vmin, vmax) for the color scale.
        annotate: overlay numeric value in each cell.

    Returns:
        Absolute path (str) of the written PNG.

    Raises:
        ImportError: matplotlib not installed.
        ValueError: empty variant_names, or cosine_matrix missing an entry.
    """
    if not variant_names:
        raise ValueError("variant_names is empty")

    n = len(variant_names)
    # Validate rows first, then per-row columns, so error messages are
    # unambiguous: missing-row beats missing-column when a whole row is
    # absent.
    for a in variant_names:
        if a not in cosine_matrix:
            raise ValueError(f"cosine_matrix missing row {a!r}")
    for a in variant_names:
        for b in variant_names:
            if b not in cosine_matrix[a]:
                raise ValueError(f"cosine_matrix[{a!r}] missing column {b!r}")

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        import numpy as np
        from matplotlib.colors import Normalize  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "matplotlib required for direction_heatmap. "
            "Install with: pip install lmdiff-kit[viz]"
        ) from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.full((n, n), np.nan, dtype=float)
    for i, a in enumerate(variant_names):
        for j, b in enumerate(variant_names):
            v = cosine_matrix[a][b]
            matrix[i, j] = float("nan") if v is None else float(v)

    # Scale figure a bit with N so labels don't overlap
    figsize = (max(6, 0.8 * n + 4), max(5, 0.8 * n + 3))
    fig, ax = plt.subplots(figsize=figsize, facecolor="none")
    ax.set_facecolor("none")

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#d0d0d0")  # NaN cells
    vmin, vmax = value_range
    norm = Normalize(vmin=vmin, vmax=vmax)

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap_obj, norm=norm, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(variant_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(variant_names, fontsize=10)
    ax.set_title(title, fontsize=13, pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    if annotate:
        # Pick text color by absolute value so it reads on both ends of the cmap.
        mid = (vmin + vmax) / 2
        for i in range(n):
            for j in range(n):
                v = matrix[i, j]
                if not math.isfinite(v):
                    text = "n/a"
                    color = "#555"
                else:
                    text = f"{v:+.2f}"
                    color = "white" if abs(v - mid) > (vmax - vmin) / 4 else "black"
                ax.text(
                    j, i, text,
                    ha="center", va="center", fontsize=8, color=color,
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return str(out_path.resolve())
