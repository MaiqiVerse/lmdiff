"""Specialization-fingerprint heatmap: row-wise z-score per variant.

This is the v0.2.3 paper main figure. Each row of the variants × domains
matrix is independently z-scored across its domains, exposing each
variant's training-objective specialization signature (L-023).
"""
from __future__ import annotations

from pathlib import Path

from lmdiff.experiments.family import DEFAULT_DOMAIN_ORDER, TASK_TO_DOMAIN
from lmdiff.geometry import GeoResult
from lmdiff.viz._loaders import _load_geo
from lmdiff.viz._style import DEFAULT_DPI


def plot_specialization_heatmap(
    georesult: GeoResult | dict | str | Path,
    output_path: str | Path,
    *,
    figsize: tuple[float, float] | None = None,
    dpi: int = DEFAULT_DPI,
    variant_order: list[str] | None = None,
    domain_order: list[str] | None = None,
    z_clip: float = 2.0,
) -> Path:
    """Render row-wise z-score of normalized magnitudes (variants × domains).

    Cells use diverging cmap RdBu_r centered at 0; ``z_clip`` sets the
    color saturation bounds (values outside still annotated correctly).
    """
    geo = _load_geo(georesult)
    zscore = geo.magnitudes_specialization_zscore()

    variants = list(variant_order) if variant_order else list(geo.variant_names)
    if domain_order is not None:
        domains = list(domain_order)
    else:
        seen: list[str] = []
        for v in variants:
            for d in zscore.get(v, {}):
                if d not in seen:
                    seen.append(d)
        ordered = [d for d in DEFAULT_DOMAIN_ORDER if d in seen]
        ordered.extend(d for d in seen if d not in ordered)
        domains = ordered

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import Normalize
    except ImportError as exc:
        raise ImportError(
            "matplotlib required. Install with: pip install lmdiff-kit[viz]"
        ) from exc

    n_v, n_d = len(variants), len(domains)
    if n_v == 0 or n_d == 0:
        raise ValueError("no variants or domains to plot")

    arr = np.full((n_v, n_d), np.nan, dtype=float)
    for i, v in enumerate(variants):
        for j, d in enumerate(domains):
            val = zscore.get(v, {}).get(d)
            arr[i, j] = float("nan") if val is None else float(val)

    if figsize is None:
        figsize = (max(6.5, 1.0 * n_d + 3), max(4.0, 0.55 * n_v + 2.5))

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d0d0d0")
    norm = Normalize(vmin=-z_clip, vmax=z_clip)
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, norm=norm, aspect="auto")

    task_lookup: dict[str, str] = {}
    for task, dom in TASK_TO_DOMAIN.items():
        task_lookup.setdefault(dom, task)
    xtick_labels = []
    for d in domains:
        task = task_lookup.get(d)
        xtick_labels.append(f"{task}\n({d})" if task else d)

    ax.set_xticks(range(n_d))
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_v))
    ax.set_yticklabels(variants, fontsize=10)
    ax.set_title(
        "Specialization fingerprint (z-score within each variant)",
        fontsize=11, pad=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("z-score (within variant)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    for i in range(n_v):
        for j in range(n_d):
            v = arr[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="#555")
                continue
            color = "white" if abs(v) >= 1.0 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10,
                    color=color, fontweight="bold")

    fig.tight_layout()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


__all__ = ["plot_specialization_heatmap"]
