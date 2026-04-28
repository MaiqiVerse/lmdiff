"""Direction agreement dual-view figure (commit 1.9 of v0.3.0 batch 3).

Direct port of v6 §14.3 ``prototype_cosine.py``. Two cosine matrices
side-by-side:

  - Left:  raw cosine (full direction agreement)
  - Right: selective cosine (mean-removed; probe-specific differences)

Bottom-line panel sources its narrative from the matching
``DirectionClusterFinding`` and ``DirectionOutlierFinding`` from
:func:`lmdiff._findings.extract_findings` (cross-renderer consistency,
v6 §12.6). When the findings are absent (e.g. < 3 variants), the panel
falls back to the in-figure auto-detection so the visualization remains
useful.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lmdiff.geometry import GeoResult


def _find_cluster_outlier(
    variants: list[str],
    cm,
    *,
    t_cluster: float = 0.90,
    t_outlier: float = 0.85,
) -> tuple[list[str], list[tuple[str, float]]]:
    """Greedy max-clique above ``t_cluster``; outliers are variants with
    cos < ``t_outlier`` to every cluster member."""
    import numpy as np
    n = len(variants)
    best: list[int] = []
    for size in range(n, 1, -1):
        for combo in combinations(range(n), size):
            ok = all(
                cm[i, j] > t_cluster
                for i in combo for j in combo if i != j
            )
            if ok:
                best = list(combo)
                break
        if best:
            break
    cluster = [variants[i] for i in best]
    outliers: list[tuple[str, float]] = []
    if best:
        for i in range(n):
            if i in best:
                continue
            cs = [cm[i, j] for j in best]
            if cs and all(c < t_outlier for c in cs):
                outliers.append((variants[i], float(np.mean(cs))))
    return cluster, outliers


def render_direction(
    result: "GeoResult",
    out_path: str | Path,
    *,
    variant_order: list[str] | None = None,
    findings: tuple | None = None,
    dpi: int = 180,
) -> Path:
    """Render the direction-agreement dual-view (raw + selective cosine)."""
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if findings is None:
        findings = result.findings
    findings = tuple(findings or ())

    from lmdiff._findings import (
        DirectionClusterFinding,
        DirectionOutlierFinding,
    )

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
    })

    variants = list(variant_order) if variant_order else sorted(result.variant_names)
    n_v = len(variants)
    if n_v == 0:
        raise ValueError("render_direction: no variants in result")

    # Layout that scales with N. Up to 4 variants the v0.3.0 layout is fine.
    # Beyond that, cells get crowded: per-cell text labels collide with the
    # number, x-tick labels with long names overlap, and the fixed 14×7
    # canvas no longer fits two square heatmaps side-by-side.
    compact_cells = n_v > 4
    max_label_len = max((len(v) for v in variants), default=0)
    rotate_x = 30 if (max_label_len > 8 or n_v > 5) else 0
    # Each heatmap pane gets ~1.6" per cell, capped sensibly.
    pane_in = max(4.0, min(1.6 * n_v, 10.0))
    fig_w = pane_in * 2 + 3.2  # two panes + takeaway column
    fig_h = max(6.5, pane_in + 1.7)

    def _matrix(src: dict) -> "np.ndarray":
        return np.array([
            [src.get(a, {}).get(b, float("nan")) for b in variants]
            for a in variants
        ])

    cm_raw = _matrix(result.cosine_matrix)
    sel_src = result.selective_cosine_matrix or {}
    cm_sel = _matrix(sel_src) if sel_src else cm_raw

    # 5-bucket sequential-from-cool palette (v6 §14.3).
    boundaries = [-1.01, 0.30, 0.70, 0.85, 0.95, 1.01]
    colors_list = ["#1f4e79", "#9ec5e8", "#f2f2f2", "#f5b9a8", "#c0392b"]
    labels_strip = [
        ("#1f4e79", "✗ uncorrelated\n< 0.30", "white"),
        ("#9ec5e8", "↓ weak\n0.30–0.70", "#1f3a52"),
        ("#f2f2f2", "· moderate\n0.70–0.85", "#444444"),
        ("#f5b9a8", "↑ strong\n0.85–0.95", "#5a1f17"),
        ("#c0392b", "✓ near identical\n> 0.95", "white"),
    ]
    cmap = ListedColormap(colors_list)
    norm_cmap = BoundaryNorm(boundaries, cmap.N)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[pane_in, pane_in, 2.6],
        height_ratios=[pane_in, 0.6, 0.7],
        hspace=0.30, wspace=0.22,
        left=0.06, right=0.98, top=0.85, bottom=0.05,
    )
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_sel = fig.add_subplot(gs[0, 1])
    ax_takeaway = fig.add_subplot(gs[0:2, 2])
    ax_legend = fig.add_subplot(gs[2, 0:2])
    ax_takeaway.axis("off")
    ax_legend.axis("off")

    # Cell font sizes shrink as the grid grows so numbers fit cleanly.
    num_fs = 13 if n_v <= 4 else (11 if n_v <= 6 else 9.5)
    lbl_fs = 8.5 if n_v <= 4 else 7.5
    tick_fs = 11 if n_v <= 5 else (10 if n_v <= 7 else 9)

    def _draw(ax, mat, title_top, title_bot):
        ax.imshow(mat, cmap=cmap, norm=norm_cmap)
        for i in range(n_v):
            for j in range(n_v):
                v = mat[i, j]
                if i == j:
                    num_str, lbl = "—", "self"
                    num_color, txt_color = "#888", "#999"
                else:
                    if v > 0.95:
                        lbl, num_color, txt_color = "✓ near identical", "white", "white"
                    elif v > 0.85:
                        lbl, num_color, txt_color = "↑ strong", "#5a1f17", "#5a1f17"
                    elif v > 0.70:
                        lbl, num_color, txt_color = "· moderate", "#444", "#666"
                    elif v > 0.30:
                        lbl, num_color, txt_color = "↓ weak", "#1f3a52", "#1f3a52"
                    else:
                        lbl, num_color, txt_color = "✗ uncorrelated", "white", "white"
                    num_str = f"{v:+.2f}" if v == v else "n/a"
                if compact_cells:
                    # No room for the sub-label without overlap; the cell's
                    # color band already encodes the bucket and the legend
                    # strip below explains it.
                    ax.text(j, i, num_str, ha="center", va="center",
                            fontsize=num_fs, fontweight="bold", color=num_color)
                else:
                    ax.text(j, i - 0.18, num_str, ha="center", va="center",
                            fontsize=num_fs, fontweight="bold", color=num_color)
                    ax.text(j, i + 0.22, lbl, ha="center", va="center",
                            fontsize=lbl_fs, color=txt_color, style="italic")
        ax.set_xticks(range(n_v))
        ax.set_yticks(range(n_v))
        ax.set_xticklabels(
            variants, fontsize=tick_fs, fontweight="bold",
            rotation=rotate_x, ha=("right" if rotate_x else "center"),
            rotation_mode="anchor",
        )
        ax.set_yticklabels(variants, fontsize=tick_fs, fontweight="bold")
        ax.tick_params(axis="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{title_top}\n{title_bot}", fontsize=11, color="#444", pad=10)

    _draw(ax_raw, cm_raw, "All differences", "raw cosine — full direction agreement")
    _draw(ax_sel, cm_sel,
          "Probe-specific differences only",
          "selective cosine — uniform offset removed")

    fig.text(0.06, 0.95, "Who pushes the base in the same direction?",
             fontsize=20, fontweight="bold", color="#222")
    fig.text(0.06, 0.905,
             "Two variants compared by their per-probe deviations from base",
             fontsize=11.5, color="#555", style="italic")

    # Legend strip
    ax_legend.text(
        0.0, 1.05,
        "Each cell asks: do variants A and B agree on which probes drift "
        "more or less? +1 = perfect agreement; 0 = independent.",
        ha="left", va="top", fontsize=10, color="#444",
        transform=ax_legend.transAxes,
    )
    strip_x = 0.04
    strip_w = 0.075
    for k, (color, lbl, txt_color) in enumerate(labels_strip):
        cx = strip_x + k * (strip_w + 0.012)
        ax_legend.add_patch(plt.Rectangle(
            (cx, 0.05), strip_w, 0.55,
            facecolor=color, edgecolor="#888", linewidth=0.6,
            transform=ax_legend.transAxes, clip_on=False))
        ax_legend.text(cx + strip_w / 2, 0.32, lbl,
                       ha="center", va="center",
                       fontsize=8, color=txt_color, fontweight="bold",
                       transform=ax_legend.transAxes, linespacing=1.2)

    # Bottom-line panel: prefer findings (cross-renderer consistency).
    cluster_finding = next(
        (f for f in findings if isinstance(f, DirectionClusterFinding)), None,
    )
    outlier_finding = next(
        (f for f in findings if isinstance(f, DirectionOutlierFinding)), None,
    )
    # Fallback: derive from the matrix when findings are absent (the < 3
    # variant case never produces these findings but the figure still
    # benefits from in-place auto-detection).
    if cluster_finding is None:
        cluster_raw, outlier_raw = _find_cluster_outlier(variants, cm_raw)
    else:
        cluster_raw = list(cluster_finding.details.get("variants", ()))
        if outlier_finding is not None:
            outlier_raw = [(
                outlier_finding.details["variant"],
                float(outlier_finding.details["mean_cosine_to_cluster"]),
            )]
        else:
            outlier_raw = []

    cluster_sel, outlier_sel = _find_cluster_outlier(variants, cm_sel)

    if cluster_raw:
        ix_cluster = [variants.index(v) for v in cluster_raw if v in variants]
        if len(ix_cluster) >= 2:
            mean_cluster_raw = float(np.mean([
                cm_raw[i, j]
                for i in ix_cluster for j in ix_cluster if i != j
            ]))
        else:
            mean_cluster_raw = float("nan")
    else:
        mean_cluster_raw = float("nan")

    ax_takeaway.text(0.0, 1.0, "Bottom line",
                     fontsize=14, fontweight="bold", color="#222",
                     transform=ax_takeaway.transAxes, va="top")
    if cluster_raw and not (mean_cluster_raw != mean_cluster_raw):
        cluster_size = len(cluster_raw)
        n_outliers = len(outlier_raw)
        if n_outliers:
            head_text = (
                f"{cluster_size} of {n_v} variants align\n"
                f"directionally\n"
                f"(cos ≈ {mean_cluster_raw:.2f}).\n"
                f"{n_outliers} stands apart."
            )
        else:
            head_text = (
                f"{cluster_size} of {n_v} variants align\n"
                f"directionally\n"
                f"(cos ≈ {mean_cluster_raw:.2f})."
            )
    else:
        head_text = "No directional cluster\ndetected at cos > 0.90."
    ax_takeaway.text(
        0.0, 0.92, head_text,
        fontsize=10.5, color="#333",
        transform=ax_takeaway.transAxes, va="top", linespacing=1.55,
    )

    bullets: list[str] = []
    if cluster_finding is not None:
        bullets.append("• Aligned cluster:")
        bullets.append(f"  {cluster_finding.summary}")
        bullets.append("")
    elif cluster_raw:
        bullets.append("• Aligned cluster:")
        bullets.append(f"  {{ {', '.join(cluster_raw)} }}")
        bullets.append("")

    if outlier_finding is not None:
        bullets.append("• Outlier:")
        bullets.append(f"  {outlier_finding.summary}")
        bullets.append("")
    elif outlier_raw:
        v, c = outlier_raw[0]
        bullets.append("• Outlier:")
        bullets.append(f"  {v}  (cos~{c:.2f})")
        bullets.append("")

    if cluster_raw:
        gap_raw = mean_cluster_raw - (outlier_raw[0][1] if outlier_raw else 0.0)
        if cluster_sel:
            ix_cs = [variants.index(v) for v in cluster_sel if v in variants]
            if len(ix_cs) >= 2:
                mean_cluster_sel = float(np.mean([
                    cm_sel[i, j]
                    for i in ix_cs for j in ix_cs if i != j
                ]))
            else:
                mean_cluster_sel = float("nan")
        else:
            mean_cluster_sel = float("nan")
        gap_sel = (
            mean_cluster_sel - (outlier_sel[0][1] if outlier_sel else 0.0)
            if mean_cluster_sel == mean_cluster_sel else float("nan")
        )
        bullets.append(f"• Gap: {gap_raw:+.2f} raw")
        if gap_sel == gap_sel:
            bullets.append(f"        {gap_sel:+.2f} selective")
            if gap_sel > gap_raw:
                bullets.append("  Gap widens after")
                bullets.append("  removing offset →")
                bullets.append("  real direction split.")

    y0 = 0.65
    for i, line in enumerate(bullets):
        ax_takeaway.text(0.0, y0 - i * 0.045, line,
                         fontsize=9.5, color="#222",
                         transform=ax_takeaway.transAxes, va="top",
                         family="DejaVu Sans Mono", linespacing=1.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor="white", dpi=dpi)
    plt.close(fig)
    return out_path


__all__ = ["render_direction"]
