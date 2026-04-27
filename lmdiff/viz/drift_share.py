"""Drift + share dual-view heatmap (commit 1.8 of v0.3.0 batch 3).

Direct port of v6 §14.2 ``prototype_v5_clean.py`` — the **main
application-tier figure**. Two heatmaps side-by-side:

  - Left: How big was the change? (per-domain drift magnitude — sequential blue)
  - Right: Where did the variant act biggest? (share-of-budget,
    rows sum to 100% — diverging purple-orange)

The bottom-line panel sources its text from
:func:`lmdiff._findings.extract_findings` so the figure stays
consistent with the terminal/HTML/markdown renderers (per v6 §12.6).
matplotlib is imported lazily inside ``render_drift_share`` so that
``import lmdiff.viz.drift_share`` stays cheap.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lmdiff.geometry import GeoResult


def _ordered_domains(result: "GeoResult", explicit: list[str] | None) -> list[str]:
    if explicit:
        return list(explicit)
    meta_order = (result.metadata or {}).get("domain_order")
    if meta_order:
        return list(meta_order)
    seen: list[str] = []
    for d in (result.probe_domains or ()):
        if d is None:
            continue
        if d not in seen:
            seen.append(d)
    return seen


def render_drift_share(
    result: "GeoResult",
    out_path: str | Path,
    *,
    variant_order: list[str] | None = None,
    domain_order: list[str] | None = None,
    findings: tuple | None = None,
    dpi: int = 180,
) -> Path:
    """Render the drift + share dual-view figure.

    Parameters
    ----------
    result : GeoResult
        v5 GeoResult with ``probe_domains`` and ``avg_tokens_per_probe``
        populated.
    out_path : str | Path
        PNG file to write.
    variant_order : list[str] | None
        Default: ``sorted(result.variant_names)``.
    domain_order : list[str] | None
        Default: ``result.metadata['domain_order']``, else first-occurrence
        order in ``result.probe_domains``.
    findings : tuple | None
        Override ``result.findings`` (escape hatch for callers that want
        torch-free rendering — see PR notes about the v0.2.x torch leak).
    dpi : int
        Output PNG dpi.

    Returns
    -------
    Path
        The path written.
    """
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if findings is None:
        findings = result.findings
    findings = tuple(findings or ())

    from lmdiff._findings import (
        BiggestMoveFinding,
        MostLikeBaseFinding,
        SpecializationPeakFinding,
    )

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
    })

    variants = list(variant_order) if variant_order else sorted(result.variant_names)
    domains = _ordered_domains(result, domain_order)
    if not domains:
        raise ValueError(
            "render_drift_share requires probe_domains or domain_order; both empty"
        )

    domains_all = np.array(result.probe_domains)
    T = np.array(result.avg_tokens_per_probe) if result.avg_tokens_per_probe else None
    cv = {v: np.array(result.change_vectors[v]) for v in variants}

    norm = np.zeros((len(variants), len(domains)))
    for i, v in enumerate(variants):
        for j, d in enumerate(domains):
            mask = domains_all == d
            n_d = int(mask.sum())
            if n_d == 0:
                norm[i, j] = 0.0
                continue
            if T is not None:
                Tbar = float(T[mask].mean())
                if Tbar > 0:
                    norm[i, j] = float(np.sqrt((cv[v][mask] ** 2).sum() / (n_d * Tbar)))
                else:
                    norm[i, j] = float(np.sqrt((cv[v][mask] ** 2).sum() / n_d))
            else:
                norm[i, j] = float(np.sqrt((cv[v][mask] ** 2).sum() / n_d))

    norm_sq = norm ** 2
    row_sums = norm_sq.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    share = norm_sq / row_sums

    fig = plt.figure(figsize=(15, 7.4))
    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[2.7, 2.7, 1.0],
        height_ratios=[5.5, 0.55, 0.7],
        hspace=0.30, wspace=0.18,
        left=0.05, right=0.98, top=0.83, bottom=0.05,
    )
    ax_abs = fig.add_subplot(gs[0, 0])
    ax_z = fig.add_subplot(gs[0, 1])
    ax_takeaway = fig.add_subplot(gs[0:2, 2])
    ax_legend_abs = fig.add_subplot(gs[2, 0])
    ax_legend_z = fig.add_subplot(gs[2, 1])
    ax_takeaway.axis("off")
    ax_legend_abs.axis("off")
    ax_legend_z.axis("off")

    # Left: drift magnitude (sequential blue)
    abs_max = float(norm.max()) if norm.size else 0.21
    boundaries_abs = [0, 0.025, 0.05, 0.10, 0.20, max(abs_max + 0.01, 0.21)]
    colors_abs = ["#f0f0f0", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    cmap_abs = ListedColormap(colors_abs)
    norm_cmap_abs = BoundaryNorm(boundaries_abs, cmap_abs.N)
    ax_abs.imshow(norm, cmap=cmap_abs, norm=norm_cmap_abs, aspect="auto")

    for i, v in enumerate(variants):
        for j, d in enumerate(domains):
            val = norm[i, j]
            text_color = "white" if val > 0.10 else ("white" if val > 0.05 else "#222")
            ax_abs.text(j, i, f"{val:.4f}",
                        ha="center", va="center",
                        fontsize=15, fontweight="bold", color=text_color)

    ax_abs.set_yticks(range(len(variants)))
    ax_abs.set_yticklabels(variants, fontsize=12, fontweight="bold")
    ax_abs.set_xticks(range(len(domains)))
    ax_abs.set_xticklabels(domains, fontsize=9.5)
    ax_abs.tick_params(axis="both", length=0)
    for s in ax_abs.spines.values():
        s.set_visible(False)
    ax_abs.set_title(
        "How big was the change?\n"
        "(per-domain drift magnitude — bigger value, bigger move)",
        fontsize=11.5, color="#444", pad=10,
    )

    # Right: share (diverging purple-orange)
    boundaries_share = [0, 0.10, 0.18, 0.22, 0.30, 1.0]
    colors_share = ["#542788", "#b2abd2", "#f2f2f2", "#fdb863", "#b35806"]
    cmap_share = ListedColormap(colors_share)
    norm_cmap_share = BoundaryNorm(boundaries_share, cmap_share.N)
    ax_z.imshow(share, cmap=cmap_share, norm=norm_cmap_share, aspect="auto")

    for i, v in enumerate(variants):
        for j, d in enumerate(domains):
            sv = share[i, j]
            if sv < 0.10:
                c = "white"
            elif sv < 0.18:
                c = "#3d2855"
            elif sv < 0.22:
                c = "#444"
            elif sv < 0.30:
                c = "#3d2855"
            else:
                c = "white"
            ax_z.text(j, i, f"{sv * 100:.0f}%",
                      ha="center", va="center",
                      fontsize=18, fontweight="bold", color=c)

    ax_z.set_yticks(range(len(variants)))
    ax_z.set_yticklabels(variants, fontsize=12, fontweight="bold")
    ax_z.set_xticks(range(len(domains)))
    ax_z.set_xticklabels(domains, fontsize=9.5)
    ax_z.tick_params(axis="both", length=0)
    for s in ax_z.spines.values():
        s.set_visible(False)
    ax_z.set_title(
        "Where did the variant act biggest?\n"
        "(share of total change spent on each domain — rows sum to 100%)",
        fontsize=11.5, color="#444", pad=10,
    )

    # Big titles
    fig.text(0.05, 0.95,
             "How each variant differs from base — and where it acts biggest",
             fontsize=18, fontweight="bold", color="#222")
    fig.text(0.05, 0.905,
             "Left: how big each move is.   Right: which domain the variant "
             "acts on most.",
             fontsize=11.5, color="#555", style="italic")

    # Legends
    strip_w = 0.165
    abs_legend_items = [
        ("#f0f0f0", "< 0.025\nbarely moved", "#222"),
        ("#c6dbef", "0.025–0.05\nsmall move", "#222"),
        ("#6baed6", "0.05–0.10\nmoderate", "#222"),
        ("#2171b5", "0.10–0.20\nbig move", "white"),
        ("#08306b", "> 0.20\nhuge move", "white"),
    ]
    for k, (color, lbl, txt_color) in enumerate(abs_legend_items):
        cx = k * (strip_w + 0.005)
        ax_legend_abs.add_patch(plt.Rectangle(
            (cx, 0.0), strip_w, 0.65,
            facecolor=color, edgecolor="#888", linewidth=0.6,
            transform=ax_legend_abs.transAxes, clip_on=False))
        ax_legend_abs.text(cx + strip_w / 2, 0.32, lbl,
                           ha="center", va="center",
                           fontsize=8.5, color=txt_color, fontweight="bold",
                           transform=ax_legend_abs.transAxes, linespacing=1.2)
    ax_legend_abs.text(
        0.0, 1.05,
        "Smaller value = variant behaves more like base on this domain.",
        ha="left", va="top", fontsize=9.5, color="#444",
        transform=ax_legend_abs.transAxes,
    )

    share_legend_items = [
        ("#542788", "< 10%\nbarely acted", "white"),
        ("#b2abd2", "10–18%\nsmall action", "#3d2855"),
        ("#f2f2f2", "18–22%\nbalanced", "#444"),
        ("#fdb863", "22–30%\nbig action", "#3d2855"),
        ("#b35806", "> 30%\nbiggest action", "white"),
    ]
    for k, (color, lbl, txt_color) in enumerate(share_legend_items):
        cx = k * (strip_w + 0.005)
        ax_legend_z.add_patch(plt.Rectangle(
            (cx, 0.0), strip_w, 0.65,
            facecolor=color, edgecolor="#888", linewidth=0.6,
            transform=ax_legend_z.transAxes, clip_on=False))
        ax_legend_z.text(cx + strip_w / 2, 0.32, lbl,
                         ha="center", va="center",
                         fontsize=8.5, color=txt_color, fontweight="bold",
                         transform=ax_legend_z.transAxes, linespacing=1.2)
    ax_legend_z.text(
        0.0, 1.05,
        f"Each row sums to 100%.   Even split would be "
        f"{100 // max(len(domains), 1)}% per domain ({len(domains)} domains).",
        ha="left", va="top", fontsize=9.5, color="#444",
        transform=ax_legend_z.transAxes,
    )

    # Bottom-line panel — text comes from Findings (cross-renderer consistency).
    ax_takeaway.text(0.0, 1.0, "Bottom line",
                     fontsize=14, fontweight="bold", color="#222",
                     transform=ax_takeaway.transAxes, va="top")
    ax_takeaway.text(
        0.0, 0.93,
        "These plots show how\n"
        "big and where each\n"
        "variant moved — not\n"
        "whether the moves\n"
        "improved performance.",
        fontsize=10, color="#333",
        transform=ax_takeaway.transAxes, va="top", linespacing=1.5,
    )

    bullets: list[str] = []
    most_like = next((f for f in findings if isinstance(f, MostLikeBaseFinding)), None)
    if most_like is not None:
        bullets.extend(["• Most like base:", f"  {most_like.summary}", ""])

    biggest = next((f for f in findings if isinstance(f, BiggestMoveFinding)), None)
    if biggest is not None:
        bullets.extend(["• Biggest single move:", f"  {biggest.summary}", ""])

    peaks = [f for f in findings if isinstance(f, SpecializationPeakFinding)]
    if peaks:
        bullets.extend(["• Where each variant", "  acts biggest:"])
        for p in peaks:
            bullets.append(f"  {p.summary}")

    y0 = 0.62
    for i, line in enumerate(bullets):
        ax_takeaway.text(0.0, y0 - i * 0.043, line,
                         fontsize=9, color="#222",
                         transform=ax_takeaway.transAxes, va="top",
                         family="DejaVu Sans Mono", linespacing=1.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor="white", dpi=dpi)
    plt.close(fig)
    return out_path


__all__ = ["render_drift_share"]
