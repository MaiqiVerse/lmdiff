"""Raw-vs-normalized magnitude bars (commit 1.9 of v0.3.0 batch 3).

Direct port of v6 §14.4 ``prototype_magnitude.py``.

Two horizontal-bar charts side-by-side:

  - Left:  raw ‖δ‖ with a hatched overlay showing the long-context
    contribution (a length artifact, not a real-drift artefact)
  - Right: ‖δ‖ per √token (after L-022 per-token normalization)

Bottom-line panel: per-token-normalized ranking + the methodology
caveat about length-bias. The caveat is a fixed methodology note (not a
data finding) and always shown, per spec invariant (no domain knowledge
in figure narrative — but methodology notes are explicitly OK).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lmdiff.geometry import GeoResult


# Per-variant default colour palette; matches v6 §14.4. Used when the
# variant name maps cleanly; otherwise tab10 by index.
_DEFAULT_COLORS = {
    "code": "#1f77b4",
    "long": "#2ca02c",
    "math": "#9467bd",
    "yarn": "#d62728",
}


def _color_for(variant: str, idx: int) -> str:
    if variant in _DEFAULT_COLORS:
        return _DEFAULT_COLORS[variant]
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    return palette[idx % len(palette)]


def render_change_size(
    result: "GeoResult",
    out_path: str | Path,
    *,
    variant_order: list[str] | None = None,
    findings: tuple | None = None,  # noqa: ARG001 (methodology caveat is fixed)
    long_context_domain: str = "long-context",
    dpi: int = 180,
) -> Path:
    """Render the raw-vs-normalized magnitude bars figure."""
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
    })

    variants = list(variant_order) if variant_order else sorted(result.variant_names)
    if not variants:
        raise ValueError("render_change_size: no variants in result")

    cv = {v: np.asarray(result.change_vectors[v], dtype=float) for v in variants}
    raw = {v: float(np.sqrt((cv[v] ** 2).sum())) for v in variants}
    norm_map = result.magnitudes_normalized or {}
    norm = {v: float(norm_map.get(v, 0.0)) for v in variants}

    domains_all = np.asarray(result.probe_domains)
    if domains_all.size:
        mask_long = domains_all == long_context_domain
    else:
        mask_long = np.zeros(0, dtype=bool)
    pct_long: dict[str, float] = {}
    for v in variants:
        denom = float((cv[v] ** 2).sum())
        if denom <= 0 or mask_long.size == 0:
            pct_long[v] = 0.0
        else:
            pct_long[v] = float(100.0 * (cv[v][mask_long] ** 2).sum() / denom)

    # Order the bars by per-token-normalized magnitude (descending).
    order = sorted(variants, key=lambda v: -norm[v])

    fig = plt.figure(figsize=(14, 6.2))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[2.7, 2.7, 1.0],
        height_ratios=[5.5, 0.5],
        hspace=0.35, wspace=0.20,
        left=0.07, right=0.98, top=0.83, bottom=0.10,
    )
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_norm = fig.add_subplot(gs[0, 1])
    ax_takeaway = fig.add_subplot(gs[0, 2])
    ax_legend = fig.add_subplot(gs[1, 0:2])
    ax_takeaway.axis("off")
    ax_legend.axis("off")

    y_pos = np.arange(len(order))
    raw_max = max(raw.values()) if raw else 1.0
    raw_max = raw_max if raw_max > 0 else 1.0
    for k, v in enumerate(order):
        c = _color_for(v, k)
        ax_raw.barh(k, raw[v], color=c, edgecolor="#333", linewidth=0.6)
        long_portion = raw[v] * pct_long[v] / 100.0
        if long_portion > 0:
            ax_raw.barh(
                k, long_portion,
                color=c, alpha=0.4, hatch="///",
                edgecolor="white", linewidth=0,
            )
        ax_raw.text(raw[v] + raw_max * 0.02, k, f"{raw[v]:.1f}",
                    va="center", fontsize=11, fontweight="bold")
        if long_portion > raw_max * 0.05:
            ax_raw.text(long_portion / 2, k,
                        f"longbench\n{pct_long[v]:.1f}%",
                        va="center", ha="center", fontsize=8,
                        color="white", fontweight="bold")

    ax_raw.set_yticks(y_pos)
    ax_raw.set_yticklabels(order, fontsize=12, fontweight="bold")
    ax_raw.invert_yaxis()
    ax_raw.set_xlim(0, raw_max * 1.18)
    ax_raw.set_xlabel("‖δ‖ raw", fontsize=10)
    ax_raw.set_title("Before normalization — long probes dominate",
                     fontsize=11, color="#444", pad=10)
    for s in ("top", "right"):
        ax_raw.spines[s].set_visible(False)
    ax_raw.tick_params(axis="both", length=2)

    norm_max = max(norm.values()) if norm.values() else 0.001
    norm_max = norm_max if norm_max > 0 else 0.001
    for k, v in enumerate(order):
        c = _color_for(v, k)
        ax_norm.barh(k, norm[v], color=c, edgecolor="#333", linewidth=0.6)
        ax_norm.text(norm[v] + norm_max * 0.015, k, f"{norm[v]:.4f}",
                     va="center", fontsize=11, fontweight="bold")
    ax_norm.set_yticks(y_pos)
    ax_norm.set_yticklabels(order, fontsize=12, fontweight="bold")
    ax_norm.invert_yaxis()
    ax_norm.set_xlim(0, norm_max * 1.25)
    ax_norm.set_xlabel("‖δ‖ per √token", fontsize=10)
    ax_norm.set_title("After normalization — comparable across domains",
                      fontsize=11, color="#444", pad=10)
    for s in ("top", "right"):
        ax_norm.spines[s].set_visible(False)
    ax_norm.tick_params(axis="both", length=2)

    fig.text(0.07, 0.94, "How far has each variant moved from base?",
             fontsize=20, fontweight="bold", color="#222")
    fig.text(0.07, 0.895,
             "Hatched portion = share dominated by long-context probes "
             "(a length artifact, not real drift)",
             fontsize=11, color="#555", style="italic")

    ax_legend.text(
        0.5, 0.5,
        "Longer raw bars don't mean larger real change — they mean "
        "longer probes. Per-token normalization (right) shows the actual "
        "per-token drift.",
        ha="center", va="center", fontsize=10, color="#444",
        transform=ax_legend.transAxes,
    )

    ax_takeaway.text(0.0, 1.0, "Bottom line",
                     fontsize=14, fontweight="bold", color="#222",
                     transform=ax_takeaway.transAxes, va="top")
    ax_takeaway.text(
        0.0, 0.92,
        "Longbench probes are 100×\n"
        "longer than other tasks.\n"
        "Without normalization,\n"
        "they hide real differences.",
        fontsize=10.5, color="#333",
        transform=ax_takeaway.transAxes, va="top", linespacing=1.5,
    )

    ranking_norm = sorted(variants, key=lambda v: -norm[v])
    ranking_raw = sorted(variants, key=lambda v: -raw[v])
    lines = ["• Per-token ranking:"]
    for i, v in enumerate(ranking_norm):
        lines.append(f"  {i + 1}. {v}  {norm[v]:.4f}")
    lines.append("")
    lines.append("• Closest to base:")
    lines.append(f"  {ranking_norm[-1]}")
    lines.append("")
    if ranking_norm != ranking_raw:
        lines.append("• Raw vs normalized")
        lines.append("  rankings differ.")

    y0 = 0.62
    for i, line in enumerate(lines):
        ax_takeaway.text(0.0, y0 - i * 0.055, line,
                         fontsize=9.5, color="#222",
                         transform=ax_takeaway.transAxes, va="top",
                         family="DejaVu Sans Mono", linespacing=1.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor="white", dpi=dpi)
    plt.close(fig)
    return out_path


__all__ = ["render_change_size"]
