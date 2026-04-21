"""Grouped bar chart of per-variant per-domain δ magnitude.

Input is the dict returned by ``GeoResult.domain_heatmap()``:
``{variant: {domain: magnitude}}``. Missing (variant, domain) cells render
as a thin hatched zero bar so the user sees the gap explicitly.
"""
from __future__ import annotations

from pathlib import Path


def plot_domain_bar(
    domain_heatmap: dict[str, dict[str, float]],
    *,
    out_path: str | Path,
    title: str = "Per-domain δ magnitude",
    variant_colors: dict[str, str] | None = None,
    orientation: str = "vertical",
) -> str:
    """Grouped bar chart. Each domain is a group; each variant is one bar.

    Args:
        domain_heatmap: {variant: {domain: magnitude}}.
        out_path: output PNG path (parent dirs created).
        title: figure title.
        variant_colors: optional {variant: color_str}.
        orientation: "vertical" (domains on x, magnitudes on y) or
            "horizontal" (domains on y).

    Returns:
        Absolute path (str) of the written PNG.

    Raises:
        ImportError: matplotlib not installed.
        ValueError: empty input or invalid orientation.
    """
    if not domain_heatmap:
        raise ValueError("domain_heatmap is empty")
    if orientation not in ("vertical", "horizontal"):
        raise ValueError(
            f"orientation must be 'vertical' or 'horizontal'; got {orientation!r}"
        )

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "matplotlib required for domain_bar. "
            "Install with: pip install lmdiff-kit[viz]"
        ) from exc

    # Collect stable domain + variant orderings. Variants keep insertion
    # order; domains are sorted for reproducibility.
    variants = list(domain_heatmap.keys())
    all_domains: set[str] = set()
    for per_v in domain_heatmap.values():
        all_domains.update(per_v.keys())
    domains = sorted(all_domains)
    if not domains:
        raise ValueError("domain_heatmap has variants but no domains")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_variants = len(variants)
    n_domains = len(domains)
    # Dynamic width so many domains don't squash
    base_w = max(8, 1.6 * n_domains + 2)
    fig, ax = plt.subplots(figsize=(base_w, 6), facecolor="none")
    ax.set_facecolor("none")

    default_colors = plt.cm.tab10.colors if hasattr(plt.cm.tab10, "colors") else None

    bar_width = 0.8 / max(1, n_variants)
    # Centering offsets so the cluster of bars sits over each domain tick
    offsets = np.arange(n_variants) * bar_width - (n_variants - 1) * bar_width / 2
    positions = np.arange(n_domains)

    for i, variant in enumerate(variants):
        heights: list[float] = []
        missing_mask: list[bool] = []
        for domain in domains:
            val = domain_heatmap[variant].get(domain)
            if val is None:
                heights.append(0.0)
                missing_mask.append(True)
            else:
                heights.append(float(val))
                missing_mask.append(False)

        color = None
        if variant_colors is not None and variant in variant_colors:
            color = variant_colors[variant]
        elif default_colors is not None:
            color = default_colors[i % len(default_colors)]

        if orientation == "vertical":
            bars = ax.bar(
                positions + offsets[i], heights, width=bar_width,
                color=color, edgecolor="black", linewidth=0.5, label=variant,
            )
        else:
            bars = ax.barh(
                positions + offsets[i], heights, height=bar_width,
                color=color, edgecolor="black", linewidth=0.5, label=variant,
            )
        # Hatch the missing cells so a zero there is visibly a gap, not a
        # real zero.
        for bar, missing in zip(bars, missing_mask):
            if missing:
                bar.set_hatch("//")
                bar.set_alpha(0.5)

    if orientation == "vertical":
        ax.set_xticks(positions)
        ax.set_xticklabels(domains, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel("‖δ‖", fontsize=11)
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels(domains, fontsize=10)
        ax.set_xlabel("‖δ‖", fontsize=11)

    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(True, axis="y" if orientation == "vertical" else "x", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return str(out_path.resolve())
