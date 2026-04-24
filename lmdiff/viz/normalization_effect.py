"""Side-by-side bars: raw ‖δ‖ vs per-token-normalized magnitude.

Quantifies the L-022 effect (long-context probes dominating raw L2). The
raw bars carry an annotation for each variant's percent contribution
from the longest-prompt domain; normalized bars are unannotated.
"""
from __future__ import annotations

from pathlib import Path

from lmdiff.experiments.family import DEFAULT_DOMAIN_ORDER
from lmdiff.geometry import GeoResult
from lmdiff.viz._loaders import _load_geo
from lmdiff.viz._style import DEFAULT_DPI, variant_color


def _long_context_share(geo: GeoResult, variant: str) -> float | None:
    """Percent of raw ‖δ‖² contributed by the longest-prompt domain."""
    if not geo.probe_domains or not geo.avg_tokens_per_probe:
        return None
    try:
        per_domain_raw = geo.domain_heatmap()
    except ValueError:
        return None
    sums = {d: float(v) ** 2 for d, v in per_domain_raw[variant].items()}
    total = sum(sums.values())
    if total <= 0:
        return None
    # Pick the domain with the longest average prompt; if no probe_domains
    # info, fall back to "long-context" if present.
    import numpy as np
    tokens = np.asarray(geo.avg_tokens_per_probe, dtype=float)
    by_domain: dict[str, list[int]] = {}
    for idx, d in enumerate(geo.probe_domains):
        by_domain.setdefault(d if d is not None else "unknown", []).append(idx)
    avg_tok = {d: float(tokens[idxs].mean()) for d, idxs in by_domain.items()}
    longest = max(avg_tok, key=avg_tok.get)
    return 100.0 * sums.get(longest, 0.0) / total


def plot_normalization_effect(
    georesult: GeoResult | dict | str | Path,
    output_path: str | Path,
    *,
    figsize: tuple[float, float] = (10.0, 4.5),
    dpi: int = DEFAULT_DPI,
    variant_order: list[str] | None = None,
) -> Path:
    """Two stacked bar groups: raw ‖δ‖ (with longest-domain %) vs normalized."""
    geo = _load_geo(georesult)
    variants = list(variant_order) if variant_order else list(geo.variant_names)
    if not variants:
        raise ValueError("no variants to plot")

    raw_vals = [float(geo.magnitudes.get(v, 0.0)) for v in variants]
    norm_vals = [float(geo.magnitudes_normalized.get(v, 0.0)) for v in variants]
    long_share = [_long_context_share(geo, v) for v in variants]

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "matplotlib required. Install with: pip install lmdiff-kit[viz]"
        ) from exc

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=figsize)
    x = np.arange(len(variants))
    colors = [variant_color(v, i) for i, v in enumerate(variants)]

    bars_raw = ax_raw.bar(x, raw_vals, color=colors, edgecolor="black", linewidth=0.6)
    ax_raw.set_xticks(x)
    ax_raw.set_xticklabels(variants, fontsize=10)
    ax_raw.set_ylabel("raw ‖δ‖", fontsize=10)
    ax_raw.set_title("Raw magnitude (dominated by longest-prompt domain)",
                     fontsize=10, pad=8)
    ax_raw.grid(axis="y", linestyle=":", alpha=0.4)
    for bar, share in zip(bars_raw, long_share):
        if share is None:
            continue
        ax_raw.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{share:.1f}%",
            ha="center", va="bottom", fontsize=9, color="black",
        )

    ax_norm.bar(x, norm_vals, color=colors, edgecolor="black", linewidth=0.6)
    ax_norm.set_xticks(x)
    ax_norm.set_xticklabels(variants, fontsize=10)
    ax_norm.set_ylabel("‖δ‖ / √(N · ⟨tok⟩)", fontsize=10)
    ax_norm.set_title("Per-token normalized magnitude", fontsize=10, pad=8)
    ax_norm.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        f"Normalization effect — base: {geo.base_name}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


__all__ = ["plot_normalization_effect"]
