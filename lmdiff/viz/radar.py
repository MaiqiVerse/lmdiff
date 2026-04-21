"""Matplotlib radar (spider) charts. Requires lmdiff-kit[viz].

Lazy-imports matplotlib so that ``import lmdiff.viz.radar`` on a machine
without matplotlib only fails when plot_radar() is actually called.
"""
from __future__ import annotations

from pathlib import Path


def plot_radar(
    data: dict[str, dict[str, float]],
    axes: list[str],
    *,
    title: str = "Radar",
    out_path: str | Path,
    variant_colors: dict[str, str] | None = None,
    value_range: tuple[float, float] | None = None,
    annotate: bool = True,
) -> str:
    """Render a radar chart and save as PNG.

    Args:
        data: {variant_name: {axis_name: value}}. Missing axes default to 0.
        axes: ordered list of axis labels. Length >= 3 (radar needs >= 3 axes).
        title: figure title.
        out_path: output PNG path (parent dirs created).
        variant_colors: optional {variant_name: color_str} override.
        value_range: (ymin, ymax) for the radial axis. None = inferred.
        annotate: overlay numeric values at each vertex.

    Returns:
        Absolute path (as str) of the written PNG.

    Raises:
        ImportError: matplotlib not installed (pip install lmdiff-kit[viz]).
        ValueError: len(axes) < 3 or data is empty.
    """
    if len(axes) < 3:
        raise ValueError(f"radar requires at least 3 axes, got {len(axes)}")
    if not data:
        raise ValueError("data is empty")

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "matplotlib required for radar plots. "
            "Install with: pip install lmdiff-kit[viz]"
        ) from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_axes = len(axes)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(axes, fontsize=11)

    all_vals = [v for var in data.values() for v in var.values()]
    if value_range is not None:
        ymin, ymax = value_range
    else:
        ymin = min(0.0, min(all_vals)) if all_vals else 0.0
        ymax = max(all_vals) * 1.15 if all_vals else 1.0
    ax.set_ylim(ymin, ymax)

    default_colors = plt.cm.tab10.colors if hasattr(plt.cm.tab10, "colors") else None
    for i, (variant, axis_map) in enumerate(data.items()):
        values = [axis_map.get(a, 0.0) for a in axes]
        values_closed = values + [values[0]]
        color = None
        if variant_colors is not None and variant in variant_colors:
            color = variant_colors[variant]
        elif default_colors is not None:
            color = default_colors[i % len(default_colors)]
        ax.plot(angles_closed, values_closed, linewidth=2, label=variant, color=color)
        ax.fill(angles_closed, values_closed, alpha=0.15, color=color)
        if annotate:
            for angle, val in zip(angles, values):
                ax.text(angle, val, f"{val:.2f}", fontsize=8, ha="center", va="bottom")

    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return str(out_path.resolve())
