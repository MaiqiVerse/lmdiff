"""lmdiff.viz — optional matplotlib-based visualizations.

Every plot module lazy-imports matplotlib inside its function body, so
``from lmdiff.viz import plot_radar`` on a machine without matplotlib is
a successful no-op at import time; only the actual ``plot_*()`` call
raises ``ImportError("pip install lmdiff-kit[viz]")``.

v0.2.3 introduces the paper-grade family-figure suite. The pre-v0.2.3
single-purpose modules (``direction_heatmap``, ``domain_bar``,
``pca_scatter``, ``radar``) remain importable for backward compatibility.
"""
from lmdiff.viz._style import (
    BASE_MARKER,
    DEFAULT_DPI,
    FALLBACK_CMAP,
    VARIANT_COLORS,
    VARIANT_MARKERS,
    variant_color,
    variant_marker,
)
from lmdiff.viz.cosine import (
    plot_cosine_heatmap_raw,
    plot_cosine_heatmap_selective,
)
from lmdiff.viz.direction_heatmap import plot_direction_heatmap
from lmdiff.viz.domain_bar import plot_domain_bar
from lmdiff.viz.family_figures import FIGURE_REGISTRY, plot_family_figures
from lmdiff.viz.normalization_effect import plot_normalization_effect
from lmdiff.viz.normalized_magnitude import plot_normalized_magnitude_heatmap
from lmdiff.viz.pca import plot_pca_scatter_normalized, plot_pca_scatter_raw
from lmdiff.viz.pca_scatter import plot_pca_scatter
from lmdiff.viz.radar import plot_radar
from lmdiff.viz.specialization import plot_specialization_heatmap

__all__ = [
    # v0.2.3 family-figure suite
    "plot_family_figures",
    "FIGURE_REGISTRY",
    "plot_cosine_heatmap_raw",
    "plot_cosine_heatmap_selective",
    "plot_normalized_magnitude_heatmap",
    "plot_specialization_heatmap",
    "plot_pca_scatter_raw",
    "plot_pca_scatter_normalized",
    "plot_normalization_effect",
    # Style tokens
    "VARIANT_COLORS",
    "VARIANT_MARKERS",
    "FALLBACK_CMAP",
    "DEFAULT_DPI",
    "BASE_MARKER",
    "variant_color",
    "variant_marker",
    # Pre-v0.2.3 single-purpose modules
    "plot_radar",
    "plot_direction_heatmap",
    "plot_pca_scatter",
    "plot_domain_bar",
]
