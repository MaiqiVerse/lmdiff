"""lmdiff.viz — optional matplotlib-based visualizations.

Every plot module lazy-imports matplotlib inside its function body, so
``from lmdiff.viz import plot_radar`` on a machine without matplotlib is
a successful no-op at import time; only the actual ``plot_*()`` call
raises ``ImportError("pip install lmdiff-kit[viz]")``.
"""
from lmdiff.viz.direction_heatmap import plot_direction_heatmap
from lmdiff.viz.domain_bar import plot_domain_bar
from lmdiff.viz.pca_scatter import plot_pca_scatter
from lmdiff.viz.radar import plot_radar

__all__ = [
    "plot_radar",
    "plot_direction_heatmap",
    "plot_pca_scatter",
    "plot_domain_bar",
]
