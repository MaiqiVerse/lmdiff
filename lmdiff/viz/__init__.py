"""lmdiff.viz — optional matplotlib-based visualizations.

Nothing imports matplotlib at package-init time. The radar module does
the lazy import inside its function, so ``import lmdiff.viz.radar`` is
safe on a machine without matplotlib installed (only ``plot_radar()``
actually fails).
"""
