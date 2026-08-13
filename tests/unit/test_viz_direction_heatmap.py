"""Tests for lmdiff.viz.direction_heatmap."""
from __future__ import annotations

import pytest


def _mpl_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


matplotlib_required = pytest.mark.skipif(
    not _mpl_available(), reason="matplotlib not installed",
)


class TestValidation:
    def test_empty_variant_names(self):
        from lmdiff.viz.direction_heatmap import plot_direction_heatmap
        with pytest.raises(ValueError, match="empty"):
            plot_direction_heatmap({}, variant_names=[], out_path="/tmp/x.png")

    def test_missing_row_raises(self):
        from lmdiff.viz.direction_heatmap import plot_direction_heatmap
        cos = {"A": {"A": 1.0}}
        with pytest.raises(ValueError, match="missing row"):
            plot_direction_heatmap(cos, variant_names=["A", "B"], out_path="/tmp/x.png")

    def test_missing_column_raises(self):
        from lmdiff.viz.direction_heatmap import plot_direction_heatmap
        cos = {"A": {"A": 1.0}, "B": {"B": 1.0}}  # no A-B cross entries
        with pytest.raises(ValueError, match="missing column"):
            plot_direction_heatmap(cos, variant_names=["A", "B"], out_path="/tmp/x.png")


@matplotlib_required
class TestRender:
    def test_writes_png(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from lmdiff.viz.direction_heatmap import plot_direction_heatmap

        out = tmp_path / "heatmap.png"
        cos = {
            "A": {"A": 1.0, "B": 0.5, "C": -0.3},
            "B": {"A": 0.5, "B": 1.0, "C": 0.1},
            "C": {"A": -0.3, "B": 0.1, "C": 1.0},
        }
        path = plot_direction_heatmap(
            cos, variant_names=["A", "B", "C"], out_path=out, title="Test",
        )
        assert out.exists()
        assert out.stat().st_size > 0
        assert path.endswith("heatmap.png")

    def test_handles_nan_cell(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from lmdiff.viz.direction_heatmap import plot_direction_heatmap

        cos = {
            "A": {"A": 1.0, "B": float("nan")},
            "B": {"A": float("nan"), "B": 1.0},
        }
        plot_direction_heatmap(
            cos, variant_names=["A", "B"], out_path=tmp_path / "r.png",
        )
