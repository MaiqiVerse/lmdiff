"""Tests for lmdiff.viz.domain_bar."""
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
    def test_empty_heatmap_raises(self):
        from lmdiff.viz.domain_bar import plot_domain_bar
        with pytest.raises(ValueError, match="empty"):
            plot_domain_bar({}, out_path="/tmp/x.png")

    def test_bad_orientation_raises(self):
        from lmdiff.viz.domain_bar import plot_domain_bar
        with pytest.raises(ValueError, match="orientation"):
            plot_domain_bar(
                {"A": {"math": 1.0, "code": 2.0}},
                out_path="/tmp/x.png",
                orientation="diagonal",
            )

    def test_variants_without_any_domain(self):
        from lmdiff.viz.domain_bar import plot_domain_bar
        with pytest.raises(ValueError, match="no domains"):
            plot_domain_bar({"A": {}, "B": {}}, out_path="/tmp/x.png")


@matplotlib_required
class TestRender:
    def test_vertical_writes_png(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from lmdiff.viz.domain_bar import plot_domain_bar

        out = tmp_path / "bar.png"
        path = plot_domain_bar(
            {
                "A": {"math": 5.0, "code": 2.0, "knowledge": 3.5},
                "B": {"math": 2.0, "code": 6.0, "knowledge": 1.0},
            },
            out_path=out,
            title="Per-domain δ",
        )
        assert out.exists()
        assert out.stat().st_size > 0
        assert path.endswith("bar.png")

    def test_horizontal_renders(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from lmdiff.viz.domain_bar import plot_domain_bar

        plot_domain_bar(
            {
                "A": {"math": 5.0, "code": 2.0},
                "B": {"math": 2.0, "code": 6.0},
            },
            out_path=tmp_path / "bar.png",
            orientation="horizontal",
        )

    def test_missing_cells_render_as_hatched_zero(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from lmdiff.viz.domain_bar import plot_domain_bar
        # B doesn't have "code" — should still render (hatched zero bar).
        plot_domain_bar(
            {
                "A": {"math": 5.0, "code": 2.0, "knowledge": 3.0},
                "B": {"math": 2.0, "knowledge": 1.0},
            },
            out_path=tmp_path / "bar.png",
        )
