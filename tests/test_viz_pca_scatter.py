"""Tests for lmdiff.viz.pca_scatter."""
from __future__ import annotations

import pytest

from lmdiff.geometry import PCAResult


def _mpl_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


matplotlib_required = pytest.mark.skipif(
    not _mpl_available(), reason="matplotlib not installed",
)


def _pca_2d() -> PCAResult:
    return PCAResult(
        coords={
            "A": (1.2, -0.5),
            "B": (-0.8, 0.3),
            "C": (0.1, 1.0),
        },
        explained_variance_ratio=(0.6, 0.3),
        n_components=2,
        n_variants=3,
    )


def _pca_1d() -> PCAResult:
    return PCAResult(
        coords={"A": (1.0,), "B": (-0.5,)},
        explained_variance_ratio=(1.0,),
        n_components=1,
        n_variants=2,
    )


class TestValidation:
    def test_single_component_raises(self):
        from lmdiff.viz.pca_scatter import plot_pca_scatter
        with pytest.raises(ValueError, match="n_components"):
            plot_pca_scatter(_pca_1d(), out_path="/tmp/x.png")

    def test_no_variants_raises(self):
        from lmdiff.viz.pca_scatter import plot_pca_scatter
        bad = PCAResult(
            coords={}, explained_variance_ratio=(0.5, 0.5),
            n_components=2, n_variants=0,
        )
        with pytest.raises(ValueError, match="no variants"):
            plot_pca_scatter(bad, out_path="/tmp/x.png")


@matplotlib_required
class TestRender:
    def test_writes_png(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from lmdiff.viz.pca_scatter import plot_pca_scatter

        out = tmp_path / "pca.png"
        path = plot_pca_scatter(_pca_2d(), out_path=out, title="Test PCA")
        assert out.exists()
        assert out.stat().st_size > 0
        assert path.endswith("pca.png")

    def test_without_base_mark(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from lmdiff.viz.pca_scatter import plot_pca_scatter

        plot_pca_scatter(
            _pca_2d(), out_path=tmp_path / "p.png", show_base=False,
        )

    def test_variant_colors_override(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from lmdiff.viz.pca_scatter import plot_pca_scatter

        plot_pca_scatter(
            _pca_2d(),
            out_path=tmp_path / "p.png",
            variant_colors={"A": "#ff0000", "B": "#00ff00"},
        )
