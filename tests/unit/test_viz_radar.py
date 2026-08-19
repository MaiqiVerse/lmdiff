"""Tests for lmdiff.viz.radar — validates input shape and import-error path.

Uses matplotlib's Agg backend to render without a display. If matplotlib
isn't installed in the test env, the ImportError test still runs; the
render tests are skipped.
"""
from __future__ import annotations

import builtins

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
    def test_too_few_axes(self):
        from lmdiff.viz.radar import plot_radar
        with pytest.raises(ValueError, match="at least 3 axes"):
            plot_radar(
                {"v1": {"a": 1, "b": 2}},
                axes=["a", "b"],
                out_path="/tmp/x.png",
            )

    def test_empty_data(self):
        from lmdiff.viz.radar import plot_radar
        with pytest.raises(ValueError, match="empty"):
            plot_radar({}, axes=["a", "b", "c"], out_path="/tmp/x.png")


@matplotlib_required
class TestRender:
    def test_writes_png(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from lmdiff.viz.radar import plot_radar

        out = tmp_path / "radar.png"
        path = plot_radar(
            {
                "base": {"task_a": 0.5, "task_b": 0.7, "task_c": 0.6},
                "variant_x": {"task_a": 0.7, "task_b": 0.8, "task_c": 0.5},
            },
            axes=["task_a", "task_b", "task_c"],
            out_path=out,
            title="Test Radar",
        )
        assert out.exists()
        assert out.stat().st_size > 0
        assert path.endswith("radar.png")

    def test_missing_axis_treated_as_zero(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from lmdiff.viz.radar import plot_radar
        # v2 missing "task_b" — should render with 0, not crash.
        plot_radar(
            {
                "v1": {"task_a": 1.0, "task_b": 2.0, "task_c": 3.0},
                "v2": {"task_a": 0.5, "task_c": 1.5},
            },
            axes=["task_a", "task_b", "task_c"],
            out_path=tmp_path / "r.png",
        )

    def test_value_range_applied(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from lmdiff.viz.radar import plot_radar
        # Must not crash when value_range is explicit.
        plot_radar(
            {"v": {"a": 0.3, "b": 0.4, "c": 0.5}},
            axes=["a", "b", "c"],
            value_range=(0.0, 1.0),
            out_path=tmp_path / "r.png",
        )


class TestImportError:
    def test_raises_when_matplotlib_missing(self, monkeypatch):
        """Force the in-function matplotlib import to fail and check the message."""
        import lmdiff.viz.radar as radar_mod

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError("matplotlib missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match=r"pip install lmdiff-kit\[viz\]"):
            radar_mod.plot_radar(
                {"v": {"a": 1, "b": 2, "c": 3}},
                axes=["a", "b", "c"],
                out_path="/tmp/x.png",
            )
