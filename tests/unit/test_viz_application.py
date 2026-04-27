"""Unit tests for the v0.3.0 application-tier figures (commits 1.8 + 1.9).

Each figure is rendered against a synthetic 4-variant fixture and (when
present) the calibration Llama-2 georesult. Tests assert PNG-file
existence + reasonable size + presence of finding summaries in the SVG
shadow render (matplotlib renders the same Figure to either backend,
so SVG text proves what would appear on the PNG).
"""
from __future__ import annotations

import io
import json
import warnings
from pathlib import Path

import pytest

# Force a non-interactive backend before any matplotlib import.
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402

from lmdiff.geometry import GeoResult, _compute_share_per_domain  # noqa: E402


# ── Fixture ──────────────────────────────────────────────────────────


def _make_calibration_like() -> GeoResult:
    variants = ["code", "long", "math", "yarn"]
    domains = (
        "commonsense", "commonsense",
        "reasoning", "reasoning",
        "math", "math",
        "code", "code",
        "long-context", "long-context",
    )
    n = len(domains)
    cv = {
        "code":  [0.10, 0.10, 0.10, 0.10, 0.15, 0.15, 0.40, 0.40, 0.10, 0.10],
        "long":  [0.10, 0.10, 0.45, 0.45, 0.20, 0.20, 0.10, 0.10, 0.10, 0.10],
        "math":  [0.10, 0.10, 0.20, 0.20, 0.40, 0.40, 0.15, 0.15, 0.10, 0.10],
        "yarn":  [0.40, 0.40, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.30, 0.30],
    }
    cos = {
        "code": {"code": 1.0, "long": 0.95, "math": 0.79, "yarn": 0.96},
        "long": {"code": 0.95, "long": 1.0, "math": 0.80, "yarn": 0.95},
        "math": {"code": 0.79, "long": 0.80, "math": 1.0, "yarn": 0.80},
        "yarn": {"code": 0.96, "long": 0.95, "math": 0.80, "yarn": 1.0},
    }
    sel = {
        v: {w: 1.0 if v == w else cos[v][w] - 0.02 for w in cos} for v in cos
    }
    geo = GeoResult(
        base_name="meta-llama/Llama-2-7b-hf",
        variant_names=variants,
        n_probes=n,
        magnitudes={v: float(np.linalg.norm(cv[v])) for v in variants},
        cosine_matrix=cos,
        selective_cosine_matrix=sel,
        change_vectors=cv,
        per_probe={v: {f"p{i}": cv[v][i] for i in range(n)} for v in variants},
        probe_domains=domains,
        avg_tokens_per_probe=tuple([8.0] * n),
        magnitudes_normalized={v: float(np.linalg.norm(cv[v]) / 4.0) for v in variants},
    )
    geo.share_per_domain = _compute_share_per_domain(geo)
    return geo


def _make_pairwise() -> GeoResult:
    """2-variant case — no cluster/outlier; all figures must still render."""
    variants = ["v1", "v2"]
    domains = ("a", "a", "b", "b")
    n = len(domains)
    cv = {"v1": [3.0, 4.0, 0.0, 0.0], "v2": [0.0, 0.0, 6.0, 8.0]}
    geo = GeoResult(
        base_name="base",
        variant_names=variants,
        n_probes=n,
        magnitudes={v: float(np.linalg.norm(cv[v])) for v in variants},
        cosine_matrix={v: {w: 1.0 if v == w else 0.1 for w in variants} for v in variants},
        selective_cosine_matrix={
            v: {w: 1.0 if v == w else 0.05 for w in variants} for v in variants
        },
        change_vectors=cv,
        per_probe={v: {f"p{i}": cv[v][i] for i in range(n)} for v in variants},
        probe_domains=domains,
        avg_tokens_per_probe=tuple([8.0] * n),
        magnitudes_normalized={v: float(np.linalg.norm(cv[v]) / 2.0) for v in variants},
    )
    geo.share_per_domain = _compute_share_per_domain(geo)
    return geo


_CALIB_PATH = Path("runs/llama2-4variants/family_geometry_lm_eval_georesult.json")


@pytest.fixture(scope="module")
def calibration_geo():
    if not _CALIB_PATH.exists():
        pytest.skip("Llama-2 4-variant georesult JSON not present")
    from lmdiff.report.json_report import geo_result_from_json_dict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with open(_CALIB_PATH, encoding="utf-8") as f:
            return geo_result_from_json_dict(json.load(f))


# ── drift_share (commit 1.8) ─────────────────────────────────────────


class TestDriftShare:
    def test_render_writes_png(self, tmp_path):
        from lmdiff.viz.drift_share import render_drift_share
        out = render_drift_share(_make_calibration_like(), tmp_path / "ds.png")
        assert out.exists()
        # >50KB sanity (empty PNG renders are ~5KB).
        assert out.stat().st_size > 50_000

    def test_synthetic_does_not_crash(self, tmp_path):
        from lmdiff.viz.drift_share import render_drift_share
        out = render_drift_share(_make_calibration_like(), tmp_path / "syn.png")
        assert out.exists()

    def test_pairwise_renders_two_rows(self, tmp_path):
        from lmdiff.viz.drift_share import render_drift_share
        out = render_drift_share(_make_pairwise(), tmp_path / "pair.png")
        assert out.exists()

    def test_custom_variant_order_respected(self, tmp_path):
        from lmdiff.viz.drift_share import render_drift_share
        # Render with reversed variant order — y-axis labels reverse.
        geo = _make_calibration_like()
        out_alpha = render_drift_share(geo, tmp_path / "alpha.png")
        out_rev = render_drift_share(
            geo, tmp_path / "rev.png",
            variant_order=["yarn", "math", "long", "code"],
        )
        # Different order → different bytes.
        assert out_alpha.read_bytes() != out_rev.read_bytes()

    def test_bottom_line_carries_finding_summaries(self, tmp_path, monkeypatch):
        """Render to SVG via the same pipeline + assert finding summaries
        appear as <text> contents."""
        from lmdiff.viz.drift_share import render_drift_share
        # Render to PNG first (the production path).
        png = render_drift_share(_make_calibration_like(), tmp_path / "ds.png")
        assert png.exists()
        # Then re-render to SVG by patching plt.savefig to grab the SVG.
        # Easier path: temp-render the same data with savefig to .svg.
        out_svg = tmp_path / "ds.svg"
        from lmdiff.viz.drift_share import render_drift_share as _r
        # Trick: pass an .svg path; matplotlib infers format from suffix.
        _r(_make_calibration_like(), out_svg)
        svg_text = out_svg.read_text(encoding="utf-8")
        geo = _make_calibration_like()
        # At least one finding summary should appear in the SVG text.
        summaries = [f.summary for f in geo.findings]
        assert any(s in svg_text for s in summaries), summaries

    def test_calibration_dataset_renders(self, tmp_path, calibration_geo):
        from lmdiff.viz.drift_share import render_drift_share
        out = render_drift_share(calibration_geo, tmp_path / "calib.png")
        assert out.exists()
        assert out.stat().st_size > 50_000


# ── direction (commit 1.9) ───────────────────────────────────────────


class TestDirection:
    def test_render_writes_png(self, tmp_path):
        from lmdiff.viz.direction import render_direction
        out = render_direction(_make_calibration_like(), tmp_path / "dir.png")
        assert out.exists()
        assert out.stat().st_size > 50_000

    def test_pairwise_two_by_two_no_crash(self, tmp_path):
        from lmdiff.viz.direction import render_direction
        out = render_direction(_make_pairwise(), tmp_path / "dir2.png")
        assert out.exists()

    def test_bottom_line_uses_cluster_finding(self, tmp_path):
        from lmdiff.viz.direction import render_direction
        out_svg = tmp_path / "dir.svg"
        render_direction(_make_calibration_like(), out_svg)
        svg_text = out_svg.read_text(encoding="utf-8")
        geo = _make_calibration_like()
        from lmdiff._findings import (
            DirectionClusterFinding,
            DirectionOutlierFinding,
        )
        cluster = next(
            (f for f in geo.findings if isinstance(f, DirectionClusterFinding)),
            None,
        )
        outlier = next(
            (f for f in geo.findings if isinstance(f, DirectionOutlierFinding)),
            None,
        )
        # At least one of the cluster/outlier summaries should appear in
        # the rendered SVG bottom-line panel.
        assert (
            (cluster is not None and cluster.summary in svg_text)
            or (outlier is not None and outlier.summary in svg_text)
        )

    def test_calibration_dataset_renders(self, tmp_path, calibration_geo):
        from lmdiff.viz.direction import render_direction
        out = render_direction(calibration_geo, tmp_path / "calib.png")
        assert out.exists()


# ── change_size (commit 1.9) ─────────────────────────────────────────


class TestChangeSize:
    def test_render_writes_png(self, tmp_path):
        from lmdiff.viz.change_size import render_change_size
        out = render_change_size(_make_calibration_like(), tmp_path / "cs.png")
        assert out.exists()
        assert out.stat().st_size > 30_000

    def test_pairwise_renders(self, tmp_path):
        from lmdiff.viz.change_size import render_change_size
        out = render_change_size(_make_pairwise(), tmp_path / "cs2.png")
        assert out.exists()

    def test_ranking_in_svg(self, tmp_path):
        from lmdiff.viz.change_size import render_change_size
        out_svg = tmp_path / "cs.svg"
        render_change_size(_make_calibration_like(), out_svg)
        svg_text = out_svg.read_text(encoding="utf-8")
        # Per-token ranking block must be in the bottom-line panel.
        assert "Per-token ranking" in svg_text
        # And the variant names should all appear at least once in the SVG.
        for v in ("code", "long", "math", "yarn"):
            assert v in svg_text

    def test_calibration_dataset_renders(self, tmp_path, calibration_geo):
        from lmdiff.viz.change_size import render_change_size
        out = render_change_size(calibration_geo, tmp_path / "calib.png")
        assert out.exists()


# ── figures.py pipeline wire ─────────────────────────────────────────


class TestFiguresPipelineWire:
    def test_applied_tier_writes_three_pngs(self, tmp_path):
        geo = _make_calibration_like()
        rendered = geo.figures(out_dir=str(tmp_path))
        assert isinstance(rendered, list)
        assert len(rendered) == 3
        names = sorted(p.name for p in rendered)
        assert names == [
            "change_size_bars.png",
            "direction_agreement.png",
            "drift_share_dual.png",
        ]
        for p in rendered:
            assert p.exists()
            assert p.stat().st_size > 30_000

    def test_paper_tier_returns_dict(self, tmp_path):
        geo = _make_calibration_like()
        rendered = geo.figures(out_dir=str(tmp_path), tier="paper")
        # v0.2.x suite returns {key: path}; just smoke-test it ran.
        assert isinstance(rendered, dict)

    def test_unknown_tier_raises(self, tmp_path):
        geo = _make_calibration_like()
        with pytest.raises(ValueError, match="Unknown tier"):
            geo.figures(out_dir=str(tmp_path), tier="bogus")

    def test_custom_variant_order_propagates(self, tmp_path):
        geo = _make_calibration_like()
        out_default = tmp_path / "def"
        out_custom = tmp_path / "cus"
        geo.figures(out_dir=str(out_default))
        geo.figures(
            out_dir=str(out_custom),
            variant_order=["yarn", "math", "long", "code"],
        )
        # drift_share row order honors variant_order; bytes differ.
        a = (out_default / "drift_share_dual.png").read_bytes()
        b = (out_custom / "drift_share_dual.png").read_bytes()
        assert a != b
