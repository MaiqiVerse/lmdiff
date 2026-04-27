"""Unit tests for the v0.3.0 5-layer terminal renderer (commit 1.7).

Drives ``lmdiff.report.terminal.render`` against a synthetic 4-variant
GeoResult that fires the major finding types, plus the calibration
Llama-2 georesult when present (skipped if absent).
"""
from __future__ import annotations

import io
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from lmdiff._findings import (
    AccuracyArtifactFinding,
    BaseAccuracyMissingFinding,
    BiggestMoveFinding,
    DirectionClusterFinding,
    DirectionOutlierFinding,
    MostLikeBaseFinding,
    SpecializationPeakFinding,
    TokenizerMismatchFinding,
)
from lmdiff.geometry import GeoResult, _compute_share_per_domain
from lmdiff.report import terminal as terminal_mod
from lmdiff.report._pipeline import _compose_one_liner


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


# ── Fixture builders ─────────────────────────────────────────────────


def _make_calibration_like() -> GeoResult:
    """4-variant × 5-domain GeoResult that triggers cluster + outlier +
    specialization peaks + accuracy artifact + base-accuracy-missing
    findings — i.e. exercises every Layer-2/4 branch."""
    variants = ["code", "long", "math", "yarn"]
    # 5 domains, 2 probes each.
    domains = (
        "commonsense", "commonsense",
        "reasoning", "reasoning",
        "math", "math",
        "code", "code",
        "long-context", "long-context",
    )
    n = len(domains)

    # Variant change-vector layouts that yield distinct per-domain peaks.
    cv = {
        "code":  [0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.40, 0.40, 0.10, 0.10],
        "long":  [0.10, 0.10, 0.45, 0.45, 0.20, 0.20, 0.10, 0.10, 0.10, 0.10],
        "math":  [0.10, 0.10, 0.20, 0.20, 0.40, 0.40, 0.15, 0.15, 0.10, 0.10],
        "yarn":  [0.40, 0.40, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.30, 0.30],
    }

    # cosine matrix: yarn/long/code agree pairwise > 0.90; math is < 0.85.
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
        metadata={
            "max_new_tokens": 16,
            "accuracy_by_variant": {
                "code": {"hellaswag": 0.53, "gsm8k": 0.0, "longbench_2wikimqa": 0.0},
                "long": {"hellaswag": 0.61, "gsm8k": 0.0, "longbench_2wikimqa": 0.0},
                "math": {"hellaswag": 0.48, "gsm8k": 0.01, "longbench_2wikimqa": 0.0},
                "yarn": {"hellaswag": 0.55, "gsm8k": 0.04, "longbench_2wikimqa": 0.0},
            },
        },
        probe_domains=domains,
        avg_tokens_per_probe=tuple([8.0] * n),
        magnitudes_normalized={v: float(np.linalg.norm(cv[v]) / 4.0) for v in variants},
    )
    geo.share_per_domain = _compute_share_per_domain(geo)
    return geo


def _make_pairwise() -> GeoResult:
    """2-variant GeoResult — DirectionCluster / Outlier won't fire."""
    domains = ("a", "a", "b", "b")
    cv = {"v1": [3.0, 4.0, 0.0, 0.0], "v2": [0.0, 0.0, 6.0, 8.0]}
    n = 4
    geo = GeoResult(
        base_name="base",
        variant_names=list(cv),
        n_probes=n,
        magnitudes={v: float(np.linalg.norm(cv[v])) for v in cv},
        cosine_matrix={v: {w: 1.0 if v == w else 0.1 for w in cv} for v in cv},
        change_vectors=cv,
        per_probe={v: {f"p{i}": cv[v][i] for i in range(n)} for v in cv},
        probe_domains=domains,
        avg_tokens_per_probe=tuple([8.0] * n),
    )
    geo.share_per_domain = _compute_share_per_domain(geo)
    return geo


# ── _compose_one_liner ───────────────────────────────────────────────


class TestComposeOneLiner:
    def test_cluster_plus_outlier(self):
        cluster = DirectionClusterFinding(
            severity="info",
            summary="{a, b, c} agree (cos ~0.95)",
            details={"variants": ("a", "b", "c"), "mean_cosine": 0.95,
                     "method": "pairwise_threshold_0.90"},
        )
        outlier = DirectionOutlierFinding(
            severity="info",
            summary="d stands apart (cos ~0.80)",
            details={"variant": "d", "mean_cosine_to_cluster": 0.80,
                     "cluster": ("a", "b", "c")},
        )
        out = _compose_one_liner((cluster, outlier))
        assert "3 variants align" in out
        assert "outlier" in out

    def test_distinct_specialization_peaks(self):
        peaks = tuple(
            SpecializationPeakFinding(
                severity="info",
                summary=f"{v}: 50% on {d}",
                details={"variant": v, "domain": d, "share": 0.5},
            )
            for v, d in [("a", "x"), ("b", "y"), ("c", "z")]
        )
        out = _compose_one_liner(peaks)
        assert "Each variant acts biggest on a different domain" in out
        # Each summary should appear verbatim per v6 §12.6.
        for v, d in [("a", "x"), ("b", "y"), ("c", "z")]:
            assert f"{v}: 50% on {d}" in out

    def test_biggest_move_alone(self):
        big = BiggestMoveFinding(
            severity="info",
            summary="long on reasoning (drift 0.3355)",
            details={"variant": "long", "domain": "reasoning", "drift": 0.3355},
        )
        out = _compose_one_liner((big,))
        assert "long on reasoning" in out

    def test_generic_fallback(self):
        out = _compose_one_liner((), n_probes=42, n_domains=5)
        assert "42 probes" in out
        assert "5 domains" in out


# ── Color auto-disable ───────────────────────────────────────────────


class TestColorAutoDisable:
    def test_no_color_env_disables_ansi(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        geo = _make_pairwise()
        buf = io.StringIO()
        out = terminal_mod.render(geo, file=buf)
        assert "\x1b[" not in out

    def test_non_tty_disables_ansi(self):
        geo = _make_pairwise()
        buf = io.StringIO()  # StringIO is not a tty
        out = terminal_mod.render(geo, file=buf)
        assert "\x1b[" not in out

    def test_force_color_overrides_non_tty(self):
        geo = _make_pairwise()
        buf = io.StringIO()
        out = terminal_mod.render(geo, file=buf, color=True)
        assert "\x1b[" in out

    def test_color_false_overrides_tty(self, monkeypatch):
        # Even with NO_COLOR unset, color=False stops ANSI.
        monkeypatch.delenv("NO_COLOR", raising=False)
        geo = _make_pairwise()
        buf = io.StringIO()
        out = terminal_mod.render(geo, file=buf, color=False)
        assert "\x1b[" not in out


# ── 5-layer structure on calibration-like data ───────────────────────


class TestFiveLayerStructure:
    def test_all_five_layers_present(self):
        geo = _make_calibration_like()
        out = _strip_ansi(terminal_mod.render(geo, file=io.StringIO()))
        # Layer 0/banner
        assert "Family experiment:" in out
        assert "═" in out
        # Layer 2
        assert "Headlines" in out
        # Layer 3 — at least one of each table title
        assert "Where each variant acts biggest" in out
        assert "How big is each move" in out
        assert "Direction agreement" in out
        assert "Per-task accuracy" in out
        # Layer 4
        assert "Caveats" in out
        # Layer 5
        assert "See also" in out
        # Mental-model reminder fires unconditionally
        assert "Drift magnitude shows where" in out

    def test_finding_summaries_appear_verbatim(self):
        geo = _make_calibration_like()
        out = _strip_ansi(terminal_mod.render(geo, file=io.StringIO()))
        # Each finding's `.summary` must appear unchanged in output.
        for f in geo.findings:
            if not f.summary:
                continue
            # `summary` may contain characters that aren't visible in the
            # specific color theme — strip ansi already; assert literal.
            assert f.summary in out, (
                f"Finding {type(f).__name__} summary {f.summary!r} not in output"
            )

    def test_one_liner_for_calibration_picks_specialization(self):
        geo = _make_calibration_like()
        out = _strip_ansi(terminal_mod.render(geo, file=io.StringIO()))
        # 4 distinct peaks → "Each variant acts biggest on a different domain"
        # OR cluster+outlier (whichever fires first per dispatch)
        assert (
            "Each variant acts biggest on a different domain" in out
            or "variants align (cluster)" in out
        )

    def test_accuracy_artifact_marker_in_table(self):
        geo = _make_calibration_like()
        out = terminal_mod.render(geo, file=io.StringIO())
        plain = _strip_ansi(out)
        # gsm8k value should carry a `*` (artifact marker), longbench too.
        # Verify the literal "0.00*" or "0.01*" appears.
        assert "0.00*" in plain
        assert "0.04*" in plain


# ── Pairwise (2 variants) — no cluster / outlier ─────────────────────


class TestPairwiseRender:
    def test_no_cluster_or_outlier_in_output(self):
        geo = _make_pairwise()
        out = _strip_ansi(terminal_mod.render(geo, file=io.StringIO()))
        # The Layer 2 headline labels must not contain cluster/outlier.
        # (Their summaries can still appear inside the output if other
        # finding text accidentally collides, so keep the assertion narrow.)
        assert "Direction cluster" not in out
        assert "Direction outlier" not in out

    def test_falls_back_to_biggest_move_one_liner(self):
        geo = _make_pairwise()
        out = _strip_ansi(terminal_mod.render(geo, file=io.StringIO()))
        # Layer 1 should be the BiggestMove summary, not the generic fallback.
        big = next(f for f in geo.findings if isinstance(f, BiggestMoveFinding))
        first_lines = "\n".join(out.split("\n")[:8])
        assert big.summary in first_lines


# ── Empty findings → still valid 5-layer output ─────────────────────


class TestEmptyFindings:
    def test_no_findings_no_exception(self):
        geo = _make_pairwise()
        out = terminal_mod.render(geo, file=io.StringIO(), findings=())
        plain = _strip_ansi(out)
        assert "Family experiment" in plain
        assert "Headlines" in plain
        assert "Caveats" in plain
        assert "See also" in plain


# ── Adaptive width ───────────────────────────────────────────────────


class TestAdaptiveWidth:
    def test_compact_format_under_80_cols(self):
        geo = _make_calibration_like()
        out = _strip_ansi(terminal_mod.render(geo, file=io.StringIO(), width=60))
        # Compact mode: single per-variant block instead of matrices.
        assert "Per-variant breakdown" in out
        # Standard tables suppressed in narrow mode.
        assert "Where each variant acts biggest" not in out

    def test_standard_format_at_100_cols(self):
        geo = _make_calibration_like()
        out = _strip_ansi(terminal_mod.render(geo, file=io.StringIO(), width=100))
        assert "Where each variant acts biggest" in out


# ── In-memory pointer fallback ───────────────────────────────────────


class TestPointersInMemory:
    def test_in_memory_message_when_no_paths(self):
        geo = _make_pairwise()
        # Strip all path metadata.
        geo.metadata.pop("summary_json_path", None)
        geo.metadata.pop("georesult_json_path", None)
        geo.metadata.pop("figures_dir", None)
        out = _strip_ansi(terminal_mod.render(geo, file=io.StringIO()))
        assert "in-memory result" in out
        assert "result.save" in out


# ── Calibration: render the real 4-variant Llama-2 georesult ─────────


_CALIB_PATH = Path("runs/llama2-4variants/family_geometry_lm_eval_georesult.json")


@pytest.mark.skipif(
    not _CALIB_PATH.exists(),
    reason="Llama-2 4-variant calibration GeoResult not present in the repo",
)
class TestLlama2CalibrationRender:
    @pytest.fixture(scope="class")
    def geo(self) -> GeoResult:
        from lmdiff.report.json_report import geo_result_from_json_dict
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with open(_CALIB_PATH, encoding="utf-8") as f:
                return geo_result_from_json_dict(json.load(f))

    def test_renders_without_exception(self, geo):
        out = terminal_mod.render(geo, file=io.StringIO())
        assert "Family experiment" in _strip_ansi(out)

    def test_specialization_peaks_visible(self, geo):
        out = _strip_ansi(terminal_mod.render(geo, file=io.StringIO()))
        # Each variant from the Llama-2 family should appear on a row.
        for v in ("yarn", "long", "code", "math"):
            assert v in out
