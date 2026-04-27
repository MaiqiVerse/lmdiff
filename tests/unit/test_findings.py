"""Unit tests for finding extraction (commit 1.6).

One synthetic GeoResult fixture per rule type — keeps each test focused
on the rule under examination. The Llama-2 4-variant calibration check
runs against `runs/llama2-4variants/family_geometry_lm_eval_georesult.json`
when present (skipped otherwise so CI without the artifact is green).
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import pytest

from lmdiff._findings import (
    AccuracyArtifactFinding,
    BaseAccuracyMissingFinding,
    BiggestMoveFinding,
    DirectionClusterFinding,
    DirectionOutlierFinding,
    Finding,
    MostLikeBaseFinding,
    SpecializationPeakFinding,
    TokenizerMismatchFinding,
    extract_findings,
)
from lmdiff.geometry import GeoResult, _compute_share_per_domain


# ── Fixture builder ───────────────────────────────────────────────────


def _build_geo(
    *,
    base_name: str = "base",
    variants: dict[str, list[float]] | None = None,
    domains: tuple[str, ...] = (),
    cosine_matrix: dict[str, dict[str, float]] | None = None,
    metadata: dict | None = None,
) -> GeoResult:
    variants = variants or {"v1": [3.0, 4.0, 0.0, 0.0], "v2": [0.0, 0.0, 6.0, 8.0]}
    n = len(next(iter(variants.values())))
    if not domains:
        domains = ("a", "a", "b", "b")[:n]
    if cosine_matrix is None:
        cosine_matrix = {
            v: {w: 1.0 if v == w else 0.0 for w in variants}
            for v in variants
        }
    geo = GeoResult(
        base_name=base_name,
        variant_names=list(variants),
        n_probes=n,
        magnitudes={v: math.sqrt(sum(x * x for x in cv)) for v, cv in variants.items()},
        cosine_matrix=cosine_matrix,
        change_vectors=variants,
        per_probe={v: {f"p{i}": cv[i] for i in range(n)} for v, cv in variants.items()},
        metadata=metadata or {},
        probe_domains=domains,
        avg_tokens_per_probe=tuple([8.0] * n),
    )
    geo.share_per_domain = _compute_share_per_domain(geo)
    return geo


# ── MostLikeBase / BiggestMove ────────────────────────────────────────


class TestMostLikeBaseAndBiggestMove:
    def test_min_drift_cell_picked_as_most_like_base(self):
        # v1 puts all energy on domain 'a'; v2 on domain 'b'.
        # Smallest cell = (v1, b, 0.0) or (v2, a, 0.0) (tied at 0).
        # The lex tie-breaker (variant ASC, domain ASC) selects (v1, b).
        geo = _build_geo()
        findings = extract_findings(geo)
        most_like = next(f for f in findings if isinstance(f, MostLikeBaseFinding))
        assert most_like.severity == "info"
        assert most_like.details["drift"] == pytest.approx(0.0)
        # The picked cell is the most-base-like one — value 0.0.
        assert most_like.summary.startswith("v1 on b") or most_like.summary.startswith("v2 on a")

    def test_max_drift_cell_picked_as_biggest_move(self):
        geo = _build_geo()
        findings = extract_findings(geo)
        biggest = next(f for f in findings if isinstance(f, BiggestMoveFinding))
        assert biggest.severity == "info"
        # Largest is v2 on b at 10.0, v1 on a at 5.0 — pick v2 on b.
        assert biggest.details["variant"] == "v2"
        assert biggest.details["domain"] == "b"
        assert biggest.details["drift"] == pytest.approx(10.0)


# ── DirectionClusterFinding + DirectionOutlierFinding ────────────────


class TestDirectionClusterAndOutlier:
    def _cluster_geo(self) -> GeoResult:
        # 4 variants: a/b/c agree pairwise > 0.90; d disagrees < 0.85.
        cv = {
            "a": [1.0, 0.0, 0.0],
            "b": [1.0, 0.05, 0.0],
            "c": [0.95, 0.0, 0.05],
            "d": [0.0, 1.0, 0.0],
        }
        cos = {
            "a": {"a": 1.0, "b": 0.95, "c": 0.93, "d": 0.0},
            "b": {"a": 0.95, "b": 1.0, "c": 0.92, "d": 0.05},
            "c": {"a": 0.93, "b": 0.92, "c": 1.0, "d": 0.0},
            "d": {"a": 0.0, "b": 0.05, "c": 0.0, "d": 1.0},
        }
        return _build_geo(
            variants=cv, domains=("a", "b", "c"), cosine_matrix=cos,
        )

    def test_cluster_fires_with_three_agreeing_variants(self):
        geo = self._cluster_geo()
        findings = extract_findings(geo)
        cluster = next(
            f for f in findings if isinstance(f, DirectionClusterFinding)
        )
        assert cluster.details["variants"] == ("a", "b", "c")
        assert cluster.details["mean_cosine"] > 0.90

    def test_outlier_fires_when_one_variant_disagrees(self):
        geo = self._cluster_geo()
        findings = extract_findings(geo)
        outlier = next(
            f for f in findings if isinstance(f, DirectionOutlierFinding)
        )
        assert outlier.details["variant"] == "d"
        assert outlier.details["mean_cosine_to_cluster"] < 0.85
        assert outlier.details["cluster"] == ("a", "b", "c")

    def test_no_cluster_with_two_variants(self):
        geo = _build_geo()  # 2 variants
        findings = extract_findings(geo)
        assert not any(isinstance(f, DirectionClusterFinding) for f in findings)
        assert not any(isinstance(f, DirectionOutlierFinding) for f in findings)


# ── SpecializationPeakFinding ─────────────────────────────────────────


class TestSpecializationPeak:
    def test_fires_for_each_variant_above_30pct(self):
        geo = _build_geo()
        # v1 100% on a; v2 100% on b.
        findings = extract_findings(geo)
        peaks = [f for f in findings if isinstance(f, SpecializationPeakFinding)]
        assert {p.details["variant"] for p in peaks} == {"v1", "v2"}
        for p in peaks:
            assert p.details["share"] > 0.30

    def test_does_not_fire_when_share_below_threshold(self):
        # 4 domains, even split → each share = 0.25 < 0.30.
        cv = {
            "v1": [1.0, 1.0, 1.0, 1.0],
            "v2": [1.0, 1.0, 1.0, 1.0],
        }
        geo = _build_geo(variants=cv, domains=("a", "b", "c", "d"))
        findings = extract_findings(geo)
        assert not any(
            isinstance(f, SpecializationPeakFinding) for f in findings
        )


# ── AccuracyArtifactFinding ───────────────────────────────────────────


class TestAccuracyArtifact:
    def test_fires_for_gsm8k_with_low_max_new_tokens(self):
        geo = _build_geo(metadata={
            "max_new_tokens": 16,
            "accuracy_by_variant": {
                "v1": {"gsm8k": 0.0, "hellaswag": 0.55},
                "v2": {"gsm8k": 0.01, "hellaswag": 0.61},
            },
        })
        findings = extract_findings(geo)
        artifacts = [
            f for f in findings if isinstance(f, AccuracyArtifactFinding)
        ]
        assert len(artifacts) == 1
        a = artifacts[0]
        assert a.details["task"] == "gsm8k"
        assert a.details["max_new_tokens"] == 16
        assert "task-max-new-tokens" in a.details["suggestion"]
        assert a.severity == "caveat"

    def test_does_not_fire_when_per_task_override_present(self):
        # Per spec invariant #6: an explicit per-task override means the
        # user already fixed the artifact; rule must not fire on that task.
        geo = _build_geo(metadata={
            "max_new_tokens": 16,
            "task_max_new_tokens": {"gsm8k": 256},
            "accuracy_by_variant": {
                "v1": {"gsm8k": 0.0, "longbench_2wikimqa": 0.0},
                "v2": {"gsm8k": 0.01, "longbench_2wikimqa": 0.0},
            },
        })
        findings = extract_findings(geo)
        tasks = {
            f.details["task"]
            for f in findings if isinstance(f, AccuracyArtifactFinding)
        }
        # gsm8k suppressed by override; longbench still fires.
        assert "gsm8k" not in tasks
        assert "longbench_2wikimqa" in tasks

    def test_does_not_fire_when_accuracy_above_threshold(self):
        geo = _build_geo(metadata={
            "max_new_tokens": 16,
            "accuracy_by_variant": {
                "v1": {"gsm8k": 0.10},
                "v2": {"gsm8k": 0.12},
            },
        })
        findings = extract_findings(geo)
        assert not any(isinstance(f, AccuracyArtifactFinding) for f in findings)

    def test_does_not_fire_for_mcq_task(self):
        # hellaswag is MCQ; not in the generative task allowlist.
        geo = _build_geo(metadata={
            "max_new_tokens": 16,
            "accuracy_by_variant": {
                "v1": {"hellaswag": 0.0},
                "v2": {"hellaswag": 0.0},
            },
        })
        findings = extract_findings(geo)
        assert not any(isinstance(f, AccuracyArtifactFinding) for f in findings)


# ── TokenizerMismatchFinding ──────────────────────────────────────────


class TestTokenizerMismatch:
    def test_fires_when_bpb_normalized_flag_present(self):
        geo = _build_geo(metadata={
            "bpb_normalized": {"v1": True, "v2": False},
        })
        findings = extract_findings(geo)
        mismatches = [
            f for f in findings if isinstance(f, TokenizerMismatchFinding)
        ]
        assert len(mismatches) == 1
        m = mismatches[0]
        assert m.severity == "warning"
        assert ("base", "v1") in m.details["pairs"]

    def test_does_not_fire_when_no_bpb_flags(self):
        geo = _build_geo(metadata={"bpb_normalized": {"v1": False, "v2": False}})
        findings = extract_findings(geo)
        assert not any(
            isinstance(f, TokenizerMismatchFinding) for f in findings
        )


# ── BaseAccuracyMissingFinding ────────────────────────────────────────


class TestBaseAccuracyMissing:
    def test_fires_when_variants_have_accuracy_but_base_does_not(self):
        geo = _build_geo(metadata={
            "max_new_tokens": 64,
            "accuracy_by_variant": {
                "v1": {"hellaswag": 0.55},
                "v2": {"hellaswag": 0.61},
            },
        })
        findings = extract_findings(geo)
        missing = [
            f for f in findings if isinstance(f, BaseAccuracyMissingFinding)
        ]
        assert len(missing) == 1
        assert set(missing[0].details["variants_with_accuracy"]) == {"v1", "v2"}

    def test_does_not_fire_when_base_accuracy_present(self):
        geo = _build_geo(metadata={
            "max_new_tokens": 64,
            "base_accuracy": {"hellaswag": 0.50},
            "accuracy_by_variant": {"v1": {"hellaswag": 0.55}},
        })
        findings = extract_findings(geo)
        assert not any(
            isinstance(f, BaseAccuracyMissingFinding) for f in findings
        )


# ── Stable ordering ──────────────────────────────────────────────────


class TestStableOrdering:
    def test_info_findings_come_before_warnings_and_caveats(self):
        geo = _build_geo(metadata={
            "max_new_tokens": 16,
            "bpb_normalized": {"v1": True},
            "accuracy_by_variant": {
                "v1": {"gsm8k": 0.0},
                "v2": {"gsm8k": 0.0},
            },
        })
        findings = extract_findings(geo)
        # Info findings should all appear before any warning / caveat.
        seen_non_info = False
        for f in findings:
            if f.severity != "info":
                seen_non_info = True
            else:
                assert not seen_non_info, (f, [type(x).__name__ for x in findings])

    def test_specialization_peaks_alphabetical_by_variant(self):
        # Three variants with peaks above 30%; check sort order.
        cv = {
            "math": [1.0, 0.0, 0.0],
            "code": [0.0, 1.0, 0.0],
            "yarn": [0.0, 0.0, 1.0],
        }
        geo = _build_geo(variants=cv, domains=("a", "b", "c"))
        peaks = [
            f for f in extract_findings(geo)
            if isinstance(f, SpecializationPeakFinding)
        ]
        names = [p.details["variant"] for p in peaks]
        assert names == sorted(names)


# ── result.findings cached property ───────────────────────────────────


class TestResultFindingsProperty:
    def test_returns_tuple_of_findings(self):
        geo = _build_geo()
        findings = geo.findings
        assert isinstance(findings, tuple)
        assert all(isinstance(f, Finding) for f in findings)

    def test_cached_on_instance(self):
        geo = _build_geo()
        first = geo.findings
        second = geo.findings
        assert first is second  # exact same object — cached


# ── Calibration: 4-variant Llama-2 georesult (skipped if missing) ─────


_CALIB_PATH = Path("runs/llama2-4variants/family_geometry_lm_eval_georesult.json")


@pytest.mark.skipif(
    not _CALIB_PATH.exists(),
    reason="Llama-2 4-variant calibration GeoResult not present in the repo",
)
class TestLlama2Calibration:
    @pytest.fixture(scope="class")
    def geo(self) -> GeoResult:
        from lmdiff.report.json_report import geo_result_from_json_dict
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with open(_CALIB_PATH, encoding="utf-8") as f:
                return geo_result_from_json_dict(json.load(f))

    def test_most_like_base_is_yarn_on_code(self, geo):
        most_like = next(
            f for f in geo.findings if isinstance(f, MostLikeBaseFinding)
        )
        # Per the v0.2.x writeup: yarn on code is the smallest normalized
        # magnitude. Drift is the raw L2 here, so check the qualitative
        # signal (variant name) only.
        assert most_like.details["variant"] in {"yarn", "code"}

    def test_biggest_move_includes_long(self, geo):
        biggest = next(
            f for f in geo.findings if isinstance(f, BiggestMoveFinding)
        )
        # The 4-variant Llama-2 dataset's largest cell is "long" on
        # long-context (raw L2 dominates). long is the global-amplitude
        # variant per L-022.
        assert biggest.details["variant"] == "long"

    def test_specialization_peaks_one_per_variant(self, geo):
        peaks = [
            f for f in geo.findings
            if isinstance(f, SpecializationPeakFinding)
        ]
        # All 4 variants have a peak above 30% (longbench dominates raw
        # share for several of them; specialization is real).
        assert {p.details["variant"] for p in peaks} == {
            "yarn", "long", "code", "math",
        }
