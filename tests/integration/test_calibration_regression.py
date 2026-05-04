"""Calibration regression test for the v0.4.0 backend cutover.

The hard contract for cutover safety: the new HFEngine pipeline must
produce a GeoResult whose every numeric field matches the v0.3.2
calibration baseline (committed at
``tests/fixtures/calibration_v032_baseline.json``) within 1e-6 per
element on the canonical Llama-2 4-variant case.

Marked ``slow`` AND ``gpu``: requires a GPU big enough for two
Llama-2-7B variants resident at once (~28 GB VRAM peak after the
v0.3.2 engine-reuse fix). Skipped by default ``pytest -m "not slow and
not gpu"`` runs.

If this test fails after a cutover or backend change, the cutover does
not ship. No exceptions for "we found a bug in v0.3.2 too" — that's a
separate commit, not part of cutover (see L-028).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

BASELINE_PATH = Path(__file__).parent.parent / "fixtures" / "calibration_v032_baseline.json"
TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not BASELINE_PATH.exists():
        pytest.skip(
            f"calibration baseline not present at {BASELINE_PATH}. "
            "Generate via the snippet at the bottom of "
            "docs/internal/v040_cutover_audit.md and commit before "
            "running this test."
        )
    with BASELINE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def cutover_result() -> dict:
    """Run family() through the new HFEngine pipeline with identical
    inputs to the baseline-generation script. Returns the to_json_dict
    payload for byte-comparison."""
    import lmdiff
    from lmdiff.report.json_report import to_json_dict

    result = lmdiff.family(
        base="meta-llama/Llama-2-7b-hf",
        variants={
            "yarn": "NousResearch/Yarn-Llama-2-7b-128k",
            "long": "togethercomputer/LLaMA-2-7B-32K",
            "code": "codellama/CodeLlama-7b-hf",
            "math": "EleutherAI/llemma_7b",
        },
        probes="lm_eval:hellaswag+arc_challenge+gsm8k+mmlu_college_computer_science+longbench_2wikimqa",
        n_probes=100,
        max_new_tokens=16,
        task_overrides={
            "gsm8k": {"max_new_tokens": 256},
            "longbench_2wikimqa": {"max_new_tokens": 128},
        },
        seed=42,
    )
    payload = to_json_dict(result)
    payload.pop("generated_at", None)  # timestamp would always differ
    return payload


# ── Per-field equivalence ────────────────────────────────────────────


def test_variant_names_match(baseline, cutover_result):
    assert baseline["variant_names"] == cutover_result["variant_names"]


def test_n_probes_match(baseline, cutover_result):
    assert baseline["n_probes"] == cutover_result["n_probes"]


def test_change_vectors_match(baseline, cutover_result):
    """The most fundamental contract: every per-probe δ value matches
    within 1e-6 on every (variant, probe) cell."""
    for v in baseline["variant_names"]:
        bvec = baseline["change_vectors"][v]
        cvec = cutover_result["change_vectors"][v]
        assert len(bvec) == len(cvec), v
        for i, (b, c) in enumerate(zip(bvec, cvec)):
            assert abs(b - c) < TOLERANCE, (
                f"change_vectors[{v}][{i}]: baseline={b}, cutover={c}, "
                f"diff={abs(b-c)}"
            )


def test_cosine_matrix_match(baseline, cutover_result):
    for a in baseline["variant_names"]:
        for b in baseline["variant_names"]:
            assert abs(
                baseline["cosine_matrix"][a][b]
                - cutover_result["cosine_matrix"][a][b]
            ) < TOLERANCE, (a, b)


def test_selective_cosine_matrix_match(baseline, cutover_result):
    for a in baseline["variant_names"]:
        for b in baseline["variant_names"]:
            assert abs(
                baseline["selective_cosine_matrix"][a][b]
                - cutover_result["selective_cosine_matrix"][a][b]
            ) < TOLERANCE, (a, b)


def test_magnitudes_match(baseline, cutover_result):
    for v in baseline["variant_names"]:
        assert abs(
            baseline["magnitudes"][v] - cutover_result["magnitudes"][v]
        ) < TOLERANCE, v


def test_magnitudes_normalized_match(baseline, cutover_result):
    for v in baseline["variant_names"]:
        assert abs(
            baseline["magnitudes_normalized"][v]
            - cutover_result["magnitudes_normalized"][v]
        ) < TOLERANCE, v


def test_per_domain_normalized_match(baseline, cutover_result):
    """v0.3.2 added magnitudes_per_domain_normalized; verify the cutover
    preserves it field-for-field."""
    for v in baseline["variant_names"]:
        for d in baseline["magnitudes_per_domain_normalized"][v]:
            b = baseline["magnitudes_per_domain_normalized"][v][d]
            c = cutover_result["magnitudes_per_domain_normalized"][v][d]
            assert abs(b - c) < TOLERANCE, (v, d, b, c)


def test_share_per_domain_match(baseline, cutover_result):
    for v in baseline["variant_names"]:
        for d in baseline["share_per_domain"][v]:
            b = baseline["share_per_domain"][v][d]
            c = cutover_result["share_per_domain"][v][d]
            assert abs(b - c) < TOLERANCE, (v, d, b, c)


def test_probe_count_and_distribution_match(baseline, cutover_result):
    """Per-domain probe counts must be identical — confirms the lm_eval
    probe loader (per-task n_probes from v0.3.2 PR #8) is wired the
    same way through the new pipeline."""
    assert len(baseline["probe_domains"]) == len(cutover_result["probe_domains"])
    assert Counter(baseline["probe_domains"]) == Counter(
        cutover_result["probe_domains"],
    )


def test_findings_match(baseline, cutover_result):
    """Findings are derived from the numeric fields above; if those all
    match within tolerance, the findings tuple must be identical."""
    import lmdiff

    b_result = lmdiff.load_result(str(BASELINE_PATH))
    # cutover_result is already a dict; reconstruct via from_dict.
    from lmdiff.report.json_report import geo_result_from_json_dict
    c_result = geo_result_from_json_dict(cutover_result)

    b_findings = sorted([
        (type(f).__name__, f.summary) for f in b_result.findings
    ])
    c_findings = sorted([
        (type(f).__name__, f.summary) for f in c_result.findings
    ])
    assert b_findings == c_findings, (
        "finding sets diverged — one of the upstream numeric fields "
        "is outside tolerance even though the per-field test passed"
    )
