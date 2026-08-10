"""Calibration regression test for the v0.4.0 backend cutover.

The hard contract for cutover safety: the new HFEngine pipeline must
produce a GeoResult whose every numeric field matches the calibration
baseline within 1e-6 per element on the canonical Llama-2 4-variant
case.

v0.4.1 update: the formula change (Q9.10 Formula A) shifts pdn values
by factor √T̄_d everywhere — the v0.3.2 fixture is no longer valid as
the v0.4.1 baseline. Fixture name updated to
``calibration_v041_4variant_baseline.json``; the file is regenerated
on a GPU box via ``scripts/_regenerate_v041_4variant_fixture.py``
(commit 8 of v0.4.1 PR). Until that regeneration lands the test
automatically skips.

Per-domain assertions are selected per (variant, domain) cell by the
validity status encoded in ``_v041_4variant_spec``:

  - ``full`` / ``partial`` → numeric assertion within 1e-6
  - ``variant_only`` / ``out_of_range`` → ``share`` and ``pdn`` must
    both be ``None``

Under the v0.4.1 ``min_valid_fraction`` floor, long-context is the only
non-``full`` column — ``variant_only`` for yarn / long / code,
``out_of_range`` for math. 16 measured cells, 4 unmeasured. Every
variant here decodes greedily, so there is no looser sample-decode
tolerance tier (contrast the 7-variant test).

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

from tests.integration._v041_4variant_spec import (
    ALL_VARIANTS,
    EXPECTED_DOMAIN_STATUS,
    MEASURED_CELLS,
    UNMEASURED_CELLS,
)

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

# v0.4.1 fixture path. Regenerated on GPU via
# scripts/_regenerate_v041_4variant_fixture.py (commit 8). Until then
# the per-test skip below auto-skips the suite.
BASELINE_PATH = (
    Path(__file__).parent.parent / "fixtures"
    / "calibration_v041_4variant_baseline.json"
)
TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not BASELINE_PATH.exists():
        pytest.skip(
            f"v0.4.1 calibration baseline not present at "
            f"{BASELINE_PATH.name}. Regenerate on a GPU box via "
            "``python scripts/_regenerate_v041_4variant_fixture.py`` "
            "and commit the produced JSON. The v0.4.1 formula change "
            "(see L-033 / docs/methodology/normalization.md) makes the "
            "pre-v0.4.1 fixture incompatible — pdn values shift by "
            "factor √T̄_d."
        )
    with BASELINE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def cutover_result() -> dict:
    """Run family() through the new HFEngine pipeline with identical
    inputs to the baseline-generation script. Returns the to_json_dict
    payload for byte-comparison.

    Kwargs come from ``_v041_4variant_spec.build_run_kwargs()`` —
    same source as ``scripts/_regenerate_v041_4variant_fixture.py`` to
    eliminate "did the regen script run the same call?" risk.
    """
    import lmdiff
    from lmdiff.report.json_report import to_json_dict
    from tests.integration._v041_4variant_spec import build_run_kwargs

    result = lmdiff.family(**build_run_kwargs())
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


def test_fixture_domain_status_matches_spec(baseline):
    """The fixture's own validity classification must match what the
    spec encodes. CPU-only guard: if a future regeneration changes
    validity behaviour (a different ``min_valid_fraction``, a new
    context-window fallback), this fails fast instead of silently
    reshaping which cells below get a numeric vs a ``None``
    assertion."""
    assert baseline["domain_status"] == EXPECTED_DOMAIN_STATUS


def test_run_domain_status_matches_spec(cutover_result):
    """Same contract, against the live run."""
    assert cutover_result["domain_status"] == EXPECTED_DOMAIN_STATUS


@pytest.mark.parametrize(("variant", "domain"), MEASURED_CELLS)
def test_per_domain_normalized_match(baseline, cutover_result, variant, domain):
    """v0.3.2 added magnitudes_per_domain_normalized; verify the cutover
    preserves it field-for-field on every measurable cell."""
    b = baseline["magnitudes_per_domain_normalized"][variant][domain]
    c = cutover_result["magnitudes_per_domain_normalized"][variant][domain]
    assert b is not None, f"spec says {variant}/{domain} is measured, fixture has None"
    assert c is not None, f"spec says {variant}/{domain} is measured, run produced None"
    assert abs(b - c) < TOLERANCE, (variant, domain, b, c)


@pytest.mark.parametrize(("variant", "domain"), MEASURED_CELLS)
def test_share_per_domain_match(baseline, cutover_result, variant, domain):
    b = baseline["share_per_domain"][variant][domain]
    c = cutover_result["share_per_domain"][variant][domain]
    assert b is not None, f"spec says {variant}/{domain} is measured, fixture has None"
    assert c is not None, f"spec says {variant}/{domain} is measured, run produced None"
    assert abs(b - c) < TOLERANCE, (variant, domain, b, c)


@pytest.mark.parametrize(("variant", "domain"), UNMEASURED_CELLS)
def test_unmeasured_cells_are_none(baseline, cutover_result, variant, domain):
    """``variant_only`` / ``out_of_range`` cells carry ``None``, not a
    number and not 0.0 — the sentinel distinguishes "not measured" from
    "measured zero drift". Before the v0.4.1 ``min_valid_fraction``
    floor these were ``partial`` and carried shares built on 9 of 100
    probes."""
    for field in ("share_per_domain", "magnitudes_per_domain_normalized"):
        b = baseline[field][variant][domain]
        c = cutover_result[field][variant][domain]
        assert b is None, (
            f"{field}[{variant}][{domain}]: fixture should be None for "
            f"status {EXPECTED_DOMAIN_STATUS[variant][domain]}, got {b}"
        )
        assert c is None, (
            f"{field}[{variant}][{domain}]: run should be None for "
            f"status {EXPECTED_DOMAIN_STATUS[variant][domain]}, got {c}"
        )


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_share_rows_sum_to_one_over_measured_cells(cutover_result, variant):
    """Excluding a domain renormalizes the rest — the surviving cells
    must still form a distribution."""
    row = cutover_result["share_per_domain"][variant]
    total = sum(x for x in row.values() if x is not None)
    assert abs(total - 1.0) < 1e-9, (
        f"share_per_domain[{variant}] sums to {total}, expected 1.0 over "
        f"non-None cells"
    )


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
