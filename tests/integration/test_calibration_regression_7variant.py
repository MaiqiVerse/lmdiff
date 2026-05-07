"""7-variant calibration regression — the wider gate for v0.4.0 cutover.

The 4-variant calibration (``test_calibration_regression.py``) only
covers variants without ``system_prompt`` and with greedy decoding —
the easy path. This test covers the two variants that exercise the
v0.4.0 features the 4-variant test missed:

  - ``system_prompt`` — runtime-only Config modification, exercises
    ``HFEngine.score(prefix_text=…)`` and the split-tokenize path
    that Fix 2 added (PR #15 fixup commits)
  - ``temp_1.5`` — sample decoding with explicit ``top_k=0``,
    exercises Fix 1's top_k passthrough; reproducible under Fix 3
    via ``family(seed=42)`` (was unpinned in v0.3.2)

Plus the 5 unique-model variants (yarn/long/code/math/chat) for
spot-checking against the v0.4.0 baseline.

The exact ``family()`` kwargs the test runs against and the fixture
path are defined in ``_v040_7variant_spec.py`` so
``scripts/_regenerate_v040_7variant_fixture.py`` runs the *same*
call — no "did the regen script match the test?" risk.

Tolerance rationale:
  - ``change_vectors`` (per-probe δ): 1e-6 byte-equivalence for the
    6 deterministic variants; SKIPPED for any variant in
    ``SAMPLE_DECODE_VARIANTS_LEGACY`` (currently empty under Fix 3 —
    ``temp_1.5`` is reproducible given a pinned seed; left as a
    constant in the spec for future variants that genuinely can't
    be byte-checked, e.g. best_of_n with hardware-non-deterministic
    argmax ties)
  - ``share_per_domain``: 2pp for ALL variants. The user-visible
    headline metric; 2pp catches the 60→94% (system_prompt) and
    34→5% (temp_1.5) regressions Fix 1 + Fix 2 address while staying
    loose enough for any residual hardware float jitter
  - ``magnitudes_per_domain_normalized``: 1e-3 (slightly looser than
    1e-6 because pdn participates in the share normalization)

If any assertion fails, the cutover did not safely preserve v0.4.0
behavior. v0.4.0 doesn't ship.

The fixture is the v0.4.0 baseline (``calibration_v040_…``), not the
v0.3.2 fixture. v0.3.2's ``temp_1.5`` outputs were produced under
unpinned RNG and are no longer the contract — see L-031.
"""
from __future__ import annotations

import json

import pytest

from tests.integration._v040_7variant_spec import (
    ALL_VARIANTS,
    BYTE_EQUIVALENT_VARIANTS,
    FIXTURE_PATH,
    SAMPLE_DECODE_VARIANTS_LEGACY,
    build_run_kwargs,
)

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

# Tolerances per metric.
TOL_CHANGE_VECTORS = 1e-6
TOL_SHARE_PCT_POINTS = 2.0       # 2pp, user-spec
TOL_PDN = 1e-3
TOL_OVERALL_NORM = 1e-3

# Re-export under the test's historical name so the parameterize
# decorators below stay readable.
SAMPLE_DECODE_VARIANTS = SAMPLE_DECODE_VARIANTS_LEGACY


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"7-variant fixture not present at {FIXTURE_PATH}. "
            "Regenerate by running "
            "``python scripts/_regenerate_v040_7variant_fixture.py`` "
            "on a GPU box, then commit the produced JSON. The script "
            "uses the same family() kwargs as this test."
        )
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cutover_result() -> dict:
    """Run the 7-variant family() through the new HFEngine pipeline
    and return the to_json_dict payload for comparison.

    Kwargs come from ``_v040_7variant_spec.build_run_kwargs()`` —
    same source as the regeneration script, so test and fixture can't
    drift.
    """
    import lmdiff
    from lmdiff.report.json_report import to_json_dict

    result = lmdiff.family(**build_run_kwargs())
    payload = to_json_dict(result)
    payload.pop("generated_at", None)
    return payload


# ── Structural / probe-distribution match ───────────────────────────


def test_variant_names_match(baseline, cutover_result):
    assert baseline["variant_names"] == cutover_result["variant_names"]


def test_n_probes_match(baseline, cutover_result):
    assert baseline["n_probes"] == cutover_result["n_probes"]


def test_probe_domain_distribution_match(baseline, cutover_result):
    """Per-domain probe count must be identical (per-task n_probes
    semantics from PR #8)."""
    from collections import Counter
    assert Counter(baseline["probe_domains"]) == Counter(
        cutover_result["probe_domains"],
    )


# ── Per-variant change_vectors (byte-equivalent variants) ───────────


@pytest.mark.parametrize("variant", BYTE_EQUIVALENT_VARIANTS)
def test_change_vectors_match_for_deterministic_variants(
    baseline, cutover_result, variant,
):
    """Every byte-equivalent variant (under Fix 3, all 7) must
    reproduce per-probe δ values exactly. ``system_prompt`` exercises
    Fix 2's ``prefix_text`` threading; ``temp_1.5`` exercises Fix 1's
    ``top_k`` passthrough + Fix 3's seed plumbing. Without any of
    those, the GPU 7-variant demo regressed 60→94% on commonsense
    (system_prompt) and 34→5% on reasoning (temp_1.5)."""
    bvec = baseline["change_vectors"][variant]
    cvec = cutover_result["change_vectors"][variant]
    assert len(bvec) == len(cvec), variant
    for i, (b, c) in enumerate(zip(bvec, cvec)):
        assert abs(b - c) < TOL_CHANGE_VECTORS, (
            f"change_vectors[{variant}][{i}]: baseline={b}, cutover={c}, "
            f"diff={abs(b-c)}"
        )


# ── share_per_domain (ALL variants, 2pp tolerance) ──────────────────


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_share_per_domain_within_2pp(baseline, cutover_result, variant):
    """The headline showcase metric. 2pp tolerance catches the
    60→94% (system_prompt) and 34→5% (temp_1.5) regressions that
    Fix 1 + Fix 2 + Fix 3 address. Tighter than any natural cross-run
    variation but loose enough to accommodate residual hardware float
    jitter (BF16 attention reductions on Blackwell). Applied to every
    variant — including the legacy-unstable ones in
    ``SAMPLE_DECODE_VARIANTS_LEGACY`` (currently empty, but the test
    surface for adding one in the future)."""
    b_row = baseline["share_per_domain"][variant]
    c_row = cutover_result["share_per_domain"][variant]
    assert set(b_row.keys()) == set(c_row.keys()), variant
    for d in b_row:
        b_pct = b_row[d] * 100.0
        c_pct = c_row[d] * 100.0
        diff_pp = abs(b_pct - c_pct)
        assert diff_pp <= TOL_SHARE_PCT_POINTS, (
            f"share_per_domain[{variant}][{d}]: baseline={b_pct:.2f}%, "
            f"cutover={c_pct:.2f}%, diff={diff_pp:.2f}pp "
            f"(tolerance: {TOL_SHARE_PCT_POINTS}pp)"
        )


# ── magnitudes_per_domain_normalized (byte-equivalent variants) ─────


@pytest.mark.parametrize("variant", BYTE_EQUIVALENT_VARIANTS)
def test_pdn_match_for_deterministic_variants(
    baseline, cutover_result, variant,
):
    b_pdn = baseline["magnitudes_per_domain_normalized"][variant]
    c_pdn = cutover_result["magnitudes_per_domain_normalized"][variant]
    assert set(b_pdn.keys()) == set(c_pdn.keys()), variant
    for d in b_pdn:
        b, c = b_pdn[d], c_pdn[d]
        assert abs(b - c) < TOL_PDN, (
            f"pdn[{variant}][{d}]: baseline={b}, cutover={c}, "
            f"diff={abs(b-c)} (tolerance: {TOL_PDN})"
        )


# ── magnitudes_normalized (overall, byte-equivalent variants) ───────


@pytest.mark.parametrize("variant", BYTE_EQUIVALENT_VARIANTS)
def test_overall_normalized_for_deterministic_variants(
    baseline, cutover_result, variant,
):
    b = baseline["magnitudes_normalized"][variant]
    c = cutover_result["magnitudes_normalized"][variant]
    assert abs(b - c) < TOL_OVERALL_NORM, (
        f"magnitudes_normalized[{variant}]: baseline={b}, "
        f"cutover={c}, diff={abs(b-c)} (tolerance: {TOL_OVERALL_NORM})"
    )
