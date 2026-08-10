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
path are defined in ``_v041_7variant_spec.py`` so
``scripts/_regenerate_v041_7variant_fixture.py`` runs the *same*
call — no "did the regen script match the test?" risk.

Assertions are per (variant, domain) cell, selected by the validity
status the spec encodes:

  - ``full`` / ``partial`` → numeric assertion against the fixture
  - ``variant_only`` / ``out_of_range`` → both ``share`` and ``pdn``
    must be ``None``

Under v0.4.1's ``min_valid_fraction`` floor the long-context column is
the only non-``full`` one: ``variant_only`` for yarn / long / code,
``out_of_range`` for the other four. 28 measured cells, 7 unmeasured.

Tolerance rationale:
  - ``change_vectors`` (per-probe δ): 1e-6 byte-equivalence for the
    6 deterministic variants; SKIPPED for any variant in
    ``SAMPLE_DECODE_VARIANTS_LEGACY`` (currently empty under Fix 3 —
    ``temp_1.5`` is reproducible given a pinned seed; left as a
    constant in the spec for future variants that genuinely can't
    be byte-checked, e.g. best_of_n with hardware-non-deterministic
    argmax ties)
  - ``share_per_domain`` / ``magnitudes_per_domain_normalized`` on
    measured cells: 1e-6 for deterministic variants. Variants in
    ``SAMPLE_DECODE_VARIANTS`` keep the pre-v0.4.1 tiers — 2pp for
    share, 1e-3 for pdn — because sampling amplifies residual float
    jitter through the δ → pdn → share chain more than it shows in
    raw δ. The 2pp tier is what catches the 60→94% (system_prompt)
    and 34→5% (temp_1.5) regressions Fix 1 + Fix 2 address.

If any assertion fails, the cutover / formula change did not safely
preserve v0.4.1 behavior. v0.4.1 doesn't ship.

The fixture is the v0.4.1 baseline (``calibration_v041_7variant_summary.json``).
v0.3.2 / v0.4.0 fixtures are no longer the contract — the formula
change (Q9.10 Formula A) and validity framework shift every value,
see L-033 / docs/methodology/normalization.md. v0.3.2's ``temp_1.5``
outputs were also produced under unpinned RNG and are no longer the
contract — see L-031.
"""
from __future__ import annotations

import json

import pytest

from tests.integration._v041_7variant_spec import (
    ALL_VARIANTS,
    BYTE_EQUIVALENT_VARIANTS,
    EXPECTED_DOMAIN_STATUS,
    FIXTURE_PATH,
    MEASURED_CELLS,
    SAMPLE_DECODE_VARIANTS,
    SAMPLE_DECODE_VARIANTS_LEGACY,
    UNMEASURED_CELLS,
    build_run_kwargs,
)

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

# Tolerances per metric. Deterministic variants get byte-equivalence on
# the derived per-domain metrics; sample-decode variants keep the looser
# pre-v0.4.1 tiers, because sampling amplifies residual float jitter
# through the δ → pdn → share chain more than it shows in raw δ.
TOL_CHANGE_VECTORS = 1e-6
TOL_SHARE = 1e-6                 # deterministic variants
TOL_SHARE_PCT_POINTS = 2.0       # sample-decode variants, user-spec
TOL_PDN = 1e-6                   # deterministic variants
TOL_PDN_SAMPLE = 1e-3            # sample-decode variants
TOL_OVERALL_NORM = 1e-3

# ``SAMPLE_DECODE_VARIANTS_LEGACY`` (empty under Fix 3) gates whether a
# variant's raw ``change_vectors`` are byte-checked at all;
# ``SAMPLE_DECODE_VARIANTS`` ({"temp_1.5"}) only selects the looser
# derived-metric tolerance tier. Distinct concepts, both retained.
assert SAMPLE_DECODE_VARIANTS_LEGACY <= set(ALL_VARIANTS)


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"7-variant fixture not present at {FIXTURE_PATH}. "
            "Regenerate by running "
            "``python scripts/_regenerate_v041_7variant_fixture.py`` "
            "on a GPU box, then commit the produced JSON. The script "
            "uses the same family() kwargs as this test."
        )
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cutover_result() -> dict:
    """Run the 7-variant family() through the new HFEngine pipeline
    and return the to_json_dict payload for comparison.

    Kwargs come from ``_v041_7variant_spec.build_run_kwargs()`` —
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


# ── domain_status (v0.4.1 validity framework) ───────────────────────


def test_fixture_domain_status_matches_spec(baseline):
    """The fixture's own validity classification must match what the
    spec encodes. CPU-only guard: if a future regeneration changes
    validity behaviour (a different ``min_valid_fraction``, a new
    context-window fallback), this fails at collection-adjacent speed
    instead of silently reshaping which cells below get a numeric
    assertion vs a ``None`` assertion."""
    assert baseline["domain_status"] == EXPECTED_DOMAIN_STATUS


def test_run_domain_status_matches_spec(cutover_result):
    """Same contract, against the live run."""
    assert cutover_result["domain_status"] == EXPECTED_DOMAIN_STATUS


# ── share_per_domain / pdn on measured (full | partial) cells ───────


@pytest.mark.parametrize(("variant", "domain"), MEASURED_CELLS)
def test_share_per_domain_measured_cells(
    baseline, cutover_result, variant, domain,
):
    """The headline showcase metric, on cells the validity framework
    says are measurable.

    Deterministic variants assert byte-equivalence (1e-6). Sample-decode
    variants keep the 2pp tier, which is what caught the 60→94%
    (system_prompt) and 34→5% (temp_1.5) regressions Fix 1 + Fix 2 +
    Fix 3 address — tighter than any natural cross-run variation, loose
    enough for residual hardware float jitter (BF16 attention reductions
    on Blackwell)."""
    b = baseline["share_per_domain"][variant][domain]
    c = cutover_result["share_per_domain"][variant][domain]
    assert b is not None, f"spec says {variant}/{domain} is measured, fixture has None"
    assert c is not None, f"spec says {variant}/{domain} is measured, run produced None"

    if variant in SAMPLE_DECODE_VARIANTS:
        diff_pp = abs(b - c) * 100.0
        assert diff_pp <= TOL_SHARE_PCT_POINTS, (
            f"share_per_domain[{variant}][{domain}]: baseline={b*100:.2f}%, "
            f"cutover={c*100:.2f}%, diff={diff_pp:.2f}pp "
            f"(tolerance: {TOL_SHARE_PCT_POINTS}pp, sample-decode tier)"
        )
    else:
        assert abs(b - c) < TOL_SHARE, (
            f"share_per_domain[{variant}][{domain}]: baseline={b}, "
            f"cutover={c}, diff={abs(b-c)} (tolerance: {TOL_SHARE})"
        )


@pytest.mark.parametrize(("variant", "domain"), MEASURED_CELLS)
def test_pdn_measured_cells(baseline, cutover_result, variant, domain):
    b = baseline["magnitudes_per_domain_normalized"][variant][domain]
    c = cutover_result["magnitudes_per_domain_normalized"][variant][domain]
    assert b is not None, f"spec says {variant}/{domain} is measured, fixture has None"
    assert c is not None, f"spec says {variant}/{domain} is measured, run produced None"

    tol = TOL_PDN_SAMPLE if variant in SAMPLE_DECODE_VARIANTS else TOL_PDN
    assert abs(b - c) < tol, (
        f"pdn[{variant}][{domain}]: baseline={b}, cutover={c}, "
        f"diff={abs(b-c)} (tolerance: {tol})"
    )


# ── share_per_domain / pdn on unmeasured cells — must be None ──────


@pytest.mark.parametrize(("variant", "domain"), UNMEASURED_CELLS)
def test_unmeasured_cells_are_none(
    baseline, cutover_result, variant, domain,
):
    """``variant_only`` / ``out_of_range`` cells carry ``None``, not a
    number and not 0.0 — the sentinel distinguishes "not measured" from
    "measured zero drift".

    Before the ``min_valid_fraction`` floor these cells were ``partial``
    and carried real shares built on 9 of 100 probes (27.6 % for
    ``temp_1.5``, 18.4 % for ``chat``). This is the assertion that pins
    them out of the share."""
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
