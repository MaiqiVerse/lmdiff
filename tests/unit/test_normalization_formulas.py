"""Regression tests for the per-domain normalization helpers.

Updated for v0.4.1 (Q9.10 Formula A):

  pdn[v][d]                = sqrt(mean_{i∈d∧valid}(δ_i²))   [units: nats/token]
  share_per_domain[v][d]   = pdn[v][d]² / Σ_d' pdn[v][d']²  over valid domains;
                             None when domain_status is out_of_range / variant_only
  magnitudes_normalized[v] = sqrt( (1/D_valid) · Σ_d pdn[v][d]² )  over valid domains

These tests verify:
- new pdn formula on hand-computable inputs (plain unweighted RMS)
- share + overall handle ``None`` sentinels for invalid domains
- ``avg_tokens_per_probe`` is accepted for backward compat but NOT used
  under Formula A (legacy callers passing it must still work)
- domain_status=None mode (legacy ChangeGeometry path) works

Tests for v0.3.2 √T̄ formula behavior, the v6 §13 mockup numbers
(66%/51%), and v3/v4-load-recompute were removed in v0.4.1 as the
formula they verified no longer applies. See L-033 for the full
context of the formula change.
"""
from __future__ import annotations

import math

import pytest

from lmdiff.geometry import (
    GeoResult,
    _compute_overall_normalized_from_pdn,
    _compute_per_domain_normalized,
    _compute_share_per_domain,
)


# ── Hand-computable Formula A sanity ─────────────────────────────────


class TestPerDomainNormalizedFormula:
    """pdn[v][d] = sqrt(mean(δ²)) over valid probes — Q9.10 Formula A."""

    def test_single_domain_rms(self):
        # Single domain with 2 probes, δ = [3, 4].
        # mean(δ²) = (9+16)/2 = 12.5, pdn = sqrt(12.5).
        cv = {"v": [3.0, 4.0]}
        pdn = _compute_per_domain_normalized(
            ["v"], cv, ("d",) * 2, (10.0, 10.0),
        )
        assert pdn["v"]["d"] == pytest.approx(math.sqrt(12.5))

    def test_two_domains_separated(self):
        # δ = [3, 4, 6, 8], domains = [a, a, b, b].
        # pdn[v][a] = sqrt((9+16)/2) = sqrt(12.5)
        # pdn[v][b] = sqrt((36+64)/2) = sqrt(50.0)
        cv = {"v": [3.0, 4.0, 6.0, 8.0]}
        pdn = _compute_per_domain_normalized(
            ["v"], cv, ("a", "a", "b", "b"), (10.0,) * 4,
        )
        assert pdn["v"]["a"] == pytest.approx(math.sqrt(12.5))
        assert pdn["v"]["b"] == pytest.approx(math.sqrt(50.0))

    def test_avg_tokens_unused_under_formula_a(self):
        # Same δ, drastically different T_i values. Formula A doesn't
        # use T at all, so pdn is identical regardless of T_i.
        cv = {"v": [1.0, 1.0, 1.0, 1.0]}
        pdn_short = _compute_per_domain_normalized(
            ["v"], cv, ("d",) * 4, (10.0,) * 4,
        )
        pdn_long = _compute_per_domain_normalized(
            ["v"], cv, ("d",) * 4, (10000.0,) * 4,
        )
        assert pdn_short["v"]["d"] == pdn_long["v"]["d"]
        assert pdn_short["v"]["d"] == pytest.approx(1.0)  # mean(1²) = 1

    def test_empty_probe_domains_returns_empty(self):
        assert _compute_per_domain_normalized(
            ["v"], {"v": [1.0]}, (), (10.0,),
        ) == {}

    def test_avg_tokens_not_required_for_formula(self):
        # Legacy v0.3.2 path required avg_tokens_per_probe to be
        # populated. Formula A doesn't use it; passing empty tuple is
        # fine as long as probe_domains is populated.
        cv = {"v": [3.0]}
        pdn = _compute_per_domain_normalized(["v"], cv, ("d",), ())
        assert pdn["v"]["d"] == pytest.approx(3.0)  # sqrt(9/1)


# ── Validity-aware None tagging (v0.4.1 new behavior) ───────────────


class TestValidityAwarePdn:
    """When domain_status is provided, out_of_range / variant_only
    domains return None instead of a numeric value."""

    def test_full_status_computes_normally(self):
        cv = {"v": [3.0, 4.0]}
        pdn = _compute_per_domain_normalized(
            ["v"], cv, ("d",) * 2, (10.0,) * 2,
            domain_status={"v": {"d": "full"}},
        )
        assert pdn["v"]["d"] == pytest.approx(math.sqrt(12.5))

    def test_partial_status_computes_normally(self):
        # partial = some probes valid, some not. The change_vectors
        # passed to the formula is already filtered to valid probes by
        # the upstream global NaN filter.
        cv = {"v": [3.0, 4.0]}
        pdn = _compute_per_domain_normalized(
            ["v"], cv, ("d",) * 2, (10.0,) * 2,
            domain_status={"v": {"d": "partial"}},
        )
        assert pdn["v"]["d"] == pytest.approx(math.sqrt(12.5))

    def test_out_of_range_returns_none(self):
        cv = {"v": [3.0, 4.0]}
        pdn = _compute_per_domain_normalized(
            ["v"], cv, ("d",) * 2, (10.0,) * 2,
            domain_status={"v": {"d": "out_of_range"}},
        )
        assert pdn["v"]["d"] is None

    def test_variant_only_returns_none(self):
        cv = {"v": [3.0, 4.0]}
        pdn = _compute_per_domain_normalized(
            ["v"], cv, ("d",) * 2, (10.0,) * 2,
            domain_status={"v": {"d": "variant_only"}},
        )
        assert pdn["v"]["d"] is None

    def test_legacy_no_status_treats_as_full(self):
        # When domain_status is None (legacy path, e.g. v0.2.x
        # ChangeGeometry.analyze still in use), every domain is
        # treated as "full" → never None.
        cv = {"v": [3.0, 4.0]}
        pdn = _compute_per_domain_normalized(
            ["v"], cv, ("d",) * 2, (10.0,) * 2,
            domain_status=None,
        )
        assert pdn["v"]["d"] is not None
        assert pdn["v"]["d"] == pytest.approx(math.sqrt(12.5))


# ── Share formula uses pdn ──────────────────────────────────────────


def _make_result_with_pdn(pdn_by_v: dict[str, dict[str, float | None]]) -> GeoResult:
    """Build a minimal GeoResult with magnitudes_per_domain_normalized
    populated. share_per_domain is left for _compute_share_per_domain
    to derive."""
    variant_names = list(pdn_by_v.keys())
    return GeoResult(
        base_name="base",
        variant_names=variant_names,
        n_probes=0,
        magnitudes={v: 0.0 for v in variant_names},
        cosine_matrix={v: {v: 1.0} for v in variant_names},
        change_vectors={v: [] for v in variant_names},
        per_probe={v: {} for v in variant_names},
        magnitudes_per_domain_normalized=pdn_by_v,
    )


class TestShareFormulaUsesPDN:
    """share[v][d] = pdn[v][d]² / Σ pdn²; None for invalid pdn."""

    def test_share_ratios_match_pdn_squared(self):
        result = _make_result_with_pdn({
            "v": {"a": 1.0, "b": 2.0},  # 1² : 4² → share = 1/5 : 4/5
        })
        share = _compute_share_per_domain(result)
        assert share["v"]["a"] == pytest.approx(1.0 / 5.0)
        assert share["v"]["b"] == pytest.approx(4.0 / 5.0)

    def test_rows_sum_to_one_over_valid(self):
        result = _make_result_with_pdn({
            "v1": {"a": 0.5, "b": 1.5, "c": 0.7},
            "v2": {"a": 2.0, "b": 0.0, "c": 0.3},
        })
        share = _compute_share_per_domain(result)
        for variant, row in share.items():
            valid = [v for v in row.values() if v is not None]
            assert sum(valid) == pytest.approx(1.0, abs=1e-9), variant

    def test_all_zero_returns_zeros(self):
        result = _make_result_with_pdn({"v": {"a": 0.0, "b": 0.0}})
        share = _compute_share_per_domain(result)
        # Zeros, not NaN.
        assert share["v"] == {"a": 0.0, "b": 0.0}

    def test_none_pdn_yields_none_share(self):
        # An invalid domain's share is None — the "not measured"
        # sentinel — not 0.0 (which would mean "measured zero drift").
        result = _make_result_with_pdn({
            "v": {"valid_a": 1.0, "invalid_b": None, "valid_c": 1.0},
        })
        share = _compute_share_per_domain(result)
        assert share["v"]["invalid_b"] is None
        # Valid entries normalize over themselves only.
        assert share["v"]["valid_a"] == pytest.approx(0.5)
        assert share["v"]["valid_c"] == pytest.approx(0.5)

    def test_all_none_pdn_yields_all_none_share(self):
        # Variant where every domain is invalid.
        result = _make_result_with_pdn({
            "v": {"a": None, "b": None},
        })
        share = _compute_share_per_domain(result)
        assert share["v"] == {"a": None, "b": None}


# ── Overall normalized handles None ──────────────────────────────────


class TestOverallNormalizedFromPDN:
    """overall[v] = sqrt( (1/D_valid) · Σ_valid pdn²); None entries skipped."""

    def test_overall_is_per_domain_rms(self):
        pdn = {"v": {"a": 3.0, "b": 4.0}}  # rms = sqrt((9+16)/2) = sqrt(12.5)
        out = _compute_overall_normalized_from_pdn(pdn)
        assert out["v"] == pytest.approx(math.sqrt(12.5))

    def test_each_domain_weighted_equally(self):
        # Same per-domain pdn distribution → same overall regardless
        # of (theoretical) probe-count differences.
        pdn1 = {"v": {"a": 1.0, "b": 1.0}}
        pdn2 = {"v": {"a": 1.0, "b": 1.0}}
        assert (
            _compute_overall_normalized_from_pdn(pdn1)["v"]
            == _compute_overall_normalized_from_pdn(pdn2)["v"]
        )

    def test_empty_pdn_yields_empty(self):
        assert _compute_overall_normalized_from_pdn({}) == {}

    def test_variant_with_no_domains_yields_zero(self):
        pdn = {"v": {}}
        assert _compute_overall_normalized_from_pdn(pdn) == {"v": 0.0}

    def test_none_entries_excluded_from_overall(self):
        # Mixed: 2 valid + 1 None. RMS computed over the 2 valid only.
        pdn = {"v": {"a": 3.0, "b": None, "c": 4.0}}
        out = _compute_overall_normalized_from_pdn(pdn)
        assert out["v"] == pytest.approx(math.sqrt((9 + 16) / 2))

    def test_all_none_yields_zero(self):
        # Variant with every pdn=None → overall = 0.0 (no valid signal).
        pdn = {"v": {"a": None, "b": None}}
        out = _compute_overall_normalized_from_pdn(pdn)
        assert out["v"] == 0.0


# ── pdn property alias (Q9.5) ────────────────────────────────────────


class TestPdnAlias:
    def test_pdn_returns_same_dict_as_long_name(self):
        result = _make_result_with_pdn({"v": {"a": 1.0, "b": 2.0}})
        assert result.pdn is result.magnitudes_per_domain_normalized
        # Mutating via either reference visible from both.
        result.pdn["v"]["c"] = 3.0
        assert result.magnitudes_per_domain_normalized["v"]["c"] == 3.0
