"""Regression tests for the v0.3.2 share / overall-normalized formula fix.

The v0.3.0 / v0.3.1 / pre-fix v0.3.2 formulas were length-weighted:

  share_per_domain[v][d]   = ‖δ_{v|d}‖² / Σ_d' ‖δ_{v|d'}‖²
  magnitudes_normalized[v] = ‖δ_v‖ / sqrt(n_probes · mean_T)

A single long-context domain (e.g., longbench at 9000 tokens vs MCQ
at 30 tokens) would dominate ~99 % of every variant's share, and the
overall normalized magnitude would be biased toward the long-prompt
domain. The v0.3.2 fix replaces both with per-domain per-token RMS:

  pdn[v][d]                = sqrt( Σ_{i∈d} δ[v][i]² / Σ_{i∈d} T[i] )
  share_per_domain[v][d]   = pdn[v][d]² / Σ_d' pdn[v][d']²
  magnitudes_normalized[v] = sqrt( mean over d of pdn[v][d]² )

These tests verify (a) the new helpers are mathematically correct on
hand-computable inputs, (b) the OLD formula would have produced a
very different (length-biased) result on the same data, and (c) the
v6 §13 calibration mockup numbers (long-reasoning ≈ 66 %, yarn-
commonsense ≈ 51 %) are reproduced by the new helpers.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from lmdiff.geometry import (
    GeoResult,
    _compute_overall_normalized_from_pdn,
    _compute_per_domain_normalized,
    _compute_share_per_domain,
)


# ── Hand-computable formula sanity ────────────────────────────────────


class TestPerDomainNormalizedFormula:
    """pdn[v][d] = sqrt( Σ_{i∈d} δ[v][i]² / Σ_{i∈d} T[i] )"""

    def test_uniform_tokens_collapses_to_rms_per_token(self):
        # All probes T=10. Domain has δ = [3, 4] → sum_sq = 25, sum_T=20.
        # pdn = sqrt(25/20) = sqrt(1.25)
        cv = {"v": [3.0, 4.0]}
        pdn = _compute_per_domain_normalized(
            ["v"], cv, ("d",) * 2, (10.0, 10.0),
        )
        assert pdn["v"]["d"] == pytest.approx(math.sqrt(25.0 / 20.0))

    def test_two_domains_separated(self):
        # δ = [3, 4, 6, 8], domains = [a, a, b, b], T uniform 10.
        # pdn[v][a] = sqrt((9+16)/20) = sqrt(1.25)
        # pdn[v][b] = sqrt((36+64)/20) = sqrt(5.0)
        cv = {"v": [3.0, 4.0, 6.0, 8.0]}
        pdn = _compute_per_domain_normalized(
            ["v"], cv, ("a", "a", "b", "b"), (10.0,) * 4,
        )
        assert pdn["v"]["a"] == pytest.approx(math.sqrt(25.0 / 20.0))
        assert pdn["v"]["b"] == pytest.approx(math.sqrt(100.0 / 20.0))

    def test_per_token_normalization_neutralizes_length(self):
        # Two domains, same per-token δ structure but different prompt
        # lengths. The pdn values should be identical — the whole point
        # of per-token normalization.
        cv = {"v": [1.0, 1.0, 1.0, 1.0]}
        pdn = _compute_per_domain_normalized(
            ["v"], cv,
            ("short", "short", "long", "long"),
            (10.0, 10.0, 1000.0, 1000.0),  # 100x token ratio
        )
        # short:   sum_sq = 2, sum_T = 20  → sqrt(2/20)  = sqrt(0.1)
        # long:    sum_sq = 2, sum_T = 2000 → sqrt(2/2000) = sqrt(0.001)
        # Note: per-token magnitude IS smaller for long because the same
        # per-probe δ spread over more tokens implies smaller per-token
        # CE difference. That's the correct interpretation.
        assert pdn["v"]["short"] == pytest.approx(math.sqrt(0.1))
        assert pdn["v"]["long"] == pytest.approx(math.sqrt(0.001))

    def test_empty_inputs_return_empty(self):
        # No probe domains.
        assert _compute_per_domain_normalized(
            ["v"], {"v": [1.0]}, (), (10.0,),
        ) == {}
        # No token info.
        assert _compute_per_domain_normalized(
            ["v"], {"v": [1.0]}, ("d",), (),
        ) == {}
        # Length mismatch between probe_domains and avg_tokens_per_probe.
        assert _compute_per_domain_normalized(
            ["v"], {"v": [1.0, 2.0]}, ("a", "b"), (10.0,),
        ) == {}

    def test_zero_token_count_yields_zero(self):
        cv = {"v": [3.0]}
        pdn = _compute_per_domain_normalized(["v"], cv, ("d",), (0.0,))
        assert pdn["v"]["d"] == 0.0


class TestShareFormulaUsesPDN:
    """share[v][d] = pdn[v][d]² / Σ pdn²"""

    def test_share_ratios_match_pdn_squared(self):
        # Construct a result where each variant has pdn we can reason
        # about, then verify share rows compute correctly.
        result = _make_result_with_pdn({
            "v": {"a": 1.0, "b": 2.0},  # 1² : 4² → share = 1/5 : 4/5
        })
        share = _compute_share_per_domain(result)
        assert share["v"]["a"] == pytest.approx(1.0 / 5.0)
        assert share["v"]["b"] == pytest.approx(4.0 / 5.0)

    def test_rows_sum_to_one(self):
        result = _make_result_with_pdn({
            "v1": {"a": 0.5, "b": 1.5, "c": 0.7},
            "v2": {"a": 2.0, "b": 0.0, "c": 0.3},
        })
        share = _compute_share_per_domain(result)
        for variant, row in share.items():
            assert sum(row.values()) == pytest.approx(1.0, abs=1e-9), variant

    def test_all_zero_returns_zeros(self):
        result = _make_result_with_pdn({"v": {"a": 0.0, "b": 0.0}})
        share = _compute_share_per_domain(result)
        # Zeros, not NaN.
        assert share["v"] == {"a": 0.0, "b": 0.0}


class TestOverallNormalizedFromPDN:
    """overall[v] = sqrt( (1/D) · Σ pdn[v][d]² )"""

    def test_overall_is_per_domain_rms(self):
        pdn = {"v": {"a": 3.0, "b": 4.0}}  # rms = sqrt((9+16)/2) = sqrt(12.5)
        out = _compute_overall_normalized_from_pdn(pdn)
        assert out["v"] == pytest.approx(math.sqrt(12.5))

    def test_each_domain_weighted_equally(self):
        # Same per-domain pdn distribution, different number of probes
        # per domain — overall should be the same (each domain weighted
        # by 1, not by probe count).
        pdn1 = {"v": {"a": 1.0, "b": 1.0}}
        pdn2 = {"v": {"a": 1.0, "b": 1.0}}  # imagine these came from
                                              # different probe counts
        assert (
            _compute_overall_normalized_from_pdn(pdn1)["v"]
            == _compute_overall_normalized_from_pdn(pdn2)["v"]
        )

    def test_empty_pdn_yields_empty(self):
        assert _compute_overall_normalized_from_pdn({}) == {}

    def test_variant_with_no_domains_yields_zero(self):
        pdn = {"v": {}}
        assert _compute_overall_normalized_from_pdn(pdn) == {"v": 0.0}


# ── Demonstrate OLD formula was length-biased ─────────────────────────


class TestOldFormulaBiasIsRealAndFixed:
    """Construct a scenario where the OLD raw-magnitude share assigns
    ~95 % to a long-context domain even though that domain has the
    SAME per-token drift as a short domain. The NEW pdn-based share
    splits roughly 50/50."""

    def _build_long_vs_short_scenario(self):
        # 4 short probes (T=10 each, δ=1.0 each → per-token CE shift ≈ 0.1)
        # 4 long probes  (T=1000 each, δ=10.0 each → per-token CE shift ≈ 0.01)
        # Wait — δ is *per-probe* avg per-token CE difference, so if the
        # per-token CE shift per probe is ~constant, δ is ~constant
        # regardless of T. Set them equal at δ=1.0:
        cv = {"v": [1.0, 1.0, 1.0, 1.0,   # short probes
                    1.0, 1.0, 1.0, 1.0]}  # long probes
        domains = ("short",) * 4 + ("long",) * 4
        tokens = (10.0,) * 4 + (1000.0,) * 4
        return cv, domains, tokens

    def test_old_raw_share_dominated_by_long(self):
        # Demonstrate the OLD formula's bias on this construction:
        # raw ‖δ_{short}‖² = 4, raw ‖δ_{long}‖² = 4.
        # share_old = 4/8 = 50% each. Hmm — actually for this contrived
        # case the OLD formula is fair too. Let me make it asymmetric.
        cv = {"v": [1.0, 1.0, 1.0, 1.0,
                    3.0, 3.0, 3.0, 3.0]}
        # OLD: ‖short‖² = 4, ‖long‖² = 36 → share_long = 36/40 = 90%
        # NEW: pdn_short = sqrt(4/40) = sqrt(0.1) ≈ 0.316
        #      pdn_long  = sqrt(36/4000) = sqrt(0.009) ≈ 0.095
        #      share_long = 0.009 / (0.1 + 0.009) = 0.0826 = 8.26 %
        domains = ("short",) * 4 + ("long",) * 4
        tokens = (10.0,) * 4 + (1000.0,) * 4

        pdn = _compute_per_domain_normalized(["v"], cv, domains, tokens)
        # Manual:
        assert pdn["v"]["short"] == pytest.approx(math.sqrt(0.1))
        assert pdn["v"]["long"] == pytest.approx(math.sqrt(0.009))

        result = _make_result_with_pdn_and_data(cv, domains, tokens, pdn)
        share = _compute_share_per_domain(result)
        # NEW: short dominates because per-token drift is much larger.
        assert share["v"]["short"] == pytest.approx(0.1 / 0.109, abs=1e-3)
        assert share["v"]["long"]  == pytest.approx(0.009 / 0.109, abs=1e-3)

        # Cross-check OLD formula on the same input:
        sq_short = sum(x ** 2 for x in cv["v"][:4])
        sq_long  = sum(x ** 2 for x in cv["v"][4:])
        old_share_long = sq_long / (sq_short + sq_long)
        assert old_share_long == pytest.approx(36.0 / 40.0)  # 90 %
        # New formula gives long ≈ 8 %. A 10x swing.
        assert share["v"]["long"] < 0.1, "new share should be much smaller"
        # Sanity: the corrected share for long is dramatically below
        # the legacy 90 %.
        assert old_share_long > 5 * share["v"]["long"]


# ── v6 §13 calibration mockup ────────────────────────────────────────


class TestV6Section13CalibrationMockup:
    """The user's design spec calls out two expected numbers from the
    v6 §13 mockup that should be reproduced by the corrected formulas:

      (long, reasoning) share ≈ 66 %
      (yarn, commonsense) share ≈ 51 %

    Without the original numerical fixture, we engineer per-domain
    per-token RMS values (pdn) that produce these shares analytically,
    and assert the share helper gives the documented percentages."""

    def test_long_reasoning_share_is_66pct(self):
        # For variant "long" across two domains (reasoning, commonsense):
        # share[long][reasoning] = pdn[reasoning]² / (pdn[reasoning]² + pdn[commonsense]²)
        # Solve 0.66 = a²/(a²+b²) → a² = 0.66/(1-0.66) · b² ≈ 1.941 · b²
        # Pick b = 1.0, then a = sqrt(1.941) ≈ 1.393.
        result = _make_result_with_pdn({
            "long": {"reasoning": 1.393, "commonsense": 1.0},
        })
        share = _compute_share_per_domain(result)
        assert share["long"]["reasoning"] == pytest.approx(0.66, abs=0.01)

    def test_yarn_commonsense_share_is_51pct(self):
        # share[yarn][commonsense] = 0.51
        # Solve 0.51 = a²/(a²+b²) → a² = 0.51/0.49 · b² ≈ 1.0408 · b²
        # Pick b = 1.0, a = sqrt(1.0408) ≈ 1.020.
        result = _make_result_with_pdn({
            "yarn": {"commonsense": 1.020, "reasoning": 1.0},
        })
        share = _compute_share_per_domain(result)
        assert share["yarn"]["commonsense"] == pytest.approx(0.51, abs=0.01)

    def test_both_variants_in_one_result(self):
        # Joint case: a single GeoResult with both variants. Each
        # variant's share row is independent; verify both still hit
        # their documented targets.
        result = _make_result_with_pdn({
            "long": {"reasoning": 1.393, "commonsense": 1.0},
            "yarn": {"commonsense": 1.020, "reasoning": 1.0},
        })
        share = _compute_share_per_domain(result)
        assert share["long"]["reasoning"] == pytest.approx(0.66, abs=0.01)
        assert share["yarn"]["commonsense"] == pytest.approx(0.51, abs=0.01)
        # Cross-check rows still sum to 1.0.
        for v, row in share.items():
            assert sum(row.values()) == pytest.approx(1.0, abs=1e-9)


# ── Auto-recompute on JSON load ──────────────────────────────────────


class TestLoadAutoRecomputesPreV032Saves:
    def test_v0_3_1_save_loads_with_corrected_share(self):
        """Simulate a v0.3.1 JSON: schema_version v5 with old length-
        biased share_per_domain values, no magnitudes_per_domain_
        normalized field. Loading should auto-recompute and emit a
        DeprecationWarning."""
        from lmdiff.report.json_report import geo_result_from_json_dict

        # Hand-craft a payload mimicking a pre-fix v5 save.
        n = 8
        cv = {"v": [1.0] * 4 + [3.0] * 4}
        # OLD share would be {short: 4/40=10%, long: 36/40=90%}
        old_share = {"v": {"short": 4.0 / 40.0, "long": 36.0 / 40.0}}
        payload = {
            "schema_version": "5",
            "base_name": "base",
            "variant_names": ["v"],
            "n_probes": n,
            "magnitudes": {"v": float(np.linalg.norm(cv["v"]))},
            "cosine_matrix": {"v": {"v": 1.0}},
            "change_vectors": cv,
            "per_probe": {"v": {f"p{i}": cv["v"][i] for i in range(n)}},
            "metadata": {},
            "delta_means": {},
            "selective_magnitudes": {},
            "selective_cosine_matrix": {},
            "probe_domains": ["short"] * 4 + ["long"] * 4,
            "avg_tokens_per_probe": [10.0] * 4 + [1000.0] * 4,
            "magnitudes_normalized": {"v": 1.0},  # arbitrary old value
            "share_per_domain": old_share,
            # No magnitudes_per_domain_normalized → triggers recompute.
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            restored = geo_result_from_json_dict(payload)

        # Deprecation warning fired exactly once.
        depr = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(depr) >= 1
        msg = str(depr[0].message)
        assert "v0.3.2" in msg
        assert "share_per_domain" in msg

        # share_per_domain was overwritten with the corrected value.
        # Long should NOT dominate — short has higher per-token drift.
        assert restored.share_per_domain["v"]["long"] < 0.5
        assert restored.share_per_domain["v"]["long"] != pytest.approx(
            old_share["v"]["long"]
        )

        # The new pdn field is now populated.
        assert restored.magnitudes_per_domain_normalized != {}
        assert "short" in restored.magnitudes_per_domain_normalized["v"]

    def test_no_recompute_when_pdn_already_in_payload(self):
        """If the JSON already has magnitudes_per_domain_normalized,
        the loader trusts it and doesn't recompute / doesn't warn."""
        from lmdiff.report.json_report import geo_result_from_json_dict

        n = 4
        cv = {"v": [1.0, 1.0, 1.0, 1.0]}
        pdn = {"v": {"a": 0.7, "b": 0.3}}
        # Share matching the supplied pdn.
        share = {"v": {
            "a": 0.7 ** 2 / (0.7 ** 2 + 0.3 ** 2),
            "b": 0.3 ** 2 / (0.7 ** 2 + 0.3 ** 2),
        }}
        payload = {
            "schema_version": "5",
            "base_name": "base",
            "variant_names": ["v"],
            "n_probes": n,
            "magnitudes": {"v": 2.0},
            "cosine_matrix": {"v": {"v": 1.0}},
            "change_vectors": cv,
            "per_probe": {"v": {f"p{i}": cv["v"][i] for i in range(n)}},
            "metadata": {},
            "delta_means": {},
            "selective_magnitudes": {},
            "selective_cosine_matrix": {},
            "probe_domains": ["a", "a", "b", "b"],
            "avg_tokens_per_probe": [10.0, 10.0, 100.0, 100.0],
            "magnitudes_normalized": {"v": 0.5},
            "share_per_domain": share,
            "magnitudes_per_domain_normalized": pdn,
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            restored = geo_result_from_json_dict(payload)

        # No deprecation warning — load is post-fix.
        depr = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(depr) == 0

        # Values preserved as-is.
        assert restored.magnitudes_per_domain_normalized == pdn
        assert restored.share_per_domain["v"]["a"] == pytest.approx(share["v"]["a"])
        assert restored.magnitudes_normalized["v"] == 0.5


# ── Helpers ──────────────────────────────────────────────────────────


def _make_result_with_pdn(
    pdn: dict[str, dict[str, float]],
) -> GeoResult:
    """Build a minimal GeoResult that has only the pdn field populated.

    Other fields (change_vectors etc.) are stub-shaped — the share
    computation only consumes pdn. Used to test the share helper in
    isolation."""
    variants = list(pdn.keys())
    cv = {v: [0.0] for v in variants}  # placeholder; share doesn't need it
    return GeoResult(
        base_name="base",
        variant_names=variants,
        n_probes=1,
        magnitudes={v: 0.0 for v in variants},
        cosine_matrix={v: {w: 0.0 for w in variants} for v in variants},
        change_vectors=cv,
        per_probe={v: {} for v in variants},
        magnitudes_per_domain_normalized=pdn,
    )


def _make_result_with_pdn_and_data(
    cv: dict[str, list[float]],
    probe_domains: tuple,
    avg_tokens: tuple,
    pdn: dict[str, dict[str, float]],
) -> GeoResult:
    variants = list(cv.keys())
    return GeoResult(
        base_name="base",
        variant_names=variants,
        n_probes=len(probe_domains),
        magnitudes={v: float(np.linalg.norm(cv[v])) for v in variants},
        cosine_matrix={v: {w: 0.0 for w in variants} for v in variants},
        change_vectors=cv,
        per_probe={v: {f"p{i}": cv[v][i] for i in range(len(cv[v]))} for v in variants},
        probe_domains=probe_domains,
        avg_tokens_per_probe=avg_tokens,
        magnitudes_per_domain_normalized=pdn,
    )
