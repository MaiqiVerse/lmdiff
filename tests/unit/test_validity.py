"""Unit tests for ``lmdiff._validity`` — Q9.1, Q9.6, audit §1, §2.

Covers:
- ``EngineValidity`` construction + reason-tag semantics
- ``ProbeValidity`` predicates (``valid_for``, ``valid_for_all``)
- ``compute_domain_status`` against synthetic mixes covering all four
  states + the documented hybrid tie-breaking case + empty edge case
- the ``min_valid_fraction`` floor (v0.4.1): each state it can produce,
  inclusivity at the boundary, and the ``0.0`` / ``1.0`` degenerate
  floors
"""
from __future__ import annotations

import pytest

from lmdiff._validity import (
    EngineValidity,
    ProbeValidity,
    compute_domain_status,
    filter_measured_cells,
    is_measured,
)


# ── EngineValidity ──────────────────────────────────────────────────


class TestEngineValidity:
    def test_within_context_is_valid(self):
        ev = EngineValidity(
            engine_name="base",
            max_context=4096,
            T_i=2048,
            is_valid=True,
            reason="valid",
        )
        assert ev.is_valid is True
        assert ev.reason == "valid"

    def test_exceeds_context(self):
        ev = EngineValidity(
            engine_name="base",
            max_context=4096,
            T_i=9000,
            is_valid=False,
            reason="exceeds_context",
        )
        assert ev.is_valid is False
        assert ev.reason == "exceeds_context"

    def test_unknown_limit_is_treated_as_valid(self):
        # max_context=None means we don't know the engine's limit, so
        # the caller treats every probe as valid (no filter applied).
        ev = EngineValidity(
            engine_name="custom",
            max_context=None,
            T_i=99999,
            is_valid=True,
            reason="unknown_limit",
        )
        assert ev.is_valid is True
        assert ev.reason == "unknown_limit"

    def test_frozen_raises_on_mutation(self):
        ev = EngineValidity(
            engine_name="base",
            max_context=4096,
            T_i=100,
            is_valid=True,
            reason="valid",
        )
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            ev.is_valid = False  # type: ignore[misc]


# ── ProbeValidity predicates ─────────────────────────────────────────


def _ev(name: str, is_valid: bool) -> EngineValidity:
    return EngineValidity(
        engine_name=name,
        max_context=4096,
        T_i=100 if is_valid else 9000,
        is_valid=is_valid,
        reason="valid" if is_valid else "exceeds_context",
    )


class TestProbeValidity:
    def test_valid_for_known_engine(self):
        pv = ProbeValidity(
            probe_id="p1",
            domain="commonsense",
            per_engine={"base": _ev("base", True), "yarn": _ev("yarn", True)},
        )
        assert pv.valid_for("base") is True
        assert pv.valid_for("yarn") is True

    def test_valid_for_returns_false_for_invalid(self):
        pv = ProbeValidity(
            probe_id="p1",
            domain="long-context",
            per_engine={
                "base": _ev("base", False),
                "yarn": _ev("yarn", True),
            },
        )
        assert pv.valid_for("base") is False
        assert pv.valid_for("yarn") is True

    def test_valid_for_unknown_engine_returns_false(self):
        # Asking about an engine that didn't participate is treated as
        # invalid — surfaces caller-side bugs rather than silently
        # claiming validity.
        pv = ProbeValidity(
            probe_id="p1",
            domain="x",
            per_engine={"base": _ev("base", True)},
        )
        assert pv.valid_for("nonexistent") is False

    def test_valid_for_all_true_when_all_valid(self):
        pv = ProbeValidity(
            probe_id="p1",
            domain="x",
            per_engine={"base": _ev("base", True), "yarn": _ev("yarn", True)},
        )
        assert pv.valid_for_all is True

    def test_valid_for_all_false_when_any_invalid(self):
        pv = ProbeValidity(
            probe_id="p1",
            domain="long-context",
            per_engine={
                "base": _ev("base", False),
                "yarn": _ev("yarn", True),
            },
        )
        assert pv.valid_for_all is False


# ── compute_domain_status ────────────────────────────────────────────


def _probe(probe_id: str, base_valid: bool, var_valid: bool) -> ProbeValidity:
    return ProbeValidity(
        probe_id=probe_id,
        domain="d",
        per_engine={
            "base": _ev("base", base_valid),
            "var": _ev("var", var_valid),
        },
    )


class TestComputeDomainStatus:
    def test_all_valid_is_full(self):
        probes = [_probe(f"p{i}", True, True) for i in range(5)]
        assert compute_domain_status(probes, "base", "var") == "full"

    def test_all_invalid_is_out_of_range(self):
        probes = [_probe(f"p{i}", False, False) for i in range(5)]
        assert compute_domain_status(probes, "base", "var") == "out_of_range"

    def test_base_invalid_variant_valid_is_variant_only(self):
        # All probes invalid for base, all valid for variant.
        # Classic Yarn-vs-Llama2-base on long-context.
        probes = [_probe(f"p{i}", False, True) for i in range(5)]
        assert compute_domain_status(probes, "base", "var") == "variant_only"

    def test_hybrid_80_20_is_partial(self):
        # 80 valid for both, 20 valid for variant only. The
        # valid-for-both fraction (0.80) clears the default floor, so
        # the audit §2.1 tie-break applies: partial wins (the 80 still
        # produce signal; the 20 feed v0.5.0+ variant_only_metrics).
        probes = [_probe(f"p{i}", True, True) for i in range(80)]
        probes += [_probe(f"q{i}", False, True) for i in range(20)]
        assert compute_domain_status(probes, "base", "var") == "partial"

    def test_mixed_quarter_valid_is_out_of_range_under_default_floor(self):
        # 25/25/25/25 split across all four combinations. Only 1 of 4
        # probes is valid for both — below the 0.5 default floor — and
        # the variant-only fraction (0.25) is too small to redeem it.
        probes = [
            _probe("p1", True, True),
            _probe("p2", True, False),
            _probe("p3", False, True),
            _probe("p4", False, False),
        ]
        assert compute_domain_status(probes, "base", "var") == "out_of_range"
        # Pre-v0.4.1 semantics (no floor) called this partial.
        assert compute_domain_status(
            probes, "base", "var", min_valid_fraction=0.0,
        ) == "partial"

    def test_empty_probes_is_out_of_range(self):
        assert compute_domain_status([], "base", "var") == "out_of_range"

    def test_base_valid_variant_invalid(self):
        # All probes valid for base, all invalid for variant. Edge case
        # where the variant has a *smaller* context than base. Zero
        # valid-for-both, zero variant-only — nothing measurable, so the
        # floor sends it to out_of_range.
        probes = [_probe(f"p{i}", True, False) for i in range(5)]
        assert compute_domain_status(probes, "base", "var") == "out_of_range"
        # Without the floor this fell through to partial, even though no
        # base-vs-variant δ can be computed for any probe.
        assert compute_domain_status(
            probes, "base", "var", min_valid_fraction=0.0,
        ) == "partial"


# ── min_valid_fraction floor ─────────────────────────────────────────


class TestMinValidFractionFloor:
    """The floor added in v0.4.1 after the Llama-2 calibration showed a
    27.6% long-context share resting on 9 of 100 probes.

    Fixture shape throughout: 100 probes, 9 valid for both (the real
    calibration number), remainder split between variant-only and
    neither to drive the two sub-branches.
    """

    @staticmethod
    def _mix(n_both: int, n_var_only: int, n_neither: int, n_base_only: int = 0):
        probes = [_probe(f"b{i}", True, True) for i in range(n_both)]
        probes += [_probe(f"v{i}", False, True) for i in range(n_var_only)]
        probes += [_probe(f"n{i}", False, False) for i in range(n_neither)]
        probes += [_probe(f"o{i}", True, False) for i in range(n_base_only)]
        return probes

    # ── the four states under the default floor ──

    def test_full_unaffected_by_floor(self):
        # 100% valid clears any floor, including 1.0.
        probes = self._mix(n_both=100, n_var_only=0, n_neither=0)
        assert compute_domain_status(probes, "base", "var") == "full"
        assert compute_domain_status(
            probes, "base", "var", min_valid_fraction=1.0,
        ) == "full"

    def test_above_floor_stays_partial(self):
        probes = self._mix(n_both=60, n_var_only=20, n_neither=20)
        assert compute_domain_status(probes, "base", "var") == "partial"

    def test_below_floor_with_variant_coverage_is_variant_only(self):
        # The calibration's yarn / long / code shape: 9 valid for both,
        # 91 measurable by the variant alone.
        probes = self._mix(n_both=9, n_var_only=91, n_neither=0)
        assert compute_domain_status(probes, "base", "var") == "variant_only"

    def test_below_floor_without_variant_coverage_is_out_of_range(self):
        # The calibration's math / chat shape: 9 valid for both, the
        # other 91 measurable by nobody.
        probes = self._mix(n_both=9, n_var_only=0, n_neither=91)
        assert compute_domain_status(probes, "base", "var") == "out_of_range"

    def test_below_floor_with_insufficient_variant_coverage(self):
        # Variant covers more than base but still under the floor —
        # not enough to justify a variant_only sub-table entry.
        probes = self._mix(n_both=9, n_var_only=40, n_neither=51)
        assert compute_domain_status(probes, "base", "var") == "out_of_range"

    # ── boundary: exactly at the floor ──

    def test_exactly_at_floor_is_inclusive_for_partial(self):
        # frac_both == 0.5 exactly → clears the floor → partial.
        probes = self._mix(n_both=50, n_var_only=0, n_neither=50)
        assert compute_domain_status(probes, "base", "var") == "partial"

    def test_one_below_floor_flips_to_out_of_range(self):
        # frac_both == 0.49 → below → out_of_range. Pins the boundary
        # against an off-by-one in the comparison direction.
        probes = self._mix(n_both=49, n_var_only=0, n_neither=51)
        assert compute_domain_status(probes, "base", "var") == "out_of_range"

    def test_exactly_at_floor_is_inclusive_for_variant_only(self):
        # frac_both == 0.1 (below), frac_var_only == 0.5 exactly →
        # clears → variant_only.
        probes = self._mix(n_both=10, n_var_only=50, n_neither=40)
        assert compute_domain_status(probes, "base", "var") == "variant_only"

    def test_variant_only_one_below_floor(self):
        probes = self._mix(n_both=10, n_var_only=49, n_neither=41)
        assert compute_domain_status(probes, "base", "var") == "out_of_range"

    # ── degenerate floors ──

    def test_floor_zero_reproduces_pre_v041_semantics(self):
        # With the floor disabled, a single valid-for-both probe out of
        # 100 sustains partial — the behaviour the floor exists to stop.
        probes = self._mix(n_both=1, n_var_only=0, n_neither=99)
        assert compute_domain_status(
            probes, "base", "var", min_valid_fraction=0.0,
        ) == "partial"
        assert compute_domain_status(probes, "base", "var") == "out_of_range"

    def test_floor_zero_still_reaches_variant_only(self):
        # n_both == 0 with variant coverage → variant_only, via the
        # branch that only the disabled floor can reach.
        probes = self._mix(n_both=0, n_var_only=100, n_neither=0)
        assert compute_domain_status(
            probes, "base", "var", min_valid_fraction=0.0,
        ) == "variant_only"

    def test_floor_one_demands_total_coverage(self):
        # min_valid_fraction=1.0: anything short of full drops out.
        probes = self._mix(n_both=99, n_var_only=1, n_neither=0)
        assert compute_domain_status(
            probes, "base", "var", min_valid_fraction=1.0,
        ) == "out_of_range"

    # ── validation ──

    @pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0, -1.0])
    def test_out_of_range_floor_raises(self, bad):
        probes = self._mix(n_both=50, n_var_only=0, n_neither=50)
        with pytest.raises(ValueError, match="min_valid_fraction"):
            compute_domain_status(probes, "base", "var", min_valid_fraction=bad)

    def test_empty_domain_short_circuits_before_validation(self):
        # Documented edge: empty domain is out_of_range. Validation of
        # the floor happens first, so a bad floor still raises.
        assert compute_domain_status([], "base", "var") == "out_of_range"
        with pytest.raises(ValueError, match="min_valid_fraction"):
            compute_domain_status([], "base", "var", min_valid_fraction=5.0)


# ── Shared consumer-side filter (v0.4.2) ─────────────────────────────


_STATUS = {
    "yarn": {"code": "full", "long-context": "variant_only", "math": "partial"},
    "math": {"code": "full", "long-context": "out_of_range", "math": "full"},
}


class TestIsMeasured:
    def test_full_and_partial_are_measured(self):
        assert is_measured(_STATUS, "yarn", "code") is True
        assert is_measured(_STATUS, "yarn", "math") is True

    def test_variant_only_and_out_of_range_are_not(self):
        assert is_measured(_STATUS, "yarn", "long-context") is False
        assert is_measured(_STATUS, "math", "long-context") is False

    def test_absent_status_map_means_everything_measured(self):
        """Pre-v0.4.1 results (v0.2.x, v1-v5 loads) carry no status;
        every cell in them is a real measurement."""
        for empty in (None, {}):
            assert is_measured(empty, "yarn", "long-context") is True

    def test_unknown_key_defaults_to_measured(self):
        """A consumer whose domain list outruns the status map degrades
        to pre-v0.4.1 behaviour rather than silently blanking cells."""
        assert is_measured(_STATUS, "yarn", "unlisted-domain") is True
        assert is_measured(_STATUS, "unlisted-variant", "code") is True


class TestFilterMeasuredCells:
    def test_single_row_form(self):
        row = {"code": 1.0, "long-context": 2.0, "math": 3.0}
        assert filter_measured_cells(_STATUS, row, "yarn") == {
            "code": 1.0, "math": 3.0,
        }

    def test_nested_form_filters_each_row_against_its_own_status(self):
        nested = {
            "yarn": {"code": 1.0, "long-context": 2.0},
            "math": {"code": 3.0, "long-context": 4.0},
        }
        assert filter_measured_cells(_STATUS, nested) == {
            "yarn": {"code": 1.0},
            "math": {"code": 3.0},
        }

    def test_cells_are_removed_not_nulled(self):
        """Removal rather than None is the point: consumers that
        aggregate (mean / std / max / set membership) then exclude them
        by construction instead of each needing its own skip. Nulling
        is what made the specialization z-score wrong rather than
        merely mis-displayed."""
        row = {"code": 1.0, "long-context": 2.0}
        out = filter_measured_cells(_STATUS, row, "yarn")
        assert "long-context" not in out
        assert None not in out.values()

    def test_empty_status_is_identity(self):
        row = {"code": 1.0, "long-context": 2.0}
        for empty in (None, {}):
            assert filter_measured_cells(empty, row, "yarn") == row

    def test_does_not_mutate_input(self):
        row = {"code": 1.0, "long-context": 2.0}
        filter_measured_cells(_STATUS, row, "yarn")
        assert row == {"code": 1.0, "long-context": 2.0}

    def test_all_cells_excluded_yields_empty_row(self):
        row = {"long-context": 2.0}
        assert filter_measured_cells(_STATUS, row, "math") == {}
