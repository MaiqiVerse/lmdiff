"""Unit tests for ``lmdiff._validity`` — Q9.1, Q9.6, audit §1, §2.

Covers:
- ``EngineValidity`` construction + reason-tag semantics
- ``ProbeValidity`` predicates (``valid_for``, ``valid_for_all``)
- ``compute_domain_status`` against synthetic mixes covering all four
  states + the documented hybrid tie-breaking case + empty edge case
"""
from __future__ import annotations

import pytest

from lmdiff._validity import (
    EngineValidity,
    ProbeValidity,
    compute_domain_status,
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
        # 80 valid for both, 20 valid for variant only.
        # Per audit §2.1: classified as partial (the 80 still produce
        # signal; the 20 feed v0.5.0+ variant_only_metrics).
        probes = [_probe(f"p{i}", True, True) for i in range(80)]
        probes += [_probe(f"q{i}", False, True) for i in range(20)]
        assert compute_domain_status(probes, "base", "var") == "partial"

    def test_mixed_valid_some_invalid_is_partial(self):
        # 50/50 split with all four combinations present.
        probes = [
            _probe("p1", True, True),
            _probe("p2", True, False),
            _probe("p3", False, True),
            _probe("p4", False, False),
        ]
        assert compute_domain_status(probes, "base", "var") == "partial"

    def test_empty_probes_is_out_of_range(self):
        assert compute_domain_status([], "base", "var") == "out_of_range"

    def test_base_valid_variant_invalid_is_partial(self):
        # All probes valid for base, all invalid for variant. Edge case
        # where the variant has a *smaller* context than base. Not
        # variant_only (variant has zero valid). Not out_of_range (base
        # measures everything). Falls into "partial" — base measures,
        # variant doesn't, so base-vs-variant δ can't be computed.
        probes = [_probe(f"p{i}", True, False) for i in range(5)]
        assert compute_domain_status(probes, "base", "var") == "partial"
