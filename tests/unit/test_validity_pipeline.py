"""Pipeline-level validity-integration tests — v0.4.1, audit §1.3.

Asserts that ``run_family_pipeline`` correctly:
1. computes per-probe ``EngineValidity`` records for both base and variant
2. propagates them into the resulting ``GeoResult.probe_validity``
3. derives ``GeoResult.domain_status`` per (variant, domain)
4. skips per-probe sub-loop work for invalid probes (NaN δ, dropped by
   the global filter)

CPU-only — uses ``MockEngine`` with explicit ``max_context`` thresholds.
The ``MockEngine.token_count`` is word-split, so we can hand-craft
prompts of known token length to control validity outcomes.
"""
from __future__ import annotations

import math

import pytest

from lmdiff._api import _BASE_ANCHOR, _compute_anchor_map
from lmdiff._config import Config
from lmdiff._pipeline import run_family_pipeline
from lmdiff.probes.loader import Probe, ProbeSet

from tests.fixtures.mock_engine import MockEngine


def _make_probes() -> ProbeSet:
    """5 probes: 4 short + 1 deliberately long. Word-count = approx
    token-count under MockEngine's word-split tokenizer."""
    return ProbeSet([
        # 5-token probes (within base context)
        Probe(id="p0", text="one two three four five", domain="d_short"),
        Probe(id="p1", text="six seven eight nine ten", domain="d_short"),
        Probe(id="p2", text="aa bb cc dd ee", domain="d_short"),
        # 50-token probe — within variant context but BEYOND base
        Probe(id="p3", text=" ".join(f"w{i}" for i in range(50)), domain="d_long"),
        # 200-token probe — beyond BOTH engines' contexts
        Probe(id="p4", text=" ".join(f"w{i}" for i in range(200)), domain="d_long"),
    ])


# Base allows ~10-token probes (5 + max_new_tokens=4 → T_i = 9 ≤ 10 ✓
# for short probes; 50+4 = 54 > 10 ✗ for long).
# Variant allows ~100-token probes (50+4 = 54 ≤ 100 ✓; 200+4 = 204 > 100 ✗).
BASE_MAX = 10
VARIANT_MAX = 100


class TestValidityRecordsInResult:
    def test_per_probe_validity_recorded_for_both_engines(self):
        base_cfg = Config(model="m_base")
        v_cfg = Config(model="m_v")
        base_eng = MockEngine(config=base_cfg, max_context=BASE_MAX, seed=1)
        v_eng = MockEngine(config=v_cfg, max_context=VARIANT_MAX, seed=2)

        anchor_map = _compute_anchor_map(
            [(_BASE_ANCHOR, base_cfg), ("v", v_cfg)],
        )

        result = run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_configs={"v": v_cfg},
            probe_set=_make_probes(),
            variant_engines={"v": v_eng},
            max_new_tokens=4,
            engine_groups=anchor_map,
        )

        # Every probe has a record under both engines.
        assert len(result.probe_validity) == 5
        for probe_id in ("p0", "p1", "p2", "p3", "p4"):
            pv = result.probe_validity[probe_id]
            assert pv.probe_id == probe_id
            assert base_eng.name in pv.per_engine
            assert v_eng.name in pv.per_engine

    def test_short_probes_valid_for_both(self):
        base_cfg = Config(model="m_base")
        v_cfg = Config(model="m_v")
        base_eng = MockEngine(config=base_cfg, max_context=BASE_MAX, seed=1)
        v_eng = MockEngine(config=v_cfg, max_context=VARIANT_MAX, seed=2)

        result = run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_configs={"v": v_cfg},
            probe_set=_make_probes(),
            variant_engines={"v": v_eng},
            max_new_tokens=4,
        )

        for probe_id in ("p0", "p1", "p2"):
            pv = result.probe_validity[probe_id]
            assert pv.valid_for(base_eng.name) is True, probe_id
            assert pv.valid_for(v_eng.name) is True, probe_id

    def test_long_probe_p3_variant_only_valid(self):
        """p3 is 50 tokens — too long for base (max 10), within variant
        (max 100)."""
        base_cfg = Config(model="m_base")
        v_cfg = Config(model="m_v")
        base_eng = MockEngine(config=base_cfg, max_context=BASE_MAX, seed=1)
        v_eng = MockEngine(config=v_cfg, max_context=VARIANT_MAX, seed=2)

        result = run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_configs={"v": v_cfg},
            probe_set=_make_probes(),
            variant_engines={"v": v_eng},
            max_new_tokens=4,
        )

        pv = result.probe_validity["p3"]
        assert pv.valid_for(base_eng.name) is False
        assert pv.valid_for(v_eng.name) is True
        assert pv.per_engine[base_eng.name].reason == "exceeds_context"

    def test_too_long_probe_p4_invalid_for_both(self):
        """p4 is 200 tokens — exceeds both base (10) and variant (100)."""
        base_cfg = Config(model="m_base")
        v_cfg = Config(model="m_v")
        base_eng = MockEngine(config=base_cfg, max_context=BASE_MAX, seed=1)
        v_eng = MockEngine(config=v_cfg, max_context=VARIANT_MAX, seed=2)

        result = run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_configs={"v": v_cfg},
            probe_set=_make_probes(),
            variant_engines={"v": v_eng},
            max_new_tokens=4,
        )

        pv = result.probe_validity["p4"]
        assert pv.valid_for(base_eng.name) is False
        assert pv.valid_for(v_eng.name) is False


class TestDomainStatusFromValidity:
    def test_d_short_is_full(self):
        """d_short has 3 probes all valid for both engines → 'full'."""
        base_cfg = Config(model="m_base")
        v_cfg = Config(model="m_v")
        base_eng = MockEngine(config=base_cfg, max_context=BASE_MAX)
        v_eng = MockEngine(config=v_cfg, max_context=VARIANT_MAX)

        result = run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_configs={"v": v_cfg},
            probe_set=_make_probes(),
            variant_engines={"v": v_eng},
            max_new_tokens=4,
        )

        assert result.domain_status["v"]["d_short"] == "full"

    def test_d_long_is_variant_only(self):
        """d_long has p3 (variant-valid, base-invalid) and p4 (both
        invalid). Base measured nothing in this domain; variant
        covered p3. Per compute_domain_status: n_both=0 + n_var_only>0
        → 'variant_only' (the all-base-invalid criterion takes
        precedence over the partial fallback)."""
        base_cfg = Config(model="m_base")
        v_cfg = Config(model="m_v")
        base_eng = MockEngine(config=base_cfg, max_context=BASE_MAX)
        v_eng = MockEngine(config=v_cfg, max_context=VARIANT_MAX)

        result = run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_configs={"v": v_cfg},
            probe_set=_make_probes(),
            variant_engines={"v": v_eng},
            max_new_tokens=4,
        )

        assert result.domain_status["v"]["d_long"] == "variant_only"


class TestInvalidProbesProduceNaNDelta:
    """When a probe is invalid for the variant, no generate / score work
    happens; the resulting δ is NaN. The global _universally_valid_indices
    filter then drops that probe from change_vectors, so n_probes <
    len(probe_set) when invalid probes are present."""

    def test_invalid_probes_dropped_from_change_vectors(self):
        base_cfg = Config(model="m_base")
        v_cfg = Config(model="m_v")
        base_eng = MockEngine(config=base_cfg, max_context=BASE_MAX, seed=1)
        v_eng = MockEngine(config=v_cfg, max_context=VARIANT_MAX, seed=2)

        result = run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_configs={"v": v_cfg},
            probe_set=_make_probes(),
            variant_engines={"v": v_eng},
            max_new_tokens=4,
        )

        # 5 probes total, 3 valid for both (p0,p1,p2). p3 invalid for
        # base (so base δ undefined), p4 invalid for both. Both end up
        # NaN in raw_deltas → dropped by _universally_valid_indices.
        # Result: change_vectors length = 3.
        assert result.n_probes == 3
        assert len(result.change_vectors["v"]) == 3
        # All retained values are finite.
        for x in result.change_vectors["v"]:
            assert not math.isnan(x)


class TestNoMaxContextMeansAllValid:
    """When max_context_length() returns None (default for MockEngine
    without an explicit kwarg), every probe is treated as valid (no
    validity filter applied)."""

    def test_unknown_limit_keeps_all_probes(self):
        base_cfg = Config(model="m_base")
        v_cfg = Config(model="m_v")
        base_eng = MockEngine(config=base_cfg, seed=1)  # max_context=None
        v_eng = MockEngine(config=v_cfg, seed=2)        # max_context=None

        result = run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_configs={"v": v_cfg},
            probe_set=_make_probes(),
            variant_engines={"v": v_eng},
            max_new_tokens=4,
        )

        # All 5 probes survive — no validity filter triggered.
        assert result.n_probes == 5
        for probe_id in ("p0", "p1", "p2", "p3", "p4"):
            pv = result.probe_validity[probe_id]
            for eng_name, ev in pv.per_engine.items():
                assert ev.is_valid is True
                assert ev.reason == "unknown_limit"
