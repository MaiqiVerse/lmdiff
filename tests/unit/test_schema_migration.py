"""Schema migration tests — v0.4.1 v5 → v6 path (Q9.8 preserve).

The v0.4.1 contract (Q9.8): loading a v5 GeoResult JSON
**preserves** the saved ``share_per_domain`` and
``magnitudes_per_domain_normalized`` values exactly. The v0.4.1
formula (Q9.10 Formula A) is NOT applied to legacy saves —
"saved means saved." Validity stubs are synthesized so the v6
schema shape is satisfied, and a single ``DeprecationWarning``
fires advising the user to re-run for v0.4.1 numerics.

Companion to ``test_normalization_formulas.py`` (which covers
the v0.4.1 formula on fresh data) and ``test_validity.py``
(which covers the validity dataclasses standalone).
"""
from __future__ import annotations

import json
import warnings

import pytest

from lmdiff.geometry import GeoResult
from lmdiff.report.json_report import (
    SCHEMA_VERSION,
    geo_result_from_json_dict,
    to_json_dict,
)


def _v5_payload() -> dict:
    """Hand-craft a v5 JSON: schema_version "5", pre-v0.4.1 share /
    pdn values present (using the old √T̄ formula's outputs)."""
    return {
        "schema_version": "5",
        "base_name": "base",
        "variant_names": ["v"],
        "n_probes": 4,
        "magnitudes": {"v": 5.0},
        "cosine_matrix": {"v": {"v": 1.0}},
        "change_vectors": {"v": [1.0, 2.0, 3.0, 4.0]},
        "per_probe": {"v": {"p0": 1.0, "p1": 2.0, "p2": 3.0, "p3": 4.0}},
        "metadata": {"n_total_probes": 4, "n_skipped": 0},
        "delta_means": {"v": 2.5},
        "selective_magnitudes": {"v": 2.236},
        "selective_cosine_matrix": {"v": {"v": 1.0}},
        "probe_domains": ["a", "a", "b", "b"],
        "avg_tokens_per_probe": [10.0, 10.0, 1000.0, 1000.0],
        "magnitudes_normalized": {"v": 0.5},
        # Old √T̄ formula numbers — kept verbatim for comparison.
        "magnitudes_per_domain_normalized": {
            "v": {"a": 0.5, "b": 0.077},
        },
        "share_per_domain": {
            "v": {"a": 0.81, "b": 0.19},
        },
    }


# ── Schema version ──────────────────────────────────────────────────


class TestSchemaVersion:
    def test_writer_emits_v6(self):
        assert SCHEMA_VERSION == "6"


# ── v5 load: preserve, do NOT recompute (Q9.8) ──────────────────────


class TestV5LoadPreservesSavedValues:
    def test_v5_load_preserves_share_per_domain_byte_identical(self):
        v5 = _v5_payload()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            restored = geo_result_from_json_dict(v5)
        # Q9.8: share values come back EXACTLY as saved (no recompute).
        assert restored.share_per_domain["v"]["a"] == pytest.approx(0.81)
        assert restored.share_per_domain["v"]["b"] == pytest.approx(0.19)

    def test_v5_load_preserves_pdn_byte_identical(self):
        v5 = _v5_payload()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            restored = geo_result_from_json_dict(v5)
        # Q9.8: pdn values preserved verbatim from the save.
        assert restored.magnitudes_per_domain_normalized["v"]["a"] == pytest.approx(0.5)
        assert restored.magnitudes_per_domain_normalized["v"]["b"] == pytest.approx(0.077)

    def test_v5_load_preserves_magnitudes_normalized(self):
        v5 = _v5_payload()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            restored = geo_result_from_json_dict(v5)
        assert restored.magnitudes_normalized["v"] == pytest.approx(0.5)

    def test_v5_load_emits_one_deprecation_warning(self):
        v5 = _v5_payload()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            geo_result_from_json_dict(v5)
        deprecations = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecations) == 1
        msg = str(deprecations[0].message)
        # Warning calls out the preservation policy and v0.4.1 formula.
        assert "v5" in msg
        assert "preserved as saved" in msg
        assert "v0.4.1" in msg

    def test_v5_load_synthesizes_empty_probe_validity(self):
        v5 = _v5_payload()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            restored = geo_result_from_json_dict(v5)
        # No per-probe records — v5 saves don't have them.
        assert restored.probe_validity == {}

    def test_v5_load_synthesizes_full_status_for_every_pair(self):
        v5 = _v5_payload()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            restored = geo_result_from_json_dict(v5)
        # Domain status synthesized as "full" for every (variant, domain)
        # pair the saved data has — matches the v5 implicit assumption
        # "all probes valid for all engines."
        assert restored.domain_status == {"v": {"a": "full", "b": "full"}}

    def test_v5_load_variant_only_metrics_is_none(self):
        v5 = _v5_payload()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            restored = geo_result_from_json_dict(v5)
        assert restored.variant_only_metrics is None


# ── v6 round-trip: full schema in/out ───────────────────────────────


class TestV6RoundTrip:
    def test_v6_save_then_load_preserves_share_with_none(self):
        # Build a GeoResult with a None share value; round-trip; verify
        # None survives JSON serialization (encoded as null).
        result = GeoResult(
            base_name="base",
            variant_names=["v"],
            n_probes=2,
            magnitudes={"v": 1.0},
            cosine_matrix={"v": {"v": 1.0}},
            change_vectors={"v": [1.0, 2.0]},
            per_probe={"v": {"p0": 1.0, "p1": 2.0}},
            share_per_domain={"v": {"a": 0.7, "b": None}},
            magnitudes_per_domain_normalized={"v": {"a": 1.5, "b": None}},
            domain_status={"v": {"a": "full", "b": "out_of_range"}},
        )
        payload = to_json_dict(result)
        # Re-encode + decode through a JSON string to flush None → null.
        text = json.dumps(payload)
        restored = geo_result_from_json_dict(json.loads(text))
        assert restored.share_per_domain["v"]["a"] == pytest.approx(0.7)
        assert restored.share_per_domain["v"]["b"] is None
        assert restored.magnitudes_per_domain_normalized["v"]["b"] is None

    def test_v6_save_then_load_preserves_domain_status(self):
        result = GeoResult(
            base_name="base",
            variant_names=["v1", "v2"],
            n_probes=2,
            magnitudes={"v1": 1.0, "v2": 2.0},
            cosine_matrix={"v1": {"v1": 1.0, "v2": 0.5},
                           "v2": {"v1": 0.5, "v2": 1.0}},
            change_vectors={"v1": [1.0, 2.0], "v2": [2.0, 3.0]},
            per_probe={"v1": {"p0": 1.0, "p1": 2.0},
                       "v2": {"p0": 2.0, "p1": 3.0}},
            domain_status={
                "v1": {"a": "full", "b": "partial"},
                "v2": {"a": "variant_only", "b": "out_of_range"},
            },
        )
        payload = to_json_dict(result)
        restored = geo_result_from_json_dict(json.loads(json.dumps(payload)))
        assert restored.domain_status == result.domain_status

    def test_v6_save_then_load_preserves_probe_validity(self):
        from lmdiff._validity import EngineValidity, ProbeValidity

        ev_base = EngineValidity(
            engine_name="base", max_context=4096, T_i=100,
            is_valid=True, reason="valid",
        )
        ev_var = EngineValidity(
            engine_name="var", max_context=128_000, T_i=100,
            is_valid=True, reason="valid",
        )
        pv = ProbeValidity(
            probe_id="p0",
            domain="commonsense",
            per_engine={"base": ev_base, "var": ev_var},
        )
        result = GeoResult(
            base_name="base",
            variant_names=["var"],
            n_probes=1,
            magnitudes={"var": 1.0},
            cosine_matrix={"var": {"var": 1.0}},
            change_vectors={"var": [1.0]},
            per_probe={"var": {"p0": 1.0}},
            probe_validity={"p0": pv},
        )
        payload = to_json_dict(result)
        restored = geo_result_from_json_dict(json.loads(json.dumps(payload)))
        assert "p0" in restored.probe_validity
        rpv = restored.probe_validity["p0"]
        assert rpv.probe_id == "p0"
        assert rpv.domain == "commonsense"
        assert rpv.valid_for("base") is True
        assert rpv.valid_for("var") is True
        assert rpv.per_engine["base"].max_context == 4096
        assert rpv.per_engine["var"].max_context == 128_000
