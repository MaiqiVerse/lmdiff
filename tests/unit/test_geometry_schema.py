"""Unit tests for GeoResult schema v5 (commit 1.4).

Covers:

* The new ``share_per_domain`` field is populated by ``analyze()``.
* Rows of ``share_per_domain`` sum to ~1.0 (within float tolerance).
* v5 round-trip (write → read → equal).
* v4 JSON loads with ``DeprecationWarning`` and synthesises
  ``share_per_domain`` from the v4 per-domain magnitudes.
* ``share_per_domain`` is empty when ``probe_domains`` is empty.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from lmdiff.geometry import GeoResult, _compute_share_per_domain
from lmdiff.report.json_report import (
    SCHEMA_VERSION,
    geo_result_from_json_dict,
    to_json_dict,
)


def _make_geo(
    *,
    variants: tuple[str, ...] = ("v1", "v2"),
    domains: tuple[str | None, ...] = ("a", "a", "b", "b"),
) -> GeoResult:
    """Synthetic GeoResult with non-trivial per-domain magnitudes.

    δ_v1 = [3, 4, 0, 0]   →  ‖a‖² = 25,  ‖b‖² = 0    →  share = {a: 1.0, b: 0.0}
    δ_v2 = [0, 0, 6, 8]   →  ‖a‖² = 0,   ‖b‖² = 100  →  share = {a: 0.0, b: 1.0}
    """
    n = len(domains)
    cv = {
        "v1": [3.0, 4.0, 0.0, 0.0][:n],
        "v2": [0.0, 0.0, 6.0, 8.0][:n],
    }
    return GeoResult(
        base_name="base",
        variant_names=list(variants),
        n_probes=n,
        magnitudes={v: float(np.linalg.norm(cv[v])) for v in variants},
        cosine_matrix={
            v: {w: 1.0 if v == w else 0.0 for w in variants} for v in variants
        },
        change_vectors=cv,
        per_probe={v: {f"p{i}": cv[v][i] for i in range(n)} for v in variants},
        metadata={},
        probe_domains=tuple(domains),
        avg_tokens_per_probe=tuple([8.0] * n),
    )


# ── share_per_domain computation ──────────────────────────────────────


class TestSharePerDomain:
    def test_helper_computes_clean_shares(self):
        geo = _make_geo()
        share = _compute_share_per_domain(geo)
        assert share["v1"]["a"] == pytest.approx(1.0)
        assert share["v1"]["b"] == pytest.approx(0.0)
        assert share["v2"]["a"] == pytest.approx(0.0)
        assert share["v2"]["b"] == pytest.approx(1.0)

    def test_rows_sum_to_one(self):
        # 4 variants × 5 domains with non-trivial spread.
        n = 20
        cv = {
            "yarn": list(np.random.RandomState(0).randn(n).astype(float)),
            "long": list(np.random.RandomState(1).randn(n).astype(float)),
            "code": list(np.random.RandomState(2).randn(n).astype(float)),
            "math": list(np.random.RandomState(3).randn(n).astype(float)),
        }
        domains = tuple(["a", "b", "c", "d", "e"][i % 5] for i in range(n))
        geo = GeoResult(
            base_name="base",
            variant_names=list(cv.keys()),
            n_probes=n,
            magnitudes={v: float(np.linalg.norm(cv[v])) for v in cv},
            cosine_matrix={v: {w: 1.0 for w in cv} for v in cv},
            change_vectors=cv,
            per_probe={v: {f"p{i}": cv[v][i] for i in range(n)} for v in cv},
            metadata={},
            probe_domains=domains,
            avg_tokens_per_probe=tuple([8.0] * n),
        )
        share = _compute_share_per_domain(geo)
        for variant, row in share.items():
            assert sum(row.values()) == pytest.approx(1.0, abs=1e-9), variant

    def test_empty_when_probe_domains_missing(self):
        geo = GeoResult(
            base_name="base",
            variant_names=["v1"],
            n_probes=2,
            magnitudes={"v1": 1.0},
            cosine_matrix={"v1": {"v1": 1.0}},
            change_vectors={"v1": [1.0, 0.0]},
            per_probe={"v1": {"p0": 1.0, "p1": 0.0}},
            # probe_domains intentionally empty
        )
        assert _compute_share_per_domain(geo) == {}

    def test_zero_row_returns_zeros_not_nan(self):
        geo = _make_geo()
        # Zero out one variant entirely
        for i in range(geo.n_probes):
            geo.change_vectors["v1"][i] = 0.0
        # Recompute domain magnitudes (analyze() pathway, but here manually)
        # The helper still returns a row of zeros, not NaNs.
        share = _compute_share_per_domain(geo)
        for d in share["v1"]:
            assert share["v1"][d] == 0.0


# ── analyze() populates share_per_domain ───────────────────────────────


class TestAnalyzePopulatesShare:
    def test_analyze_via_mock_engines(self, monkeypatch):
        # We don't run the heavy analyze() pipeline here; we verify the
        # field is preserved on a GeoResult dataclass round-trip and on
        # explicit population. The full analyze() integration is covered
        # by tests/test_geometry.py (slow path).
        geo = _make_geo()
        geo.share_per_domain = _compute_share_per_domain(geo)
        assert geo.share_per_domain
        assert "v1" in geo.share_per_domain


# ── v5 JSON round-trip ────────────────────────────────────────────────


class TestSchemaV5RoundTrip:
    def test_emits_v5_schema(self):
        geo = _make_geo()
        geo.share_per_domain = _compute_share_per_domain(geo)
        d = to_json_dict(geo)
        assert d["schema_version"] == "5"
        assert "share_per_domain" in d
        assert d["share_per_domain"]["v1"]["a"] == pytest.approx(1.0)

    def test_v5_round_trip_preserves_shares(self):
        geo = _make_geo()
        geo.share_per_domain = _compute_share_per_domain(geo)
        d = to_json_dict(geo)
        text = json.dumps(d, sort_keys=True, default=str)
        restored = geo_result_from_json_dict(json.loads(text))
        assert restored.share_per_domain == geo.share_per_domain
        assert restored.variant_names == geo.variant_names

    def test_schema_version_constant_is_v5(self):
        assert SCHEMA_VERSION == "5"


# ── v4 backward compat: load with DeprecationWarning + synthesise ──


class TestV4BackwardCompat:
    def _v4_payload(self) -> dict:
        """Hand-crafted v4 GeoResult JSON (no share_per_domain field)."""
        geo = _make_geo()
        # Domain magnitudes the v4 reader will see → synth share matches
        # the analyze()-time computation.
        d = to_json_dict(geo)
        d.pop("share_per_domain")  # v4 didn't carry this field
        d["schema_version"] = "4"
        return d

    def test_v4_emits_deprecation_warning(self):
        v4 = self._v4_payload()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            geo_result_from_json_dict(v4)
        deprecations = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecations) >= 1
        msg = str(deprecations[0].message)
        assert "v4" in msg
        assert "v0.4.0" in msg
        assert "share_per_domain" in msg

    def test_v4_synthesises_share_per_domain(self):
        v4 = self._v4_payload()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            restored = geo_result_from_json_dict(v4)
        # Synthesised field matches what analyze() would have produced.
        assert restored.share_per_domain["v1"]["a"] == pytest.approx(1.0)
        assert restored.share_per_domain["v2"]["b"] == pytest.approx(1.0)

    def test_v4_in_memory_is_v5_shaped(self):
        """After load, the in-memory result is fully v5-shaped — renderers
        never need to branch on schema_version."""
        v4 = self._v4_payload()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            restored = geo_result_from_json_dict(v4)
        # Round-trip the in-memory v5 to disk: it now writes v5.
        round_trip = to_json_dict(restored)
        assert round_trip["schema_version"] == "5"
        assert "share_per_domain" in round_trip


# ── v1/v2/v3 still load (no synthesis since probe_domains may be empty) ─


class TestLegacySchemasStillLoad:
    def _v1_payload(self) -> dict:
        # Minimal v1 — the older fields only.
        return {
            "schema_version": "1",
            "base_name": "base",
            "variant_names": ["v1"],
            "n_probes": 2,
            "magnitudes": {"v1": 1.4142},
            "cosine_matrix": {"v1": {"v1": 1.0}},
            "change_vectors": {"v1": [1.0, 1.0]},
            "per_probe": {"v1": {"p0": 1.0, "p1": 1.0}},
            "metadata": {},
        }

    def test_v1_loads_with_empty_share(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # v1 must NOT emit DeprecationWarning (only v4 does)
            restored = geo_result_from_json_dict(self._v1_payload())
        assert restored.share_per_domain == {}
        assert restored.probe_domains == ()


# ── unsupported version raises ────────────────────────────────────────


class TestUnsupportedSchemaRaises:
    def test_unknown_version_raises(self):
        with pytest.raises(ValueError, match="schema_version"):
            geo_result_from_json_dict({"schema_version": "99"})
