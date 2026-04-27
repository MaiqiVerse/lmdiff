"""JSON renderer polish tests (commit 1.11 of v0.3.0 batch 4).

PR #4 already shipped schema v5 with the ``share_per_domain`` field. This
test module is a verification pass: deterministic output, round-trip
through ``lmdiff.load_result``, all v5 fields present, ``schema_version``
is the string ``"5"``.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from lmdiff.geometry import GeoResult, _compute_share_per_domain
from lmdiff.report import json_report as json_mod
from lmdiff.report.json_report import (
    SCHEMA_VERSION,
    geo_result_from_json_dict,
    to_json,
    to_json_dict,
    write_json,
)


def _make_geo() -> GeoResult:
    n = 4
    cv = {
        "yarn": [3.0, 4.0, 0.0, 0.0],
        "long": [0.0, 0.0, 6.0, 8.0],
    }
    geo = GeoResult(
        base_name="base-mock",
        variant_names=list(cv),
        n_probes=n,
        magnitudes={
            "yarn": float(np.linalg.norm(cv["yarn"])),
            "long": float(np.linalg.norm(cv["long"])),
        },
        cosine_matrix={v: {w: 1.0 if v == w else 0.0 for w in cv} for v in cv},
        change_vectors=cv,
        per_probe={v: {f"p{i}": cv[v][i] for i in range(n)} for v in cv},
        metadata={"max_new_tokens": 16, "n_skipped": 0},
        probe_domains=("a", "a", "b", "b"),
        avg_tokens_per_probe=tuple([8.0] * n),
        magnitudes_normalized={v: float(np.linalg.norm(cv[v]) / 2.0) for v in cv},
    )
    geo.share_per_domain = _compute_share_per_domain(geo)
    return geo


# ── Schema version string ─────────────────────────────────────────────


def test_schema_version_constant_is_v5():
    assert SCHEMA_VERSION == "5"


def test_emitted_dict_uses_string_v5():
    payload = to_json_dict(_make_geo())
    assert payload["schema_version"] == "5"
    assert isinstance(payload["schema_version"], str)


# ── v5 field presence ────────────────────────────────────────────────


def test_all_v5_fields_present():
    payload = to_json_dict(_make_geo())
    expected = {
        "schema_version", "base_name", "variant_names", "n_probes",
        "magnitudes", "cosine_matrix", "change_vectors", "per_probe",
        "metadata", "delta_means", "selective_magnitudes",
        "selective_cosine_matrix", "probe_domains", "avg_tokens_per_probe",
        "magnitudes_normalized", "share_per_domain", "generated_at",
    }
    missing = expected - payload.keys()
    assert missing == set(), f"missing v5 fields: {missing}"


def test_share_per_domain_rows_sum_to_one():
    payload = to_json_dict(_make_geo())
    for variant, by_dom in payload["share_per_domain"].items():
        assert sum(by_dom.values()) == pytest.approx(1.0, abs=1e-9), variant


# ── Deterministic output ─────────────────────────────────────────────


def test_to_json_deterministic_modulo_timestamp():
    """Same in-memory result → same JSON bytes ignoring generated_at."""
    geo = _make_geo()
    a = to_json(geo)
    b = to_json(geo)
    da = json.loads(a)
    db = json.loads(b)
    da.pop("generated_at", None)
    db.pop("generated_at", None)
    assert da == db


def test_to_json_keys_alphabetical():
    payload_str = to_json(_make_geo())
    # sort_keys=True is on; verify by reparsing and comparing keys to sorted.
    obj = json.loads(payload_str)
    assert list(obj.keys()) == sorted(obj.keys())


# ── Round-trip ───────────────────────────────────────────────────────


def test_round_trip_preserves_all_fields(tmp_path):
    geo = _make_geo()
    path = tmp_path / "geo.json"
    write_json(geo, path)

    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    restored = geo_result_from_json_dict(payload)

    assert restored.variant_names == geo.variant_names
    assert restored.n_probes == geo.n_probes
    assert restored.magnitudes == geo.magnitudes
    assert restored.share_per_domain == geo.share_per_domain
    assert restored.probe_domains == geo.probe_domains
    assert restored.avg_tokens_per_probe == geo.avg_tokens_per_probe
    assert restored.magnitudes_normalized == geo.magnitudes_normalized


def test_round_trip_via_load_result(tmp_path):
    """``lmdiff.load_result`` round-trip via the public API."""
    from lmdiff import load_result

    geo = _make_geo()
    path = tmp_path / "x.json"
    write_json(geo, path)

    restored = load_result(str(path))
    assert restored.share_per_domain == geo.share_per_domain
    assert restored.cosine_matrix == geo.cosine_matrix


# ── v4 backward-compat still loads with DeprecationWarning ───────────


def test_v4_loads_with_deprecation_warning():
    geo = _make_geo()
    payload = to_json_dict(geo)
    payload.pop("share_per_domain")
    payload["schema_version"] = "4"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        restored = geo_result_from_json_dict(payload)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    # share_per_domain synthesised from v4 data.
    assert restored.share_per_domain
    # Reserialise: now writes v5.
    restored_payload = to_json_dict(restored)
    assert restored_payload["schema_version"] == "5"
    assert "share_per_domain" in restored_payload


# ── render() module-level function (Renderer Protocol) ──────────────


class TestJsonModuleRender:
    def test_returns_dict(self):
        out = json_mod.render(_make_geo())
        assert isinstance(out, dict)
        assert out["schema_version"] == "5"

    def test_writes_file_when_path_given(self, tmp_path):
        out = json_mod.render(_make_geo(), path=tmp_path / "x.json")
        # Module-level render returns the dict either way.
        assert isinstance(out, dict)
        assert (tmp_path / "x.json").exists()
        text = (tmp_path / "x.json").read_text(encoding="utf-8")
        assert json.loads(text)["schema_version"] == "5"
