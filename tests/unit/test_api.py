"""Unit tests for the v0.3.0 top-level ``compare`` / ``family`` API.

All tests use mocks; no real model loads. The two entry points route
through ``ChangeGeometry.analyze`` which lives in v0.2.x territory, so
the tests here focus on the *new* behavior:

  - argument coercion (``str`` → ``Config``)
  - probe spec resolution
  - metric resolution + experimental rejection
  - capability negotiation runs **before** any work
  - engine-template ownership semantics (close own engines; not the
    user's template)
  - lazy-import contract
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from lmdiff import (
    CapabilityError,
    Config,
    DecodeSpec,
    compare,
    family,
)
from lmdiff._api import (
    _check_capabilities,
    _coerce_to_config,
    _coerce_to_probe_set,
    _resolve_metrics,
)

from tests.fixtures.mock_engine import MockEngine


# ── _coerce_to_config ─────────────────────────────────────────────────


class TestCoerceToConfig:
    def test_str_becomes_config(self):
        cfg = _coerce_to_config("gpt2")
        assert isinstance(cfg, Config)
        assert cfg.model == "gpt2"

    def test_config_returned_unchanged(self):
        cfg = Config(model="gpt2", system_prompt="hi")
        assert _coerce_to_config(cfg) is cfg

    def test_other_type_raises(self):
        with pytest.raises(TypeError, match="expected str or lmdiff.Config"):
            _coerce_to_config(42)


# ── _coerce_to_probe_set ──────────────────────────────────────────────


class TestCoerceToProbeSet:
    def test_none_loads_v01(self):
        ps = _coerce_to_probe_set(None)
        # v01 is the bundled 90-probe set
        assert len(ps) > 0
        assert ps.name in (None, "v01") or "v01" in str(ps.name)

    def test_v01_string_loads_bundled(self):
        ps = _coerce_to_probe_set("v01")
        assert len(ps) > 0

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unrecognized"):
            _coerce_to_probe_set("nonexistent_probe_set_xyz")

    def test_lm_eval_empty_tasks_raises(self):
        with pytest.raises(ValueError, match="no task names"):
            _coerce_to_probe_set("lm_eval:")

    def test_other_type_raises(self):
        with pytest.raises(TypeError):
            _coerce_to_probe_set(123)


# ── _resolve_metrics ──────────────────────────────────────────────────


class TestResolveMetrics:
    def test_default_returns_full_set(self):
        out = _resolve_metrics("default")
        assert "bd" in out
        assert "drift" in out
        assert "specialization_zscore" in out

    def test_subset_list(self):
        out = _resolve_metrics(["bd", "drift"])
        assert out == ["bd", "drift"]

    def test_single_str_wrapped(self):
        out = _resolve_metrics("bd")
        assert out == ["bd"]

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="unknown metric"):
            _resolve_metrics(["nonexistent"])

    def test_contrib_metric_phase_6_pointer(self):
        with pytest.raises(NotImplementedError, match="Phase 6"):
            _resolve_metrics(["lmdiff.contrib.foo"])

    def test_non_str_metric_raises(self):
        with pytest.raises(TypeError, match="metric names must be str"):
            _resolve_metrics([42])

    def test_invalid_arg_type_raises(self):
        with pytest.raises(TypeError, match="must be"):
            _resolve_metrics(42)


# ── _check_capabilities ───────────────────────────────────────────────


class TestCheckCapabilities:
    def test_default_engine_passes_default_metrics(self):
        engine = MockEngine()
        _check_capabilities(["bd", "drift"], engine)

    def test_missing_capability_raises_with_metric_name(self):
        engine = MockEngine(capabilities=frozenset({"score"}))  # no `generate`
        with pytest.raises(CapabilityError) as exc_info:
            _check_capabilities(["bd"], engine)
        err = exc_info.value
        assert err.metric_name == "bd"
        assert "generate" in err.missing
        assert "bd" in str(err)

    def test_checks_all_engines(self):
        eng_ok = MockEngine()
        eng_bad = MockEngine(capabilities=frozenset({"score"}))
        with pytest.raises(CapabilityError):
            _check_capabilities(["bd"], eng_ok, eng_bad)


# ── compare() ─────────────────────────────────────────────────────────


class TestCompareWiring:
    """Tests that don't require a real ChangeGeometry run.

    We patch the heavy ChangeGeometry pipeline so we can validate just the
    coercion + capability + ownership wiring without spinning a model.
    """

    def test_string_args_coerced_and_capabilities_checked(self):
        engine = MockEngine()
        with patch("lmdiff.geometry.ChangeGeometry") as MockCG:
            MockCG.return_value.analyze.return_value = "FAKE_GEORESULT"
            result = compare("gpt2", "distilgpt2", n_probes=5, engine=engine)
        assert result == "FAKE_GEORESULT"
        # ChangeGeometry called with v0.2.x configs whose .model match
        kwargs = MockCG.call_args.kwargs
        assert kwargs["base"].model == "gpt2"
        # variants is {"variant": v02_cfg} when no name given
        assert "variant" in kwargs["variants"]
        assert kwargs["variants"]["variant"].model == "distilgpt2"

    def test_capability_mismatch_raises_before_changegeometry_runs(self):
        engine = MockEngine(capabilities=frozenset({"score"}))
        with patch("lmdiff.geometry.ChangeGeometry") as MockCG:
            with pytest.raises(CapabilityError, match="generate"):
                compare("a", "b", n_probes=5, engine=engine)
        # ChangeGeometry must NOT have been instantiated.
        MockCG.assert_not_called()

    def test_unknown_metric_raises_before_changegeometry_runs(self):
        engine = MockEngine()
        with patch("lmdiff.geometry.ChangeGeometry") as MockCG:
            with pytest.raises(ValueError, match="unknown metric"):
                compare("a", "b", metrics=["bogus"], engine=engine)
        MockCG.assert_not_called()

    def test_contrib_metric_raises_before_changegeometry_runs(self):
        engine = MockEngine()
        with patch("lmdiff.geometry.ChangeGeometry") as MockCG:
            with pytest.raises(NotImplementedError, match="Phase 6"):
                compare("a", "b", metrics=["lmdiff.contrib.x"], engine=engine)
        MockCG.assert_not_called()

    def test_invalid_task_overrides_type_raises(self):
        engine = MockEngine()
        with pytest.raises(TypeError, match="task_overrides"):
            compare("a", "b", task_overrides="not-a-dict", engine=engine)

    def test_n_probes_truncates(self):
        engine = MockEngine()
        with patch("lmdiff.geometry.ChangeGeometry") as MockCG:
            MockCG.return_value.analyze.return_value = "OK"
            compare("a", "b", n_probes=3, engine=engine)
        # The probe set passed to ChangeGeometry has at most 3 entries.
        prompts = MockCG.call_args.kwargs["prompts"]
        assert len(prompts) <= 3


class TestCompareEngineOwnership:
    """When ``compare`` builds its own engines it must close them in finally."""

    def test_template_close_is_NOT_called_on_user_engine(self):
        engine = MockEngine()
        engine_close_called = []
        original_close = engine.close

        def tracking_close():
            engine_close_called.append(True)
            original_close()
        engine.close = tracking_close

        with patch("lmdiff.geometry.ChangeGeometry") as MockCG:
            MockCG.return_value.analyze.return_value = "OK"
            compare("a", "b", n_probes=2, engine=engine)
        # User-passed template was NOT closed by compare().
        assert engine_close_called == []

    def test_per_variant_engines_built_via_with_config(self):
        engine = MockEngine()
        original_with_config = engine.with_config
        call_count = []

        def tracking_with_config(cfg):
            call_count.append(cfg.model)
            return original_with_config(cfg)
        engine.with_config = tracking_with_config

        with patch("lmdiff.geometry.ChangeGeometry") as MockCG:
            MockCG.return_value.analyze.return_value = "OK"
            compare("base_x", "variant_y", n_probes=2, engine=engine)
        assert "base_x" in call_count
        assert "variant_y" in call_count


# ── family() ──────────────────────────────────────────────────────────


class TestFamilyWiring:
    def test_dict_variants_coerced(self):
        engine = MockEngine()
        with patch("lmdiff.geometry.ChangeGeometry") as MockCG:
            MockCG.return_value.analyze.return_value = "FAKE"
            family(
                "base_model",
                {"yarn": "yarn_path", "code": "code_path"},
                n_probes=3,
                engine=engine,
            )
        variants = MockCG.call_args.kwargs["variants"]
        assert set(variants.keys()) == {"yarn", "code"}
        assert variants["yarn"].model == "yarn_path"
        assert variants["code"].model == "code_path"

    def test_empty_variants_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            family("a", {})

    def test_capability_mismatch_short_circuits(self):
        engine = MockEngine(capabilities=frozenset({"score"}))
        with patch("lmdiff.geometry.ChangeGeometry") as MockCG:
            with pytest.raises(CapabilityError):
                family("a", {"v1": "b"}, engine=engine)
        MockCG.assert_not_called()


# ── Lazy import contract ──────────────────────────────────────────────


class TestLazyImport:
    def test_compare_family_lazy(self):
        if "torch" in sys.modules:
            pytest.skip("torch already loaded")
        from lmdiff import compare, family  # noqa: F401
        assert "torch" not in sys.modules
        assert "transformers" not in sys.modules
