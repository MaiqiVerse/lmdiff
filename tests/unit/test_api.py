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
        ps, is_lm_eval, info = _coerce_to_probe_set(None)
        # v01 is the bundled 90-probe set
        assert len(ps) > 0
        assert ps.name in (None, "v01") or "v01" in str(ps.name)
        assert is_lm_eval is False
        assert info == {}

    def test_v01_string_loads_bundled(self):
        ps, is_lm_eval, info = _coerce_to_probe_set("v01")
        assert len(ps) > 0
        assert is_lm_eval is False

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


# ── v0.3.2: per-task n_probes semantics on lm_eval multi-task strings ──


class TestNProbesLmEvalSemantics:
    """v0.3.2 fix: ``n_probes=N`` on a multi-task ``lm_eval:`` string is
    *per-task*, not total. Without this, ``lm_eval:t1+t2+...`` with
    ``n_probes=100`` had loaded ALL probes from each task, concatenated,
    and sliced the first 100 — which on multi-task specs always landed
    inside the first task only (the v0.3.1 surprise the user hit)."""

    def test_lm_eval_multitask_per_task_limit_applied(self, monkeypatch):
        # Monkeypatch from_lm_eval so we don't need lm-eval-harness installed
        # to verify the call shape — we just want to confirm `limit=` is
        # passed per-task.
        captured: list[tuple[str, int | None]] = []

        def fake_from_lm_eval(task_name, limit=None, num_fewshot=None, seed=42):
            captured.append((task_name, limit))
            from lmdiff.probes.loader import Probe, ProbeSet
            n = limit if limit is not None else 5
            return ProbeSet(
                [
                    Probe(id=f"{task_name}_{i}", text=f"p{i}", domain=task_name)
                    for i in range(n)
                ],
                name=task_name,
                version="lm-eval-harness",
            )

        monkeypatch.setattr(
            "lmdiff.probes.adapters.from_lm_eval", fake_from_lm_eval,
        )

        from lmdiff._api import _coerce_to_probe_set
        ps, is_lm_eval, info = _coerce_to_probe_set(
            "lm_eval:hellaswag+arc_challenge+gsm8k", n_probes=20,
        )
        # `limit=20` was forwarded to every task — not just the first.
        assert captured == [
            ("hellaswag", 20), ("arc_challenge", 20), ("gsm8k", 20),
        ]
        # 3 tasks × 20 = 60 total probes (the v0.3.2 contract).
        assert len(ps) == 60
        assert is_lm_eval is True
        assert info["n_probes_per_task"] == 20
        assert info["task_breakdown"] == {
            "hellaswag": 20, "arc_challenge": 20, "gsm8k": 20,
        }
        # All 3 task domains represented (the user's bug fix).
        assert set(p.domain for p in ps) == {
            "hellaswag", "arc_challenge", "gsm8k",
        }

    def test_lm_eval_single_task_per_task_equals_total(self, monkeypatch):
        from lmdiff.probes.loader import Probe, ProbeSet
        captured = []

        def fake(task_name, limit=None, num_fewshot=None, seed=42):
            captured.append((task_name, limit))
            return ProbeSet(
                [
                    Probe(id=f"{task_name}_{i}", text=f"p{i}", domain=task_name)
                    for i in range(limit or 1)
                ],
                name=task_name, version="lm-eval-harness",
            )

        monkeypatch.setattr("lmdiff.probes.adapters.from_lm_eval", fake)
        from lmdiff._api import _coerce_to_probe_set
        ps, is_lm_eval, info = _coerce_to_probe_set("lm_eval:hellaswag", n_probes=50)
        assert captured == [("hellaswag", 50)]
        assert len(ps) == 50

    def test_compare_does_not_double_truncate_lm_eval(self, monkeypatch):
        """compare() must skip the secondary slice for lm_eval inputs;
        otherwise the per-task expansion would be re-truncated at the
        front of the merged set (v0.3.1 bug)."""
        from lmdiff.probes.loader import Probe, ProbeSet

        def fake(task_name, limit=None, num_fewshot=None, seed=42):
            n = limit if limit is not None else 5
            return ProbeSet(
                [
                    Probe(id=f"{task_name}_{i}", text=f"p{i}", domain=task_name)
                    for i in range(n)
                ],
                name=task_name, version="lm-eval-harness",
            )

        monkeypatch.setattr("lmdiff.probes.adapters.from_lm_eval", fake)

        from unittest.mock import patch
        engine = MockEngine()
        with patch("lmdiff.geometry.ChangeGeometry") as MockCG:
            MockCG.return_value.analyze.return_value = MockEngine()  # placeholder
            MockCG.return_value.analyze.return_value.metadata = {}
            from lmdiff import compare
            compare(
                "a", "b",
                probes="lm_eval:hellaswag+arc_challenge+gsm8k",
                n_probes=4,
                engine=engine,
            )
        # ChangeGeometry was given the full merged set (3 × 4 = 12 probes),
        # NOT a truncated-to-4 slice.
        kwargs = MockCG.call_args.kwargs
        assert len(kwargs["prompts"]) == 12

    def test_flat_probe_set_total_semantics_unchanged(self):
        """Flat probe sets (bundled v01) keep "total" semantics — no
        per-task expansion."""
        from lmdiff._api import _coerce_to_probe_set
        ps, is_lm_eval, info = _coerce_to_probe_set("v01", n_probes=10)
        assert is_lm_eval is False
        assert info == {}
        # The slice happens in compare()/family(), not in
        # _coerce_to_probe_set, so the returned set is the full v01.
        assert len(ps) > 10

    def test_metadata_carries_task_breakdown_through_family(self, monkeypatch):
        """The resolved task_breakdown info should land in
        ``GeoResult.metadata`` so renderers can show "5 tasks × 100 each"."""
        from lmdiff.probes.loader import Probe, ProbeSet

        def fake(task_name, limit=None, num_fewshot=None, seed=42):
            n = limit if limit is not None else 5
            return ProbeSet(
                [
                    Probe(id=f"{task_name}_{i}", text=f"p{i}", domain=task_name)
                    for i in range(n)
                ],
                name=task_name, version="lm-eval-harness",
            )

        monkeypatch.setattr("lmdiff.probes.adapters.from_lm_eval", fake)

        from unittest.mock import patch
        from lmdiff.geometry import GeoResult
        engine = MockEngine()

        # ChangeGeometry returns a real-shaped fake GeoResult so we can
        # check that probe_info is merged into metadata.
        fake_geo = GeoResult(
            base_name="b",
            variant_names=["v"],
            n_probes=15,
            magnitudes={"v": 1.0},
            cosine_matrix={"v": {"v": 1.0}},
            change_vectors={"v": [0.0] * 15},
            per_probe={"v": {f"p{i}": 0.0 for i in range(15)}},
            metadata={},
        )
        with patch("lmdiff.geometry.ChangeGeometry") as MockCG:
            MockCG.return_value.analyze.return_value = fake_geo
            from lmdiff import family
            result = family(
                "base",
                {"v1": "m"},
                probes="lm_eval:hellaswag+arc_challenge+gsm8k",
                n_probes=5,
                engine=engine,
            )
        assert result.metadata.get("n_probes_per_task") == 5
        assert result.metadata.get("task_breakdown") == {
            "hellaswag": 5, "arc_challenge": 5, "gsm8k": 5,
        }
        assert result.metadata.get("tasks") == [
            "hellaswag", "arc_challenge", "gsm8k",
        ]
