"""``InferenceEngine`` runtime-param overrides (v0.3.2 engine reuse).

These tests validate the contract that one engine instance can serve
multiple Configs that differ only in runtime-only fields. The engine
must take ``system_prompt`` / ``context`` / ``decode`` as method-level
overrides and use them in place of ``self.config`` for the call,
without polluting state across calls.

We don't need a real model for this — we exercise the prompt-building
helpers directly to verify they consume the per-call override.
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from lmdiff.config import Config as V02Config


def _make_engine_skeleton(cfg: "V02Config"):
    """Construct an InferenceEngine WITHOUT loading a model.

    ``InferenceEngine.__init__`` calls ``self._load`` which downloads/
    loads weights. For prompt-building tests we don't need that; we
    bypass the model load and stub the tokenizer.
    """
    from lmdiff.engine import InferenceEngine

    eng = InferenceEngine.__new__(InferenceEngine)
    eng.config = cfg
    eng.device = "cpu"
    eng._model = MagicMock()
    # Tokenizer stub: returns predictable ids based on text length so
    # we can assert the prompt content reached the encoder.
    fake_tok = MagicMock()

    def _tokenize(text, add_special_tokens=True):
        return {"input_ids": list(range(len(text)))}

    fake_tok.side_effect = _tokenize
    fake_tok.__call__ = _tokenize
    # MagicMock already supports calling — wire side_effect on the tok itself.
    fake_tok.side_effect = _tokenize
    eng._tokenizer = fake_tok
    return eng


# ── _resolve helpers ────────────────────────────────────────────────


class TestResolveSystemPromptFallback:
    def test_explicit_override_wins(self):
        cfg = V02Config(model="gpt2", system_prompt="from-config")
        eng = _make_engine_skeleton(cfg)
        assert eng._resolve_system_prompt("override") == "override"

    def test_none_falls_back_to_config(self):
        cfg = V02Config(model="gpt2", system_prompt="from-config")
        eng = _make_engine_skeleton(cfg)
        assert eng._resolve_system_prompt(None) == "from-config"

    def test_none_with_no_config_value_stays_none(self):
        cfg = V02Config(model="gpt2", system_prompt=None)
        eng = _make_engine_skeleton(cfg)
        assert eng._resolve_system_prompt(None) is None


class TestResolveContextFallback:
    def test_explicit_override_wins(self):
        cfg = V02Config(model="gpt2", context=[{"content": "from-config"}])
        eng = _make_engine_skeleton(cfg)
        assert eng._resolve_context([{"content": "override"}]) == [
            {"content": "override"},
        ]

    def test_none_falls_back_to_config(self):
        cfg = V02Config(model="gpt2", context=[{"content": "from-config"}])
        eng = _make_engine_skeleton(cfg)
        assert eng._resolve_context(None) == [{"content": "from-config"}]


class TestResolveDecodeFallback:
    def test_explicit_override_wins(self):
        cfg = V02Config(model="gpt2", decode={"strategy": "greedy"})
        eng = _make_engine_skeleton(cfg)
        assert eng._resolve_decode({"strategy": "sample"}) == {"strategy": "sample"}

    def test_none_falls_back_to_config(self):
        cfg = V02Config(model="gpt2", decode={"strategy": "greedy"})
        eng = _make_engine_skeleton(cfg)
        assert eng._resolve_decode(None) == {"strategy": "greedy"}


# ── _build_prompt + _prefix_text honour overrides ───────────────────


class TestBuildPromptUsesOverrides:
    def test_no_override_uses_config_system_prompt(self):
        cfg = V02Config(model="gpt2", system_prompt="cfg-sys")
        eng = _make_engine_skeleton(cfg)
        prompt = eng._build_prompt("user-text")
        assert prompt.startswith("cfg-sys")
        assert "user-text" in prompt

    def test_explicit_override_replaces_config_system_prompt(self):
        cfg = V02Config(model="gpt2", system_prompt="cfg-sys")
        eng = _make_engine_skeleton(cfg)
        prompt = eng._build_prompt("user-text", system_prompt="override-sys")
        assert "cfg-sys" not in prompt
        assert prompt.startswith("override-sys")

    def test_explicit_override_with_no_config(self):
        cfg = V02Config(model="gpt2")  # system_prompt=None
        eng = _make_engine_skeleton(cfg)
        prompt = eng._build_prompt("user-text", system_prompt="from-call")
        assert prompt.startswith("from-call")

    def test_call_with_override_does_not_pollute_state(self):
        # The engine's stored config must be unchanged after a call
        # with overrides. (Critical: if we accidentally mutated
        # self.config during the override path, two calls in
        # different orders would produce different results.)
        cfg = V02Config(model="gpt2", system_prompt="original")
        eng = _make_engine_skeleton(cfg)
        eng._build_prompt("x", system_prompt="ephemeral")
        # Subsequent call with no override should still get "original".
        result = eng._build_prompt("y")
        assert result.startswith("original")


# ── _decode_params honours override ─────────────────────────────────


class TestDecodeParamsUsesOverride:
    def test_greedy_default(self):
        cfg = V02Config(model="gpt2", decode={"strategy": "greedy"})
        eng = _make_engine_skeleton(cfg)
        params = eng._decode_params()
        assert params == {"do_sample": False}

    def test_override_changes_strategy(self):
        cfg = V02Config(model="gpt2", decode={"strategy": "greedy"})
        eng = _make_engine_skeleton(cfg)
        params = eng._decode_params(
            {"strategy": "sample", "temperature": 1.5},
        )
        assert params["do_sample"] is True
        assert params["temperature"] == 1.5

    def test_override_doesnt_modify_config(self):
        cfg = V02Config(model="gpt2", decode={"strategy": "greedy"})
        eng = _make_engine_skeleton(cfg)
        eng._decode_params({"strategy": "sample", "temperature": 0.5})
        # Stored config still reflects original.
        assert eng.config.decode == {"strategy": "greedy"}


# ── score / generate signature accepts the kwargs ───────────────────


class TestSignaturesExposeOverrideKwargs:
    def test_generate_accepts_runtime_kwargs(self):
        import inspect
        from lmdiff.engine import InferenceEngine
        sig = inspect.signature(InferenceEngine.generate)
        for name in ("system_prompt", "context", "decode"):
            assert name in sig.parameters, name

    def test_score_accepts_runtime_kwargs(self):
        import inspect
        from lmdiff.engine import InferenceEngine
        sig = inspect.signature(InferenceEngine.score)
        for name in ("system_prompt", "context"):
            assert name in sig.parameters, name
        # score doesn't use decode (no sampling involved).
        assert "decode" not in sig.parameters


# ── geometry call site passes overrides ─────────────────────────────


class TestGeometryThreadsOverridesToEngine:
    """When ``_delta_for_variant`` runs, it should pass v_config's
    runtime params to the engine — required for engine reuse where the
    same engine serves base + a runtime-only variant."""

    def test_delta_for_variant_passes_v_config_overrides(self):
        # Build a ChangeGeometry with mock engines, capture the call
        # kwargs to .generate / .score, assert overrides flow through.
        from lmdiff.geometry import ChangeGeometry

        base_cfg = V02Config(model="gpt2", system_prompt="base-sys")
        v_cfg = V02Config(model="gpt2", system_prompt="variant-sys", name="V")

        # Mock engines that record call kwargs.
        base_eng = MagicMock(name="base_eng")
        base_eng.config = base_cfg
        base_eng.tokenizer = MagicMock()
        # generate returns a result with one completion per prompt
        gen_out = MagicMock(
            completions=[["c1"], ["c2"]],
            token_ids=[[[1, 2]], [[3, 4]]],
        )
        v_eng = MagicMock(name="v_eng")
        v_eng.config = v_cfg
        v_eng.tokenizer = MagicMock()
        v_eng.generate = MagicMock(return_value=gen_out)

        # Both engines' score returns matching CEs; the values don't
        # matter for this test — we only inspect call kwargs.
        score_out = MagicMock(
            cross_entropies=[1.0, 1.0],
            token_ids=[[1, 2], [3, 4]],
        )
        base_eng.score = MagicMock(return_value=score_out)
        v_eng.score = MagicMock(return_value=score_out)

        # Stub shares_tokenizer_with → True so BPB code path doesn't fire
        cg = ChangeGeometry.__new__(ChangeGeometry)
        cg.base_config = base_cfg
        cg.variants = {"V": v_cfg}
        cg.prompts = ["p0", "p1"]
        cg.probe_set = None
        cg.n_samples = 1

        cg._delta_for_variant(
            base_engine=base_eng,
            v_engine=v_eng,
            v_config=v_cfg,
            max_new_tokens=8,
        )

        # v_eng.generate must have been called with v_cfg's runtime
        # params — this is what makes engine sharing safe.
        gen_kwargs = v_eng.generate.call_args.kwargs
        assert gen_kwargs.get("system_prompt") == "variant-sys"
        assert gen_kwargs.get("decode") == v_cfg.decode

        # v_eng.score (self-score path) also passes v_cfg overrides.
        v_score_kwargs = v_eng.score.call_args.kwargs
        assert v_score_kwargs.get("system_prompt") == "variant-sys"

        # base_eng.score uses base's own config — no override needed.
        base_score_kwargs = base_eng.score.call_args.kwargs
        assert "system_prompt" not in base_score_kwargs or \
               base_score_kwargs.get("system_prompt") is None
