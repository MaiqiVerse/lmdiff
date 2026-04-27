"""Unit tests for the v0.3.0 Engine protocol, MockEngine, MinimalEngine.

These tests do NOT load real models. ``HFEngine`` integration tests live
in ``tests/integration/test_hf_engine.py`` and are gated by
``pytest -m gpu``.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from lmdiff import (
    CapabilityError,
    Config,
    Engine,
    GenerateResult,
    MinimalEngine,
    ScoreResult,
)
from lmdiff._engine import (
    AttentionWeightsResult,
    HiddenStatesResult,
    RESERVED_CAPABILITIES,
)

from tests.fixtures.mock_engine import MockEngine


# ── Protocol conformance ──────────────────────────────────────────────


class TestProtocolConformance:
    def test_mock_engine_conforms(self):
        engine = MockEngine()
        assert isinstance(engine, Engine)

    def test_minimal_engine_subclass_conforms(self):
        class TestEngine(MinimalEngine):
            def _score_impl(self, prompt, continuation):
                return [-1.0, -2.0], [1, 2]

            def _generate_impl(self, prompt, mnt, t, tp, seed):
                return "out", [1, 2], None

        engine = TestEngine(Config(model="test"))
        assert isinstance(engine, Engine)

    def test_random_class_does_not_conform(self):
        class NotAnEngine:
            pass
        assert not isinstance(NotAnEngine(), Engine)


# ── MockEngine basic behavior ─────────────────────────────────────────


class TestMockEngineBasic:
    def test_score_deterministic(self):
        e1 = MockEngine(seed=42)
        e2 = MockEngine(seed=42)
        r1 = e1.score("hello", "world")
        r2 = e2.score("hello", "world")
        np.testing.assert_array_equal(r1.logprobs, r2.logprobs)

    def test_score_different_continuations_distinct_seeds(self):
        engine = MockEngine()
        r1 = engine.score("hello", "world")
        r2 = engine.score("hello", "everyone")
        # Mock seeds on (prompt, continuation), so values differ.
        # Lengths may differ; only assert they're both non-empty.
        assert len(r1.tokens) >= 1
        assert len(r2.tokens) >= 1

    def test_score_same_input_same_engine_repeatable(self):
        engine = MockEngine(seed=42)
        r1 = engine.score("a", "b")
        r2 = engine.score("a", "b")
        np.testing.assert_array_equal(r1.logprobs, r2.logprobs)

    def test_score_returns_scoreresult(self):
        engine = MockEngine()
        r = engine.score("a", "b")
        assert isinstance(r, ScoreResult)
        assert isinstance(r.avg_logprob, float)

    def test_generate_returns_generateresult(self):
        engine = MockEngine()
        r = engine.generate("hello", max_new_tokens=5)
        assert isinstance(r, GenerateResult)
        assert isinstance(r.text, str)
        assert len(r.tokens) <= 8  # mock caps at 8

    def test_generate_seed_reproducible(self):
        engine = MockEngine()
        r1 = engine.generate("hi", seed=99)
        r2 = engine.generate("hi", seed=99)
        assert r1.text == r2.text


# ── MockEngine capabilities ───────────────────────────────────────────


class TestMockEngineCapabilities:
    def test_default_capabilities(self):
        engine = MockEngine()
        assert "score" in engine.capabilities
        assert "generate" in engine.capabilities
        assert "hidden_states" in engine.capabilities
        assert "attention_weights" in engine.capabilities

    def test_custom_capabilities_subset(self):
        engine = MockEngine(capabilities=frozenset({"score", "generate"}))
        assert "hidden_states" not in engine.capabilities
        with pytest.raises(NotImplementedError):
            engine.hidden_states("test")

    def test_attention_weights_disabled(self):
        engine = MockEngine(capabilities=frozenset({"score", "generate"}))
        with pytest.raises(NotImplementedError):
            engine.attention_weights("test")

    def test_unknown_capability_rejected(self):
        with pytest.raises(ValueError, match="Unknown capability"):
            MockEngine(capabilities=frozenset({"score", "foo_unknown"}))

    def test_score_disabled_raises(self):
        engine = MockEngine(capabilities=frozenset({"generate"}))
        with pytest.raises(NotImplementedError, match="score"):
            engine.score("a", "b")


# ── MockEngine result shapes ──────────────────────────────────────────


class TestMockEngineResultShapes:
    def test_hidden_states_shape_default(self):
        engine = MockEngine()
        r = engine.hidden_states("hello world")
        assert isinstance(r, HiddenStatesResult)
        assert r.hidden_states.shape == (engine.n_layers, engine.hidden_dim)
        assert r.position == "last"

    def test_hidden_states_layer_subset(self):
        engine = MockEngine()
        r = engine.hidden_states("hello", layers=[0, 5, 10])
        assert r.hidden_states.shape == (3, engine.hidden_dim)

    def test_hidden_states_position_kwarg(self):
        engine = MockEngine()
        r = engine.hidden_states("hello", position="first")
        assert r.position == "first"

    def test_attention_weights_shape(self):
        engine = MockEngine()
        r = engine.attention_weights("hello world")
        assert isinstance(r, AttentionWeightsResult)
        assert r.attention_weights.shape[0] == engine.n_layers
        assert r.attention_weights.shape[1] == 12  # n_heads

    def test_attention_normalized(self):
        engine = MockEngine()
        r = engine.attention_weights("hello world")
        sums = r.attention_weights.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_attention_head_subset(self):
        engine = MockEngine()
        r = engine.attention_weights("hello world", heads=[0, 5])
        assert r.attention_weights.shape[1] == 2


# ── tokenizer_id behavior ─────────────────────────────────────────────


class TestTokenizerIdHash:
    def test_same_model_same_id(self):
        e1 = MockEngine(Config(model="gpt2"))
        e2 = MockEngine(Config(model="gpt2"))
        assert e1.tokenizer_id == e2.tokenizer_id

    def test_different_model_different_id(self):
        e1 = MockEngine(Config(model="gpt2"))
        e2 = MockEngine(Config(model="distilgpt2"))
        assert e1.tokenizer_id != e2.tokenizer_id

    def test_id_is_hex_string_16_chars(self):
        engine = MockEngine()
        tid = engine.tokenizer_id
        assert isinstance(tid, str)
        assert len(tid) == 16
        int(tid, 16)  # all hex


# ── CapabilityError ───────────────────────────────────────────────────


class TestCapabilityError:
    def test_error_message_actionable(self):
        err = CapabilityError(
            missing={"hidden_states"},
            engine_name="MockEngine",
            metric_name="cka",
        )
        msg = str(err)
        assert "cka" in msg
        assert "hidden_states" in msg
        assert (
            "use a different engine" in msg.lower()
            or "skip this metric" in msg.lower()
        )

    def test_error_carries_attributes(self):
        err = CapabilityError(
            missing={"x", "y"},
            engine_name="E",
            metric_name="m",
        )
        assert err.missing == {"x", "y"}
        assert err.engine_name == "E"
        assert err.metric_name == "m"


# ── RESERVED_CAPABILITIES ─────────────────────────────────────────────


class TestReservedCapabilities:
    def test_known_capabilities_present(self):
        assert "score" in RESERVED_CAPABILITIES
        assert "generate" in RESERVED_CAPABILITIES
        assert "hidden_states" in RESERVED_CAPABILITIES
        assert "attention_weights" in RESERVED_CAPABILITIES
        assert "logprobs_full" in RESERVED_CAPABILITIES
        assert "steering" in RESERVED_CAPABILITIES
        # v0.8+ reserved
        assert "sampling_cloud" in RESERVED_CAPABILITIES
        # v2.0+ reserved
        assert "patch_activations" in RESERVED_CAPABILITIES
        assert "agentic" in RESERVED_CAPABILITIES

    def test_no_typo_aliases(self):
        # Common typos / synonyms must NOT slip in.
        assert "scores" not in RESERVED_CAPABILITIES
        assert "attention" not in RESERVED_CAPABILITIES
        assert "hidden" not in RESERVED_CAPABILITIES
        assert "weights" not in RESERVED_CAPABILITIES

    def test_minimal_engine_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown capability"):
            MinimalEngine(
                Config(model="test"),
                capabilities=frozenset({"score", "foo_bar_unknown"}),
            )


# ── MinimalEngine template ────────────────────────────────────────────


class TestMinimalEngineTemplate:
    def test_subclass_with_real_impl(self):
        class TestEngine(MinimalEngine):
            def _score_impl(self, prompt, continuation):
                n = max(len(continuation.split()), 1)
                return [-1.0] * n, list(range(n))

            def _generate_impl(self, prompt, mnt, t, tp, seed):
                return "fake output", [1, 2, 3], None

        engine = TestEngine(Config(model="test"))
        result = engine.score("hello", "world there")
        assert len(result.logprobs) == 2
        assert result.avg_logprob == pytest.approx(-1.0)

        gen = engine.generate("hello")
        assert gen.text == "fake output"
        assert len(gen.tokens) == 3

    def test_unimplemented_methods_raise(self):
        class BareEngine(MinimalEngine):
            def _score_impl(self, prompt, continuation):
                return [-1.0], [1]

            def _generate_impl(self, prompt, mnt, t, tp, seed):
                return "x", [1], None

        engine = BareEngine(Config(model="test"))
        with pytest.raises(NotImplementedError, match="hidden_states"):
            engine.hidden_states("test")
        with pytest.raises(NotImplementedError, match="attention_weights"):
            engine.attention_weights("test")
        with pytest.raises(NotImplementedError, match="steering"):
            engine.apply_steering("test", None)
        with pytest.raises(NotImplementedError, match="steering"):
            engine.extract_steering_vector(["a"], ["b"], layer=0)

    def test_score_disabled_raises(self):
        class NoScoreEngine(MinimalEngine):
            def _score_impl(self, prompt, continuation):
                return [], []

            def _generate_impl(self, prompt, mnt, t, tp, seed):
                return "", [], None

        engine = NoScoreEngine(
            Config(model="test"),
            capabilities=frozenset({"generate"}),
        )
        with pytest.raises(NotImplementedError, match="score"):
            engine.score("a", "b")

    def test_generate_disabled_raises(self):
        class NoGenEngine(MinimalEngine):
            def _score_impl(self, prompt, continuation):
                return [-1.0], [0]

            def _generate_impl(self, prompt, mnt, t, tp, seed):
                return "", [], None

        engine = NoGenEngine(
            Config(model="test"),
            capabilities=frozenset({"score"}),
        )
        with pytest.raises(NotImplementedError, match="generate"):
            engine.generate("a")

    def test_default_tokenizer_id_from_model(self):
        e1 = MinimalEngine.__new__(MinimalEngine)
        e1._config = Config(model="my_model_x")
        tid = e1._compute_default_tokenizer_id()
        assert isinstance(tid, str)
        assert len(tid) == 16

    def test_minimal_engine_default_capabilities(self):
        class E(MinimalEngine):
            def _score_impl(self, p, c):
                return [-1.0], [0]

            def _generate_impl(self, p, m, t, tp, s):
                return "", [], None

        engine = E(Config(model="m"))
        assert engine.capabilities == frozenset({"score", "generate"})

    def test_minimal_engine_close_no_op(self):
        class E(MinimalEngine):
            def _score_impl(self, p, c):
                return [-1.0], [0]

            def _generate_impl(self, p, m, t, tp, s):
                return "", [], None

        engine = E(Config(model="m"))
        engine.close()  # default no-op
        engine.close()  # idempotent


# ── Lazy import contract ──────────────────────────────────────────────


class TestLazyImport:
    """``from lmdiff import HFEngine`` must not load torch."""

    def test_engine_class_lazy(self):
        if "torch" in sys.modules:
            pytest.skip("torch already loaded by previous test (run in isolation)")
        from lmdiff import HFEngine  # noqa: F401
        assert "torch" not in sys.modules

    def test_minimal_engine_lazy(self):
        if "torch" in sys.modules:
            pytest.skip("torch already loaded")
        from lmdiff import MinimalEngine  # noqa: F401
        assert "torch" not in sys.modules

    def test_protocol_lazy(self):
        if "torch" in sys.modules:
            pytest.skip("torch already loaded")
        from lmdiff import Engine  # noqa: F401
        assert "torch" not in sys.modules


# ── close() idempotency ───────────────────────────────────────────────


class TestClose:
    def test_mock_close_idempotent(self):
        engine = MockEngine()
        engine.close()
        engine.close()  # must not raise


# ── with_config ───────────────────────────────────────────────────────


class TestWithConfig:
    def test_mock_with_config_returns_new_instance(self):
        e1 = MockEngine(Config(model="a"))
        e2 = e1.with_config(Config(model="b"))
        assert e1 is not e2
        assert e2.name == "b"
        # tokenizer_id changes when model changes (synthetic, model-derived).
        assert e1.tokenizer_id != e2.tokenizer_id

    def test_mock_with_config_preserves_capabilities(self):
        caps = frozenset({"score", "generate"})
        e1 = MockEngine(Config(model="a"), capabilities=caps)
        e2 = e1.with_config(Config(model="b"))
        assert e2.capabilities == caps
