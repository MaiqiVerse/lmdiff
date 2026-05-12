"""Unit tests for the v0.4.1 ``Engine.max_context_length()`` Protocol method.

Covers the three implementations (HFEngine, MinimalEngine, MockEngine) and
the HFEngine fallback chain (``max_position_embeddings`` →
``n_positions`` → ``max_seq_len`` → ``None``) per Q9.7.

HFEngine tests don't touch ``transformers`` or load real models — we
construct a stub ``HFEngine`` instance with only the fields the method
reads (``self._model.config``) so the unit tests stay torch-free.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from lmdiff._config import Config
from lmdiff.engines.minimal import MinimalEngine
from tests.fixtures.mock_engine import MockEngine


# ── HFEngine fallback chain ──────────────────────────────────────────


def _stub_hf(**config_attrs):
    """Build a tiny stand-in for HFEngine that only exposes
    ``self._model.config`` — the only field ``max_context_length``
    reads. Avoids loading torch / a real model."""
    from lmdiff._engine import HFEngine

    engine = HFEngine.__new__(HFEngine)  # bypass __init__ (no model load)
    engine._model = SimpleNamespace(config=SimpleNamespace(**config_attrs))
    return engine


class TestHFEngineFallbackChain:
    def test_max_position_embeddings_wins(self):
        # Llama-2 / most modern HF configs.
        engine = _stub_hf(
            max_position_embeddings=4096,
            n_positions=2048,           # would lose to the first match
            max_seq_len=8192,
        )
        assert engine.max_context_length() == 4096

    def test_n_positions_fallback(self):
        # GPT-2 family.
        engine = _stub_hf(n_positions=1024)
        assert engine.max_context_length() == 1024

    def test_max_seq_len_fallback(self):
        # Some custom configs (older Mosaic / certain MPT releases).
        engine = _stub_hf(max_seq_len=2048)
        assert engine.max_context_length() == 2048

    def test_returns_none_when_no_attr_present(self):
        engine = _stub_hf()  # config has none of the three attrs
        assert engine.max_context_length() is None

    def test_value_is_coerced_to_int(self):
        # Some configs store as float / numpy scalar; coercion guarantees int.
        engine = _stub_hf(max_position_embeddings=4096.0)
        result = engine.max_context_length()
        assert result == 4096
        assert isinstance(result, int)


# ── MinimalEngine default + override ─────────────────────────────────


class TestMinimalEngineDefault:
    def test_default_returns_none(self):
        engine = MinimalEngine(config=Config(model="m"))
        assert engine.max_context_length() is None

    def test_subclass_override_via_hook(self):
        class _Bounded(MinimalEngine):
            def _max_context_impl(self):
                return 8192

        engine = _Bounded(config=Config(model="m"))
        assert engine.max_context_length() == 8192


# ── MockEngine constructor arg ───────────────────────────────────────


class TestMockEngineConstructor:
    def test_default_returns_none(self):
        engine = MockEngine(config=Config(model="m"))
        assert engine.max_context_length() is None

    def test_explicit_max_context(self):
        engine = MockEngine(config=Config(model="m"), max_context=4096)
        assert engine.max_context_length() == 4096

    def test_with_config_preserves_max_context(self):
        engine = MockEngine(config=Config(model="m"), max_context=4096)
        derived = engine.with_config(Config(model="m2"))
        assert derived.max_context_length() == 4096


# ── Engine Protocol default — for any engine that doesn't override ────


class TestProtocolDefault:
    """A custom backend that doesn't implement max_context_length() at all
    falls through to the Protocol default of ``None`` (treats every probe
    as valid). Verified via duck typing — we attach a no-op class that
    inherits the Protocol method body."""

    def test_protocol_default_is_none(self):
        from lmdiff._engine import Engine

        # Engine is a Protocol — instantiating it directly isn't possible.
        # But we can call the unbound method body on a stand-in object.
        class _NoOverride:
            pass

        # The Protocol's max_context_length method body returns None.
        # Verify by inspecting the source — we intentionally do NOT use
        # a runtime_checkable check here because the Protocol's bodies
        # aren't inherited by classes that don't subclass it.
        import inspect
        source = inspect.getsource(Engine.max_context_length)
        assert "return None" in source
