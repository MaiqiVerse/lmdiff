"""Integration tests for HFEngine — requires GPU and downloads gpt2.

Run with::

    pytest -m gpu tests/integration/test_hf_engine.py

These tests are skipped by default (CI fast suite excludes ``-m gpu``).
Run manually before merging Phase 1 to verify HFEngine works end-to-end
on a real model.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu

from lmdiff import Config, HFEngine  # noqa: E402


@pytest.fixture(scope="module")
def gpt2_engine():
    """Load gpt2 once for all integration tests in this module."""
    engine = HFEngine(Config(model="gpt2"), device="cpu", dtype="fp32")
    yield engine
    engine.close()


class TestHFEngineLoading:
    def test_loads_gpt2(self, gpt2_engine):
        assert gpt2_engine.name == "gpt2"
        assert gpt2_engine.n_layers > 0
        assert gpt2_engine.hidden_dim > 0

    def test_tokenizer_id_stable_across_instances(self, gpt2_engine):
        e2 = HFEngine(Config(model="gpt2"), device="cpu", dtype="fp32")
        try:
            assert e2.tokenizer_id == gpt2_engine.tokenizer_id
        finally:
            e2.close()

    def test_capabilities_default_set(self, gpt2_engine):
        caps = gpt2_engine.capabilities
        assert "score" in caps
        assert "generate" in caps
        assert "hidden_states" in caps
        assert "attention_weights" in caps
        # Reserved-but-not-implemented in v0.3.0:
        assert "steering" not in caps


class TestHFEngineScore:
    def test_score_basic(self, gpt2_engine):
        result = gpt2_engine.score("The capital of France is", " Paris")
        assert len(result.tokens) > 0
        assert len(result.logprobs) == len(result.tokens)
        assert isinstance(result.avg_logprob, float)
        assert -20.0 < result.avg_logprob < 0.0

    def test_score_known_continuation_more_likely(self, gpt2_engine):
        """Real continuation should out-score nonsense (small models can be
        flaky here; only assert the inequality with a tolerance)."""
        good = gpt2_engine.score("The capital of France is", " Paris")
        bad = gpt2_engine.score("The capital of France is", " banana")
        assert good.avg_logprob > bad.avg_logprob

    def test_score_empty_continuation(self, gpt2_engine):
        result = gpt2_engine.score("The capital of France is", "")
        assert len(result.tokens) == 0
        assert result.avg_logprob == 0.0


class TestHFEngineGenerate:
    def test_generate_basic(self, gpt2_engine):
        r = gpt2_engine.generate("Hello, world", max_new_tokens=5)
        assert isinstance(r.text, str)
        assert 0 < len(r.tokens) <= 5

    def test_generate_seed_reproducible(self, gpt2_engine):
        r1 = gpt2_engine.generate(
            "Hello", max_new_tokens=10, temperature=0.7, top_p=0.9, seed=42,
        )
        r2 = gpt2_engine.generate(
            "Hello", max_new_tokens=10, temperature=0.7, top_p=0.9, seed=42,
        )
        assert r1.text == r2.text

    def test_generate_logprobs_populated(self, gpt2_engine):
        r = gpt2_engine.generate("Hello", max_new_tokens=5)
        assert r.logprobs is not None
        assert len(r.logprobs) > 0


class TestHFEngineHiddenStates:
    def test_hidden_states_default_shape(self, gpt2_engine):
        r = gpt2_engine.hidden_states("The quick brown fox")
        assert r.hidden_states.shape == (gpt2_engine.n_layers, gpt2_engine.hidden_dim)

    def test_hidden_states_layer_subset(self, gpt2_engine):
        r = gpt2_engine.hidden_states("hello", layers=[1, 5, 10])
        assert r.hidden_states.shape == (3, gpt2_engine.hidden_dim)

    def test_hidden_states_first_position(self, gpt2_engine):
        r = gpt2_engine.hidden_states("hello", position="first")
        assert r.position == "first"


class TestHFEngineAttention:
    def test_attention_weights_shape(self, gpt2_engine):
        r = gpt2_engine.attention_weights("Hello world")
        # gpt2: 12 layers, 12 heads
        assert r.attention_weights.shape[0] == gpt2_engine.n_layers
        assert r.attention_weights.shape[1] == 12


class TestHFEngineClose:
    def test_close_idempotent(self):
        engine = HFEngine(Config(model="gpt2"), device="cpu", dtype="fp32")
        engine.close()
        engine.close()  # must not raise
