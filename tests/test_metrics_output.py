import math
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from modeldiff.metrics.base import MetricLevel, MetricResult
from modeldiff.metrics.output.token_entropy import TokenEntropy, _entropy_from_logits
from modeldiff.metrics.output.token_kl import TokenKL, _kl_divergence


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_engine_stub(logits_list: list[torch.Tensor]) -> MagicMock:
    """Create a mock engine whose get_logits returns the given tensors."""
    engine = MagicMock()
    result = MagicMock()
    result.logits = logits_list
    engine.get_logits.return_value = result
    return engine


def _uniform_logits(seq_len: int, vocab: int) -> torch.Tensor:
    return torch.zeros(seq_len, vocab)


def _peaked_logits(seq_len: int, vocab: int, peak_idx: int = 0, peak_val: float = 10.0) -> torch.Tensor:
    logits = torch.zeros(seq_len, vocab)
    logits[:, peak_idx] = peak_val
    return logits


# ── _entropy_from_logits ────────────────────────────────────────────────

class TestEntropyFromLogits:
    def test_uniform_distribution(self):
        vocab = 100
        logits = _uniform_logits(3, vocab)
        ent = _entropy_from_logits(logits)
        expected = math.log(vocab)
        np.testing.assert_allclose(ent, expected, atol=1e-5)

    def test_peaked_distribution_low_entropy(self):
        logits = _peaked_logits(3, 100, peak_val=50.0)
        ent = _entropy_from_logits(logits)
        assert np.all(ent < 0.1)

    def test_shape(self):
        logits = torch.randn(5, 200)
        ent = _entropy_from_logits(logits)
        assert ent.shape == (5,)

    def test_non_negative(self):
        logits = torch.randn(10, 50)
        ent = _entropy_from_logits(logits)
        assert np.all(ent >= -1e-6)


# ── _kl_divergence ──────────────────────────────────────────────────────

class TestKLDivergence:
    def test_same_distribution_zero_kl(self):
        logits = torch.randn(4, 50)
        kl = _kl_divergence(logits, logits)
        np.testing.assert_allclose(kl, 0.0, atol=1e-5)

    def test_kl_non_negative(self):
        logits_p = torch.randn(4, 50)
        logits_q = torch.randn(4, 50)
        kl = _kl_divergence(logits_p, logits_q)
        assert np.all(kl >= -1e-5)

    def test_kl_asymmetric(self):
        logits_p = _peaked_logits(3, 50, peak_val=10.0)
        logits_q = _uniform_logits(3, 50)
        kl_pq = _kl_divergence(logits_p, logits_q)
        kl_qp = _kl_divergence(logits_q, logits_p)
        assert not np.allclose(kl_pq, kl_qp, atol=0.1)

    def test_peaked_vs_uniform_known_value(self):
        vocab = 4
        logits_p = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        logits_q = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

        p = torch.softmax(logits_p, dim=-1)
        q = torch.softmax(logits_q, dim=-1)
        expected_kl = (p * (p.log() - q.log())).sum(dim=-1).numpy()

        kl = _kl_divergence(logits_p, logits_q)
        np.testing.assert_allclose(kl, expected_kl, atol=1e-5)

    def test_shape(self):
        kl = _kl_divergence(torch.randn(7, 30), torch.randn(7, 30))
        assert kl.shape == (7,)


# ── TokenEntropy metric ─────────────────────────────────────────────────

class TestTokenEntropyMetric:
    def test_returns_metric_result(self):
        engine_a = _make_engine_stub([_uniform_logits(3, 50)])
        engine_b = _make_engine_stub([_peaked_logits(3, 50)])
        metric = TokenEntropy()
        result = metric.compute(engine_a, engine_b, ["test prompt"])
        assert isinstance(result, MetricResult)
        assert result.name == "token_entropy"
        assert result.level == MetricLevel.OUTPUT

    def test_uniform_vs_peaked(self):
        engine_a = _make_engine_stub([_uniform_logits(3, 50)])
        engine_b = _make_engine_stub([_peaked_logits(3, 50, peak_val=20.0)])
        result = TokenEntropy().compute(engine_a, engine_b, ["probe"])
        assert result.value < 0, "peaked B should have lower entropy than uniform A"
        assert result.details["mean_entropy_a"] > result.details["mean_entropy_b"]

    def test_same_distribution_zero_delta(self):
        logits = torch.randn(4, 100)
        engine_a = _make_engine_stub([logits])
        engine_b = _make_engine_stub([logits])
        result = TokenEntropy().compute(engine_a, engine_b, ["probe"])
        assert abs(result.value) < 1e-5

    def test_per_prompt_details(self):
        engine_a = _make_engine_stub([_uniform_logits(2, 30), _peaked_logits(2, 30)])
        engine_b = _make_engine_stub([_peaked_logits(2, 30), _uniform_logits(2, 30)])
        result = TokenEntropy().compute(engine_a, engine_b, ["p1", "p2"])
        assert len(result.details["per_prompt"]) == 2
        assert result.details["per_prompt"][0]["delta"] < 0
        assert result.details["per_prompt"][1]["delta"] > 0

    def test_numerical_correctness(self):
        vocab = 10
        logits_a = _uniform_logits(1, vocab)
        logits_b = _peaked_logits(1, vocab, peak_val=5.0)

        expected_a = math.log(vocab)
        p_b = torch.softmax(logits_b, dim=-1)
        expected_b = -(p_b * p_b.log()).sum(dim=-1).item()

        engine_a = _make_engine_stub([logits_a])
        engine_b = _make_engine_stub([logits_b])
        result = TokenEntropy().compute(engine_a, engine_b, ["probe"])

        assert abs(result.details["mean_entropy_a"] - expected_a) < 1e-5
        assert abs(result.details["mean_entropy_b"] - expected_b) < 1e-5
        assert abs(result.value - (expected_b - expected_a)) < 1e-5

    def test_requirements(self):
        assert TokenEntropy.requirements()["logits"] is True

    def test_is_applicable_default_true(self):
        assert TokenEntropy.is_applicable(None, None)


# ── TokenKL metric ──────────────────────────────────────────────────────

class TestTokenKLMetric:
    def test_returns_metric_result(self):
        logits = torch.randn(3, 50)
        engine_a = _make_engine_stub([logits])
        engine_b = _make_engine_stub([logits])
        result = TokenKL().compute(engine_a, engine_b, ["probe"])
        assert isinstance(result, MetricResult)
        assert result.name == "token_kl"
        assert result.level == MetricLevel.OUTPUT

    def test_same_distribution_zero(self):
        logits = torch.randn(4, 100)
        engine_a = _make_engine_stub([logits])
        engine_b = _make_engine_stub([logits])
        result = TokenKL().compute(engine_a, engine_b, ["probe"])
        assert abs(result.value) < 1e-5
        assert abs(result.details["kl_ab"]) < 1e-5
        assert abs(result.details["kl_ba"]) < 1e-5

    def test_different_distributions_positive(self):
        engine_a = _make_engine_stub([_uniform_logits(3, 50)])
        engine_b = _make_engine_stub([_peaked_logits(3, 50, peak_val=10.0)])
        result = TokenKL().compute(engine_a, engine_b, ["probe"])
        assert result.value > 0

    def test_symmetric_value(self):
        engine_a = _make_engine_stub([torch.randn(3, 50)])
        engine_b = _make_engine_stub([torch.randn(3, 50)])
        result = TokenKL().compute(engine_a, engine_b, ["probe"])
        expected_sym = (result.details["kl_ab"] + result.details["kl_ba"]) / 2
        assert abs(result.value - expected_sym) < 1e-6

    def test_per_prompt_details(self):
        engine_a = _make_engine_stub([torch.randn(2, 30), torch.randn(2, 30)])
        engine_b = _make_engine_stub([torch.randn(2, 30), torch.randn(2, 30)])
        result = TokenKL().compute(engine_a, engine_b, ["p1", "p2"])
        assert len(result.details["per_prompt"]) == 2
        for entry in result.details["per_prompt"]:
            assert "kl_ab" in entry
            assert "kl_ba" in entry
            assert "symmetric" in entry

    def test_numerical_correctness(self):
        vocab = 4
        logits_a = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
        logits_b = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

        p_a = torch.softmax(logits_a, dim=-1)
        p_b = torch.softmax(logits_b, dim=-1)
        expected_kl_ab = (p_a * (p_a.log() - p_b.log())).sum(dim=-1).item()
        expected_kl_ba = (p_b * (p_b.log() - p_a.log())).sum(dim=-1).item()

        engine_a = _make_engine_stub([logits_a])
        engine_b = _make_engine_stub([logits_b])
        result = TokenKL().compute(engine_a, engine_b, ["probe"])

        assert abs(result.details["kl_ab"] - expected_kl_ab) < 1e-5
        assert abs(result.details["kl_ba"] - expected_kl_ba) < 1e-5
        assert abs(result.value - (expected_kl_ab + expected_kl_ba) / 2) < 1e-5

    def test_requirements(self):
        assert TokenKL.requirements()["logits"] is True
