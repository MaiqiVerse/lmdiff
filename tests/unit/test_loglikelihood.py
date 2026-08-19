"""Tests for loglikelihood_accuracy — fully mocked engine."""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from lmdiff.probes.loader import Probe, ProbeSet
from lmdiff.tasks.loglikelihood import loglikelihood_accuracy


def _make_probe(id_, text, choices, correct_index, domain="test"):
    return Probe(
        id=id_, text=text, domain=domain,
        expected=choices[correct_index],
        metadata={"choices": choices, "correct_index": correct_index},
    )


def _mock_engine_with_scores(all_ces: list[list[float]]):
    """Mock engine whose score() returns per-call CE lists from all_ces in order."""
    engine = MagicMock()
    engine.model_name = "mock"

    iter_ces = iter(all_ces)

    def score_side(prompts, continuations=None, continuation_ids=None, **kwargs):
        sr = MagicMock()
        ces = next(iter_ces)
        assert len(ces) == len(prompts), (
            f"test setup mismatch: got {len(prompts)} prompts but CE list is {len(ces)}"
        )
        sr.cross_entropies = ces
        sr.token_ids = [[0, 1]] * len(prompts)
        return sr

    engine.score.side_effect = score_side
    return engine


class TestBasic:
    def test_picks_lowest_ce(self):
        probes = ProbeSet([
            _make_probe("p1", "Q?", ["A", "Buh", "Cuh"], 1),
        ])
        # Normalize by bytes: A=1, Buh=3, Cuh=3. Raw CE = [5, 3, 4].
        # Normalized = [5.0, 1.0, 1.333]. argmin = 1 (Buh). Correct.
        engine = _mock_engine_with_scores([[5.0, 3.0, 4.0]])
        result = loglikelihood_accuracy(probes, engine, normalize=True)
        assert result.n_correct == 1
        assert result.per_probe[0].metadata["predicted_index"] == 1

    def test_wrong_answer(self):
        probes = ProbeSet([
            _make_probe("p1", "Q?", ["A", "B", "C"], 0),
        ])
        # All choices same byte length; raw CE = [5, 1, 3]. argmin = 1, not 0.
        engine = _mock_engine_with_scores([[5.0, 1.0, 3.0]])
        result = loglikelihood_accuracy(probes, engine, normalize=True)
        assert result.n_correct == 0
        assert result.per_probe[0].correct is False

    def test_normalize_changes_winner(self):
        # Long correct answer has higher raw CE but lower per-byte CE.
        probes = ProbeSet([
            _make_probe("p1", "Q?", ["x", "longer_correct"], 1),
        ])
        # Raw: [1.0, 3.0] → argmin=0 (wrong, gold=1).
        # Normalized by bytes (1 vs 14): [1.0, 0.214] → argmin=1 (correct).
        engine_raw = _mock_engine_with_scores([[1.0, 3.0]])
        r_raw = loglikelihood_accuracy(probes, engine_raw, normalize=False)
        assert r_raw.n_correct == 0

        engine_norm = _mock_engine_with_scores([[1.0, 3.0]])
        r_norm = loglikelihood_accuracy(probes, engine_norm, normalize=True)
        assert r_norm.n_correct == 1

    def test_per_domain_aggregation(self):
        probes = ProbeSet([
            _make_probe("m1", "Q?", ["A", "B"], 0, domain="math"),
            _make_probe("m2", "Q?", ["A", "B"], 1, domain="math"),
            _make_probe("c1", "Q?", ["A", "B"], 0, domain="code"),
        ])
        # All choices len 1 byte, so normalize == no-op.
        engine = _mock_engine_with_scores([
            [1.0, 2.0],  # m1 picks 0 (correct)
            [1.0, 2.0],  # m2 picks 0 (wrong: gold=1)
            [1.0, 2.0],  # c1 picks 0 (correct)
        ])
        result = loglikelihood_accuracy(probes, engine)
        assert result.per_domain["math"]["correct"] == 1
        assert result.per_domain["math"]["n"] == 2
        assert result.per_domain["code"]["correct"] == 1
        assert result.per_domain["code"]["n"] == 1


class TestValidation:
    def test_missing_choices_raises(self):
        probe = Probe(
            id="p", text="Q?", domain="x", expected="A",
            metadata={"correct_index": 0},  # no choices
        )
        engine = _mock_engine_with_scores([])
        with pytest.raises(ValueError, match="metadata"):
            loglikelihood_accuracy(ProbeSet([probe]), engine)

    def test_missing_correct_index_raises(self):
        probe = Probe(
            id="p", text="Q?", domain="x", expected="A",
            metadata={"choices": ["A", "B"]},  # no correct_index
        )
        engine = _mock_engine_with_scores([])
        with pytest.raises(ValueError, match="metadata"):
            loglikelihood_accuracy(ProbeSet([probe]), engine)

    def test_index_out_of_range_raises(self):
        probe = Probe(
            id="p", text="Q?", domain="x", expected="A",
            metadata={"choices": ["A", "B"], "correct_index": 5},
        )
        engine = _mock_engine_with_scores([])
        with pytest.raises(ValueError, match="out of range"):
            loglikelihood_accuracy(ProbeSet([probe]), engine)


class TestNaNHandling:
    def test_nan_ce_treated_as_inf(self):
        probes = ProbeSet([_make_probe("p", "Q?", ["A", "B", "C"], 2)])
        # NaN on choice 2; others 1.0 and 2.0. NaN → inf → not picked.
        # argmin = 0 (A). Gold = 2 (C). Wrong.
        engine = _mock_engine_with_scores([[1.0, 2.0, math.nan]])
        result = loglikelihood_accuracy(probes, engine)
        assert result.per_probe[0].metadata["predicted_index"] == 0
        assert result.n_correct == 0


class TestTopLevelExport:
    def test_import_from_lmdiff(self):
        from lmdiff import loglikelihood_accuracy as imported
        from lmdiff.tasks.loglikelihood import loglikelihood_accuracy as direct
        assert imported is direct
