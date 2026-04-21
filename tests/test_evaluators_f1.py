"""Tests for the F1 evaluator (SQuAD-style)."""
from __future__ import annotations

import pytest

from lmdiff.tasks.evaluators import F1


class TestF1Basic:
    def test_exact_match(self):
        correct, score, extra = F1().evaluate("Paris", "Paris")
        assert correct is True
        assert score == pytest.approx(1.0)
        assert extra["f1"] == pytest.approx(1.0)
        assert extra["best_target"] == "Paris"

    def test_complete_mismatch(self):
        # "red apple" vs "blue sky dome": no overlapping tokens → f1 = 0
        correct, score, extra = F1().evaluate("red apple", "blue sky dome")
        assert correct is False
        assert score == 0.0
        assert extra["f1"] == 0.0

    def test_partial_overlap_numerical(self):
        # pred = {"big", "cat", "sat"} (article "the" stripped), len=3
        # gold = {"cat"} (article "the" stripped), len=1
        # overlap = 1. precision = 1/3, recall = 1/1, f1 = 2*(1/3*1) / (1/3+1) = 0.5
        correct, score, extra = F1().evaluate("the big cat sat", "the cat")
        assert score == pytest.approx(0.5, abs=1e-9)
        # At exact threshold (0.5) we treat it as correct (>=).
        assert correct is True

    def test_partial_overlap_above_threshold(self):
        # pred = {"paris", "france"}, gold = {"paris"}
        # overlap = 1. p = 1/2, r = 1, f1 = 2*(0.5*1)/(1.5) = 2/3 ≈ 0.667
        correct, score, _ = F1().evaluate("Paris, France", "Paris")
        assert score == pytest.approx(2 / 3, abs=1e-9)
        assert correct is True

    def test_partial_overlap_below_threshold(self):
        # pred = {"the", "capital"} → articles drop → {"capital"}, len=1
        # gold = {"paris", "is", "capital", "of", "france"}, len=5
        # overlap 1 → p=1/1, r=1/5, f1 = 2*(1*0.2)/(1.2) = 1/3
        correct, score, _ = F1().evaluate("the capital", "paris is capital of france")
        assert score == pytest.approx(1 / 3, abs=1e-9)
        assert correct is False


class TestF1Normalization:
    def test_articles_dropped(self):
        correct, score, _ = F1().evaluate("the cat", "cat")
        assert score == pytest.approx(1.0)
        assert correct is True

    def test_punctuation_dropped(self):
        correct, score, _ = F1().evaluate("Paris.", "paris")
        assert score == pytest.approx(1.0)
        assert correct is True

    def test_case_insensitive(self):
        correct, score, _ = F1().evaluate("PARIS", "paris")
        assert score == pytest.approx(1.0)
        assert correct is True

    def test_whitespace_collapse(self):
        correct, score, _ = F1().evaluate("  Paris   ", "Paris")
        assert score == pytest.approx(1.0)
        assert correct is True


class TestF1MultiTarget:
    def test_aliases_pick_best(self):
        # expected='London' → f1=0; aliases includes 'Paris' → f1=1
        correct, score, extra = F1().evaluate(
            "Paris", "London",
            probe_metadata={"aliases": ["Paris", "paris"]},
        )
        assert score == pytest.approx(1.0)
        assert correct is True
        assert extra["best_target"] == "Paris"
        assert extra["n_targets"] == 3  # 1 primary + 2 aliases

    def test_no_aliases_field(self):
        correct, score, extra = F1().evaluate("Paris", "Paris", probe_metadata=None)
        assert score == pytest.approx(1.0)
        assert extra["n_targets"] == 1

    def test_aliases_none_ignored(self):
        # probe_metadata has aliases key but value is None — should be ignored
        correct, score, extra = F1().evaluate(
            "Paris", "Paris",
            probe_metadata={"aliases": None},
        )
        assert extra["n_targets"] == 1


class TestF1EdgeCases:
    def test_empty_pred_nonempty_gold(self):
        correct, score, _ = F1().evaluate("", "Paris")
        assert score == 0.0
        assert correct is False

    def test_empty_pred_empty_gold(self):
        correct, score, _ = F1().evaluate("", "")
        assert score == pytest.approx(1.0)
        assert correct is True

    def test_expected_none_returns_reason(self):
        correct, score, extra = F1().evaluate("anything", None)
        assert correct is False
        assert score == 0.0
        assert extra["reason"] == "no_expected"

    def test_only_articles_in_both(self):
        # After normalize, both become empty → f1=1.0
        correct, score, _ = F1().evaluate("the a an", "an the")
        assert score == pytest.approx(1.0)
        assert correct is True


class TestF1Contract:
    def test_name_attribute(self):
        assert F1.name == "f1"

    def test_returns_three_tuple(self):
        r = F1().evaluate("cat", "cat")
        assert isinstance(r, tuple)
        assert len(r) == 3
        assert isinstance(r[0], bool)
        assert isinstance(r[1], float)
        assert isinstance(r[2], dict)
