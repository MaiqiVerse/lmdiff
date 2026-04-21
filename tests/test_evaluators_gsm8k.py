"""Tests for Gsm8kNumberMatch evaluator."""
from __future__ import annotations

import pytest

from lmdiff.tasks.evaluators import Gsm8kNumberMatch


class TestGsm8kExtraction:
    def test_hash_answer_format(self):
        correct, score, extra = Gsm8kNumberMatch().evaluate(
            "Some reasoning here.\n#### 42",
            "#### 42",
        )
        assert correct is True
        assert score == 1.0
        assert extra["pred_num"] == 42.0
        assert extra["gold_num"] == 42.0

    def test_fallback_to_last_number(self):
        correct, score, _ = Gsm8kNumberMatch().evaluate(
            "The answer is 42.",
            "#### 42",
        )
        assert correct is True
        assert score == 1.0

    def test_mismatch(self):
        correct, score, _ = Gsm8kNumberMatch().evaluate(
            "#### 42",
            "#### 43",
        )
        assert correct is False
        assert score == 0.0

    def test_thousands_separator(self):
        correct, score, extra = Gsm8kNumberMatch().evaluate(
            "Final: #### 1,234",
            "1234",
        )
        assert correct is True
        assert extra["pred_num"] == 1234.0

    def test_decimal_close_match(self):
        correct, score, _ = Gsm8kNumberMatch().evaluate(
            "#### 3.14",
            "3.140",
        )
        assert correct is True
        assert score == 1.0

    def test_negative_number(self):
        correct, score, extra = Gsm8kNumberMatch().evaluate(
            "#### -5",
            "-5",
        )
        assert correct is True
        assert extra["pred_num"] == -5.0

    def test_no_number_in_pred(self):
        correct, score, extra = Gsm8kNumberMatch().evaluate(
            "I don't know",
            "#### 42",
        )
        assert correct is False
        assert score == 0.0
        assert extra["reason"] == "extraction_failed"
        assert extra["pred_num"] is None
        assert extra["gold_num"] == 42.0

    def test_multiple_hash_markers_take_last(self):
        # Several '####' markers, last one holds the real answer.
        correct, score, extra = Gsm8kNumberMatch().evaluate(
            "step 1: #### 7\nstep 2: #### 42",
            "#### 42",
        )
        assert correct is True
        assert extra["pred_num"] == 42.0

    def test_hash_then_trailing_words(self):
        # '#### 42 hello' → '####' regex captures '42' only.
        correct, score, _ = Gsm8kNumberMatch().evaluate(
            "#### 42 hello world",
            "42",
        )
        assert correct is True
        assert score == 1.0


class TestGsm8kEdgeCases:
    def test_expected_none(self):
        correct, score, extra = Gsm8kNumberMatch().evaluate("#### 42", None)
        assert correct is False
        assert score == 0.0
        assert extra["reason"] == "no_expected"

    def test_empty_strings(self):
        correct, score, _ = Gsm8kNumberMatch().evaluate("", "")
        assert correct is False
        assert score == 0.0

    def test_rel_tol_passes_for_near_equal(self):
        correct, _, _ = Gsm8kNumberMatch().evaluate("#### 1.0000001", "1.0")
        # 1e-7 relative diff → within rel_tol=1e-6
        assert correct is True

    def test_rel_tol_fails_for_far_values(self):
        correct, score, _ = Gsm8kNumberMatch().evaluate("#### 1.001", "1.0")
        assert correct is False
        assert score == 0.0


class TestGsm8kContract:
    def test_name_attribute(self):
        assert Gsm8kNumberMatch.name == "gsm8k_number_match"

    def test_returns_three_tuple(self):
        r = Gsm8kNumberMatch().evaluate("#### 1", "#### 1")
        assert isinstance(r, tuple)
        assert len(r) == 3
        assert isinstance(r[0], bool)
        assert isinstance(r[1], float)
        assert isinstance(r[2], dict)

    def test_top_level_export(self):
        from lmdiff import Gsm8kNumberMatch as TopLevel
        assert TopLevel is Gsm8kNumberMatch
