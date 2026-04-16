import inspect
from unittest.mock import MagicMock

import pytest

from lmdiff.probes.loader import Probe, ProbeSet
from lmdiff.tasks.base import EvalResult, Task, TaskResult
from lmdiff.tasks.evaluators import ContainsAnswer, ExactMatch, MultipleChoice


# ── ExactMatch ───────────────────────────────────────────────────────────

class TestExactMatch:
    def test_identical(self):
        ok, score, meta = ExactMatch().evaluate("42", "42")
        assert ok is True
        assert score == 1.0

    def test_case_sensitive_default(self):
        ok, score, _ = ExactMatch().evaluate("Paris", "paris")
        assert ok is False

    def test_case_insensitive(self):
        ok, score, _ = ExactMatch(case_sensitive=False).evaluate("Paris", "paris")
        assert ok is True

    def test_strip_whitespace(self):
        ok, score, _ = ExactMatch(strip=True).evaluate("  42  ", "42")
        assert ok is True

    def test_no_strip(self):
        ok, score, _ = ExactMatch(strip=False).evaluate("  42  ", "42")
        assert ok is False

    def test_expected_none(self):
        ok, score, meta = ExactMatch().evaluate("anything", None)
        assert ok is False
        assert meta["reason"] == "no_expected"


# ── ContainsAnswer ───────────────────────────────────────────────────────

class TestContainsAnswer:
    def test_contains(self):
        ok, score, meta = ContainsAnswer().evaluate("The answer is 42.", "42")
        assert ok is True
        assert meta["position"] >= 0

    def test_not_contains(self):
        ok, score, meta = ContainsAnswer().evaluate("completely unrelated", "42")
        assert ok is False
        assert meta["position"] == -1

    def test_case_insensitive_default(self):
        ok, _, _ = ContainsAnswer().evaluate("PARIS is the capital", "paris")
        assert ok is True

    def test_case_sensitive(self):
        ok, _, _ = ContainsAnswer(case_sensitive=True).evaluate("PARIS", "paris")
        assert ok is False

    def test_expected_none(self):
        ok, score, meta = ContainsAnswer().evaluate("anything", None)
        assert ok is False
        assert meta["reason"] == "no_expected"


# ── MultipleChoice ───────────────────────────────────────────────────────

class TestMultipleChoice:
    def test_letter_match(self):
        ok, score, meta = MultipleChoice().evaluate(
            "A", None, {"correct_index": 0},
        )
        assert ok is True
        assert meta["predicted_index"] == 0

    def test_letter_in_sentence(self):
        ok, _, meta = MultipleChoice().evaluate(
            "The answer is B.", None, {"correct_index": 1},
        )
        assert ok is True
        assert meta["predicted_index"] == 1

    def test_integer_fallback(self):
        ok, _, meta = MultipleChoice().evaluate(
            "3", None, {"correct_index": 3},
        )
        assert ok is True
        assert meta["predicted_index"] == 3

    def test_no_parse(self):
        ok, _, meta = MultipleChoice().evaluate(
            "???", None, {"correct_index": 0},
        )
        assert ok is False
        assert meta["reason"] == "no_choice_parsed"

    def test_missing_metadata(self):
        ok, _, meta = MultipleChoice().evaluate("A", None, None)
        assert ok is False
        assert meta["reason"] == "missing_mc_metadata"

    def test_wrong_answer(self):
        ok, _, meta = MultipleChoice().evaluate(
            "A", None, {"correct_index": 2},
        )
        assert ok is False
        assert meta["predicted_index"] == 0


# ── Task.run ─────────────────────────────────────────────────────────────

def _make_mock_engine(completions: list[str], name: str = "mock") -> MagicMock:
    engine = MagicMock()
    engine.model_name = name
    gen = MagicMock()
    gen.completions = [[c] for c in completions]
    engine.generate.return_value = gen
    return engine


class TestTaskRun:
    def test_basic(self):
        probes = ProbeSet([
            Probe(id="m1", text="1+1=", domain="math", expected="2"),
            Probe(id="m2", text="2+2=", domain="math", expected="4"),
            Probe(id="k1", text="Capital of France is", domain="knowledge", expected="Paris"),
        ])
        engine = _make_mock_engine(["2", "5", "Paris is the capital"])
        task = Task("test-task", probes, ExactMatch(), max_new_tokens=16)
        result = task.run(engine)

        assert isinstance(result, TaskResult)
        assert result.n_probes == 3
        assert result.n_correct == 1  # only "2" exact matches
        assert abs(result.accuracy - 1 / 3) < 1e-6
        assert len(result.per_probe) == 3
        assert result.per_probe[0].correct is True
        assert result.per_probe[1].correct is False
        assert result.per_probe[2].correct is False  # "Paris is the capital" != "Paris"

        assert "math" in result.per_domain
        assert result.per_domain["math"]["n"] == 2
        assert result.per_domain["math"]["correct"] == 1
        assert "knowledge" in result.per_domain

    def test_contains_answer_more_forgiving(self):
        probes = ProbeSet([
            Probe(id="k1", text="Capital of France is", domain="knowledge", expected="Paris"),
        ])
        engine = _make_mock_engine(["Paris is the capital"])
        task = Task("test", probes, ContainsAnswer(), max_new_tokens=16)
        result = task.run(engine)
        assert result.n_correct == 1

    def test_empty_output(self):
        probes = ProbeSet([
            Probe(id="p1", text="test", expected="answer"),
        ])
        engine = _make_mock_engine([""])
        task = Task("test", probes, ExactMatch(), max_new_tokens=16)
        result = task.run(engine)

        assert result.n_correct == 0
        assert result.per_probe[0].correct is False
        assert result.per_probe[0].metadata.get("empty_output") is True

    def test_whitespace_only_output(self):
        probes = ProbeSet([
            Probe(id="p1", text="test", expected="answer"),
        ])
        engine = _make_mock_engine(["   \n\n  "])
        task = Task("test", probes, ExactMatch(), max_new_tokens=16)
        result = task.run(engine)
        assert result.per_probe[0].metadata.get("empty_output") is True

    def test_expected_none(self):
        probes = ProbeSet([
            Probe(id="p1", text="def foo():", expected=None),
        ])
        engine = _make_mock_engine(["pass"])
        task = Task("test", probes, ExactMatch(), max_new_tokens=16)
        result = task.run(engine)
        assert result.per_probe[0].correct is False
        assert result.per_probe[0].metadata.get("reason") == "no_expected"

    def test_get_by_probe_id(self):
        probes = ProbeSet([
            Probe(id="a", text="x", expected="y"),
            Probe(id="b", text="z", expected="w"),
        ])
        engine = _make_mock_engine(["y", "wrong"])
        task = Task("test", probes, ExactMatch(), max_new_tokens=16)
        result = task.run(engine)
        assert result.get("a").correct is True
        assert result.get("b").correct is False
        assert result.get("nonexistent") is None

    def test_engine_name_in_result(self):
        probes = ProbeSet([Probe(id="p", text="t", expected="t")])
        engine = _make_mock_engine(["t"], name="gpt2")
        task = Task("test", probes, ExactMatch(), max_new_tokens=16)
        result = task.run(engine)
        assert result.engine_name == "gpt2"


# ── Architecture ─────────────────────────────────────────────────────────

class TestTaskArchitecture:
    def test_base_no_transformers(self):
        import lmdiff.tasks.base as mod
        src = inspect.getsource(mod)
        assert "import transformers" not in src

    def test_base_no_metrics(self):
        import lmdiff.tasks.base as mod
        src = inspect.getsource(mod)
        assert "from lmdiff.metrics" not in src

    def test_evaluators_no_transformers(self):
        import lmdiff.tasks.evaluators as mod
        src = inspect.getsource(mod)
        assert "import transformers" not in src

    def test_evaluators_no_metrics(self):
        import lmdiff.tasks.evaluators as mod
        src = inspect.getsource(mod)
        assert "from lmdiff.metrics" not in src
