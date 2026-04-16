from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lmdiff.probes.loader import Probe, ProbeSet
from lmdiff.tasks.base import EvalResult, TaskResult
from lmdiff.tasks.capability_radar import (
    CapabilityRadar,
    DomainRadarResult,
    RadarResult,
)
from lmdiff.tasks.evaluators import ContainsAnswer, ExactMatch
from lmdiff.report.terminal import print_radar

V01_PATH = Path(__file__).parent.parent / "lmdiff" / "probes" / "v01.json"


def _make_probes(domains: dict[str, int]) -> ProbeSet:
    """Create a ProbeSet with specified domain counts."""
    probes = []
    for domain, count in domains.items():
        for i in range(count):
            probes.append(Probe(
                id=f"{domain}_{i:03d}",
                text=f"probe {domain} {i}",
                domain=domain,
                expected=f"answer_{domain}_{i}",
            ))
    return ProbeSet(probes, name="test", version="0.0.1")


def _make_mock_engine(name: str, correct_answers: dict[str, list[str]] | None = None) -> MagicMock:
    """Create a mock engine that returns specified completions per domain."""
    engine = MagicMock()
    engine.model_name = name

    def generate_side_effect(texts, n_samples=1, max_new_tokens=16):
        gen = MagicMock()
        completions = []
        for t in texts:
            if correct_answers:
                # Find domain from text
                found = False
                for domain, answers in correct_answers.items():
                    if domain in t:
                        idx = sum(1 for c in completions)  # rough index
                        ans = answers[idx % len(answers)] if answers else "wrong"
                        completions.append([ans])
                        found = True
                        break
                if not found:
                    completions.append(["wrong"])
            else:
                completions.append(["wrong"])
        gen.completions = completions
        return gen

    engine.generate.side_effect = generate_side_effect
    return engine


# ── Unit tests ──────────────────────────────────────────────────────────


class TestDomainRadarResult:
    def test_frozen(self):
        dr = DomainRadarResult(domain="math", n_probes=10, accuracy=0.8)
        with pytest.raises(AttributeError):
            dr.domain = "code"

    def test_defaults(self):
        dr = DomainRadarResult(domain="math", n_probes=10, accuracy=0.8)
        assert dr.bd_vs_baseline is None


class TestRadarResultSummaryTable:
    def test_single_engine(self):
        result = RadarResult(
            engine_a_name="gpt2",
            engine_b_name=None,
            domains=["code", "math"],
            a_by_domain={
                "code": DomainRadarResult("code", 10, 0.7),
                "math": DomainRadarResult("math", 10, 0.5),
            },
            b_by_domain=None,
            bd_by_domain=None,
            bd_healthy_by_domain=None,
            degeneracy_rates=None,
        )
        rows = result.summary_table()
        assert len(rows) == 2
        assert rows[0]["domain"] == "code"
        assert rows[0]["accuracy_a"] == 0.7
        assert "accuracy_b" not in rows[0]
        assert "bd" not in rows[0]

    def test_pair_engine(self):
        result = RadarResult(
            engine_a_name="gpt2",
            engine_b_name="distilgpt2",
            domains=["code", "math"],
            a_by_domain={
                "code": DomainRadarResult("code", 10, 0.7),
                "math": DomainRadarResult("math", 10, 0.5),
            },
            b_by_domain={
                "code": DomainRadarResult("code", 10, 0.6),
                "math": DomainRadarResult("math", 10, 0.4),
            },
            bd_by_domain={"code": 0.5, "math": 0.3},
            bd_healthy_by_domain={"code": 0.4, "math": None},
            degeneracy_rates={
                "code": {"a": 0.0, "b": 0.2},
                "math": {"a": 0.0, "b": 0.1},
            },
        )
        rows = result.summary_table()
        assert len(rows) == 2
        assert rows[0]["accuracy_b"] == 0.6
        assert rows[0]["delta_acc"] == pytest.approx(-0.1)
        assert rows[0]["bd"] == 0.5
        assert rows[0]["bd_healthy"] == 0.4
        assert rows[1]["bd_healthy"] is None
        assert rows[0]["degen_a"] == 0.0
        assert rows[0]["degen_b"] == 0.2


class TestCapabilityRadarValidation:
    def test_single_domain_raises(self):
        probes = ProbeSet(
            [Probe(id="m1", text="1+1=", domain="math", expected="2")],
            name="test",
        )
        with pytest.raises(ValueError, match="at least 2 domains"):
            CapabilityRadar(probes)

    def test_two_domains_ok(self):
        probes = ProbeSet([
            Probe(id="m1", text="1+1=", domain="math", expected="2"),
            Probe(id="k1", text="Capital of France is ", domain="knowledge", expected="Paris"),
        ])
        radar = CapabilityRadar(probes)
        assert radar.evaluator.name == "contains_answer"


class TestRunSingleMock:
    def test_three_domains(self):
        probes = _make_probes({"math": 5, "knowledge": 5, "code": 5})

        def gen_side_effect(texts, n_samples=1, max_new_tokens=16):
            gen = MagicMock()
            completions = []
            for t in texts:
                # Return correct answer for first 3 in each group
                for domain in ["math", "knowledge", "code"]:
                    if domain in t:
                        idx = int(t.split()[-1])
                        if idx < 3:
                            completions.append([f"answer_{domain}_{idx}"])
                        else:
                            completions.append(["wrong"])
                        break
            gen.completions = completions
            return gen

        engine = MagicMock()
        engine.model_name = "mock_model"
        engine.generate.side_effect = gen_side_effect

        radar = CapabilityRadar(probes, evaluator=ContainsAnswer())
        result = radar.run_single(engine)

        assert len(result.a_by_domain) == 3
        assert result.b_by_domain is None
        assert result.bd_by_domain is None
        assert set(result.domains) == {"math", "knowledge", "code"}

        for d in result.domains:
            dr = result.a_by_domain[d]
            assert dr.n_probes == 5
            assert dr.accuracy == pytest.approx(3 / 5)

        assert result.engine_a_name == "mock_model"
        assert result.engine_b_name is None


class TestRunPairMock:
    def test_pair_calls(self):
        probes = _make_probes({"math": 3, "knowledge": 3, "code": 3})

        def gen_side_effect(texts, n_samples=1, max_new_tokens=16):
            gen = MagicMock()
            gen.completions = [["answer"] for _ in texts]
            gen.token_ids = [[[42]] for _ in texts]
            return gen

        engine_a = MagicMock()
        engine_a.model_name = "model_a"
        engine_a.generate.side_effect = gen_side_effect
        engine_a.config = MagicMock()
        engine_a.config.shares_tokenizer_with.return_value = True
        engine_a.tokenizer = MagicMock()

        engine_b = MagicMock()
        engine_b.model_name = "model_b"
        engine_b.generate.side_effect = gen_side_effect
        engine_b.config = MagicMock()
        engine_b.config.shares_tokenizer_with.return_value = True
        engine_b.tokenizer = MagicMock()

        # Mock score results
        def score_side_effect(prompts, continuations=None, continuation_ids=None):
            sr = MagicMock()
            sr.cross_entropies = [1.0] * len(prompts)
            sr.token_ids = [[[42]]] * len(prompts)
            return sr

        engine_a.score.side_effect = score_side_effect
        engine_b.score.side_effect = score_side_effect

        radar = CapabilityRadar(probes, evaluator=ContainsAnswer())
        result = radar.run_pair(engine_a, engine_b)

        assert len(result.a_by_domain) == 3
        assert len(result.b_by_domain) == 3
        assert len(result.bd_by_domain) == 3
        assert len(result.degeneracy_rates) == 3
        assert result.engine_a_name == "model_a"
        assert result.engine_b_name == "model_b"

        # Each domain: 1 Task.run(engine_a) + 1 Task.run(engine_b) + BD generate calls
        # engine_a.generate called: 3 (task) + 3 (BD) = 6 times
        # engine_b.generate called: 3 (task) + 3 (BD) = 6 times
        assert engine_a.generate.call_count == 6
        assert engine_b.generate.call_count == 6

        for d in result.domains:
            assert d in result.bd_by_domain
            assert d in result.degeneracy_rates
            assert "a" in result.degeneracy_rates[d]
            assert "b" in result.degeneracy_rates[d]
            assert result.a_by_domain[d].bd_vs_baseline is None
            assert result.b_by_domain[d].bd_vs_baseline is None
            assert result.bd_by_domain[d] is not None


class TestPrintRadarMock:
    def test_single_no_crash(self, capsys):
        from rich.console import Console
        console = Console(force_terminal=True, width=120)

        result = RadarResult(
            engine_a_name="gpt2",
            engine_b_name=None,
            domains=["code", "math"],
            a_by_domain={
                "code": DomainRadarResult("code", 10, 0.7),
                "math": DomainRadarResult("math", 10, 0.5),
            },
            b_by_domain=None,
            bd_by_domain=None,
            bd_healthy_by_domain=None,
            degeneracy_rates=None,
        )
        print_radar(result, console=console)

    def test_pair_no_crash(self, capsys):
        from rich.console import Console
        console = Console(force_terminal=True, width=120)

        result = RadarResult(
            engine_a_name="gpt2",
            engine_b_name="distilgpt2",
            domains=["code", "knowledge", "math"],
            a_by_domain={
                "code": DomainRadarResult("code", 30, 0.4),
                "knowledge": DomainRadarResult("knowledge", 30, 0.6),
                "math": DomainRadarResult("math", 30, 0.5),
            },
            b_by_domain={
                "code": DomainRadarResult("code", 30, 0.3),
                "knowledge": DomainRadarResult("knowledge", 30, 0.5),
                "math": DomainRadarResult("math", 30, 0.35),
            },
            bd_by_domain={"code": 1.42, "knowledge": 0.89, "math": 1.20},
            bd_healthy_by_domain={"code": 0.73, "knowledge": 0.95, "math": 0.59},
            degeneracy_rates={
                "code": {"a": 0.0, "b": 0.43},
                "knowledge": {"a": 0.0, "b": 0.10},
                "math": {"a": 0.0, "b": 0.27},
            },
        )
        print_radar(result, console=console)


# ── Architecture ────────────────────────────────────────────────────────


class TestCapabilityRadarArchitecture:
    def test_no_transformers_import(self):
        import lmdiff.tasks.capability_radar as mod
        src = inspect.getsource(mod)
        assert "import transformers" not in src

    def test_no_metric_base_import(self):
        """Capability radar uses BD via compute(), not via base class."""
        import lmdiff.tasks.capability_radar as mod
        src = inspect.getsource(mod)
        assert "from lmdiff.metrics.base" not in src


# ── E2E tests (slow) ───────────────────────────────────────────────────


@pytest.mark.slow
class TestCapabilityRadarE2E:
    def test_v01_gpt2_vs_distilgpt2(self, tiny_model, distil_engine, capsys):
        from rich.console import Console

        ps = ProbeSet.from_json(V01_PATH)
        assert len(ps) == 90
        assert set(ps.domains) == {"math", "knowledge", "code"}

        radar = CapabilityRadar(ps, evaluator=ContainsAnswer(), max_new_tokens=16)
        result = radar.run_pair(tiny_model, distil_engine)

        assert len(result.domains) == 3
        assert result.engine_a_name == "gpt2"
        assert result.engine_b_name == "distilgpt2"

        for d in result.domains:
            a_dr = result.a_by_domain[d]
            b_dr = result.b_by_domain[d]
            assert 0.0 <= a_dr.accuracy <= 1.0
            assert 0.0 <= b_dr.accuracy <= 1.0
            assert a_dr.n_probes == 30
            assert b_dr.n_probes == 30
            assert result.bd_by_domain[d] > 0
            assert d in result.degeneracy_rates
            assert 0.0 <= result.degeneracy_rates[d]["a"] <= 1.0
            assert 0.0 <= result.degeneracy_rates[d]["b"] <= 1.0

        console = Console(force_terminal=True, width=140)
        print_radar(result, console=console)

    def test_v01_single_gpt2(self, tiny_model):
        ps = ProbeSet.from_json(V01_PATH)
        radar = CapabilityRadar(ps, evaluator=ContainsAnswer(), max_new_tokens=16)
        result = radar.run_single(tiny_model)

        assert len(result.domains) == 3
        assert result.b_by_domain is None
        assert result.bd_by_domain is None
        for d in result.domains:
            assert 0.0 <= result.a_by_domain[d].accuracy <= 1.0
