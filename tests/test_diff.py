from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modeldiff.config import Config
from modeldiff.diff import DiffReport, ModelDiff, _OUTPUT_METRICS
from modeldiff.metrics.base import MetricLevel, MetricResult
from modeldiff.metrics.output.behavioral_distance import BehavioralDistance
from modeldiff.metrics.output.token_entropy import TokenEntropy
from modeldiff.metrics.output.token_kl import TokenKL
from modeldiff.probes.loader import ProbeSet
from modeldiff.report.terminal import print_report

V01_PATH = Path(__file__).parent.parent / "modeldiff" / "probes" / "v01.json"


class TestDiffReport:
    def test_get_by_name(self):
        r1 = MetricResult(name="a", level=MetricLevel.OUTPUT, value=1.0)
        r2 = MetricResult(name="b", level=MetricLevel.OUTPUT, value=2.0)
        report = DiffReport(
            config_a=Config(model="x"),
            config_b=Config(model="y"),
            results=[r1, r2],
        )
        assert report.get("a") is r1
        assert report.get("b") is r2
        assert report.get("nonexistent") is None


class TestModelDiffMock:
    def test_lazy_engine_creation(self):
        md = ModelDiff(Config(model="gpt2"), Config(model="gpt2"), ["test"])
        assert md._engine_a is None
        assert md._engine_b is None

    def test_run_returns_diff_report(self):
        md = ModelDiff(Config(model="gpt2"), Config(model="gpt2"), ["test"])

        fake_result = MetricResult(
            name="fake", level=MetricLevel.OUTPUT, value=0.0,
        )

        class FakeMetric:
            name = "fake"
            level = MetricLevel.OUTPUT

            @classmethod
            def is_applicable(cls, a, b):
                return True

            def compute(self, ea, eb, probes, **kw):
                return fake_result

        with patch.object(md, "_engine_a", MagicMock()), \
             patch.object(md, "_engine_b", MagicMock()):
            report = md.run(metrics=[FakeMetric])

        assert isinstance(report, DiffReport)
        assert len(report.results) == 1
        assert report.results[0].name == "fake"

    def test_skips_inapplicable_metrics(self):
        md = ModelDiff(Config(model="gpt2"), Config(model="gpt2"), ["test"])

        class NeverApplicable:
            name = "skip_me"
            level = MetricLevel.OUTPUT

            @classmethod
            def is_applicable(cls, a, b):
                return False

            def compute(self, ea, eb, probes, **kw):
                raise AssertionError("should not be called")

        with patch.object(md, "_engine_a", MagicMock()), \
             patch.object(md, "_engine_b", MagicMock()):
            report = md.run(metrics=[NeverApplicable])

        assert len(report.results) == 0

    def test_default_output_metrics(self):
        assert len(_OUTPUT_METRICS) == 3
        assert BehavioralDistance in _OUTPUT_METRICS
        assert TokenEntropy in _OUTPUT_METRICS
        assert TokenKL in _OUTPUT_METRICS

    def test_metadata_populated(self):
        md = ModelDiff(Config(model="gpt2"), Config(model="distilgpt2"), ["p"])

        with patch.object(md, "_engine_a", MagicMock()), \
             patch.object(md, "_engine_b", MagicMock()):
            report = md.run(metrics=[])

        assert report.metadata["name_a"] == "gpt2"
        assert report.metadata["name_b"] == "distilgpt2"
        assert report.metadata["n_probes"] == 1


class TestTerminalReport:
    def test_print_report_no_crash(self, capsys):
        from rich.console import Console
        console = Console(force_terminal=True, width=120)

        results = [
            MetricResult(
                name="behavioral_distance",
                level=MetricLevel.OUTPUT,
                value=0.34,
                details={
                    "ce_aa": 1.5, "ce_ab": 1.9, "ce_ba": 1.4, "ce_bb": 1.1,
                    "asymmetry": 0.04, "bpb_normalized": False,
                    "per_prompt": [
                        {"probe": "Hello", "ce_aa": 1.5, "ce_ab": 1.9,
                         "ce_ba": 1.4, "ce_bb": 1.1, "bd": 0.34, "asymmetry": 0.04},
                    ],
                },
            ),
            MetricResult(
                name="token_entropy", level=MetricLevel.OUTPUT,
                value=-0.5, details={"mean_entropy_a": 3.0, "mean_entropy_b": 2.5},
            ),
            MetricResult(
                name="token_kl", level=MetricLevel.OUTPUT,
                value=0.12, details={"kl_ab": 0.10, "kl_ba": 0.14},
            ),
        ]

        report = DiffReport(
            config_a=Config(model="gpt2"),
            config_b=Config(model="distilgpt2"),
            results=results,
            metadata={"name_a": "gpt2", "name_b": "distilgpt2", "level": "output", "n_probes": 1},
        )

        print_report(report, console=console)


@pytest.mark.slow
class TestModelDiffE2E:
    def test_gpt2_vs_distilgpt2(self, capsys):
        from rich.console import Console

        probes = ["The capital of France is", "2 + 2 =", "Once upon a time"]
        md = ModelDiff(
            Config(model="gpt2"),
            Config(model="distilgpt2"),
            probes,
        )
        report = md.run(level="output", max_new_tokens=16)

        assert len(report.results) == 3
        bd = report.get("behavioral_distance")
        assert bd is not None
        assert bd.value > 0

        te = report.get("token_entropy")
        assert te is not None

        kl = report.get("token_kl")
        assert kl is not None
        assert kl.value > 0

        console = Console(force_terminal=True, width=120)
        print_report(report, console=console)


class TestModelDiffWithProbeSet:
    def test_accepts_probeset(self):
        ps = ProbeSet.from_list(["hello", "world"], domain="test")
        md = ModelDiff(Config(model="gpt2"), Config(model="gpt2"), ps)
        assert md.prompts == ["hello", "world"]
        assert md.probe_set.name is None
        assert len(md.probe_set) == 2

    def test_probeset_metadata_in_report(self):
        ps = ProbeSet.from_json(V01_PATH)
        md = ModelDiff(Config(model="gpt2"), Config(model="gpt2"), ps)

        fake_result = MetricResult(name="f", level=MetricLevel.OUTPUT, value=0.0)

        class Fake:
            name = "f"
            level = MetricLevel.OUTPUT

            @classmethod
            def is_applicable(cls, a, b):
                return True

            def compute(self, ea, eb, probes, **kw):
                return fake_result

        with patch.object(md, "_engine_a", MagicMock()), \
             patch.object(md, "_engine_b", MagicMock()):
            report = md.run(metrics=[Fake])

        assert report.metadata["probe_set_name"] == "v01"
        assert report.metadata["probe_set_version"] == "0.1.1"
        assert report.metadata["n_probes"] == 30


@pytest.mark.slow
class TestModelDiffV01E2E:
    def test_v01_gpt2_vs_distilgpt2(self, tiny_model, distil_engine):
        ps = ProbeSet.from_json(V01_PATH)
        md = ModelDiff(
            Config(model="gpt2"),
            Config(model="distilgpt2"),
            ps,
        )
        md._engine_a = tiny_model
        md._engine_b = distil_engine

        report = md.run(level="output", max_new_tokens=16)
        bd = report.get("behavioral_distance")
        assert bd is not None
        assert bd.value > 0

        by_domain: dict[str, list[dict]] = {}
        for pp in bd.details["per_prompt"]:
            probe_obj = next((p for p in ps if p.text == pp["probe"]), None)
            domain = probe_obj.domain if probe_obj else "unknown"
            by_domain.setdefault(domain, []).append(pp)

        print(f"\n=== v01 BD(gpt2, distilgpt2) = {bd.value:.4f} ===")
        for domain, entries in sorted(by_domain.items()):
            valid = [e for e in entries if not e.get("skipped")]
            if not valid:
                continue
            mean_bd = sum(e["bd"] for e in valid) / len(valid)
            print(f"  {domain:12s}: BD={mean_bd:.4f} ({len(valid)} probes)")

        assert report.metadata["probe_set_name"] == "v01"
