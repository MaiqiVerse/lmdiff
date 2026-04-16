from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lmdiff.config import Config
from lmdiff.diff import DiffReport, FullReport, ModelDiff, PairTaskResult, _OUTPUT_METRICS
from lmdiff.metrics.base import MetricLevel, MetricResult
from lmdiff.metrics.output.behavioral_distance import BehavioralDistance
from lmdiff.metrics.output.token_entropy import TokenEntropy
from lmdiff.metrics.output.token_kl import TokenKL
from lmdiff.probes.loader import ProbeSet
from lmdiff.report.terminal import print_report

V01_PATH = Path(__file__).parent.parent / "lmdiff" / "probes" / "v01.json"


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

    def test_bd_breakdown_truncation(self, capsys):
        from rich.console import Console

        per_prompt = [
            {"probe": f"probe_{i}", "ce_aa": 1.0, "ce_ab": 1.5,
             "ce_ba": 1.2, "ce_bb": 0.9, "bd": 0.1 * i, "asymmetry": 0.01}
            for i in range(20)
        ]
        results = [
            MetricResult(
                name="behavioral_distance",
                level=MetricLevel.OUTPUT,
                value=0.5,
                details={
                    "ce_aa": 1.0, "ce_ab": 1.5, "ce_ba": 1.2, "ce_bb": 0.9,
                    "asymmetry": 0.01, "bpb_normalized": False,
                    "per_prompt": per_prompt,
                },
            ),
        ]
        report = DiffReport(
            config_a=Config(model="gpt2"),
            config_b=Config(model="distilgpt2"),
            results=results,
            metadata={"name_a": "gpt2", "name_b": "distilgpt2", "level": "output", "n_probes": 20},
        )

        # verbose=False: should truncate
        import io
        buf = io.StringIO()
        cons = Console(force_terminal=True, width=120, file=buf)
        print_report(report, console=cons, verbose=False)
        text = buf.getvalue()
        assert "more" in text and "--verbose" in text

        # verbose=True: should show all
        buf2 = io.StringIO()
        cons2 = Console(force_terminal=True, width=120, file=buf2)
        print_report(report, console=cons2, verbose=True)
        text2 = buf2.getvalue()
        assert "--verbose" not in text2


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
        assert report.metadata["probe_set_version"] == "0.2.1"
        assert report.metadata["n_probes"] == 90


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


# ── Part A: run_task / run_tasks / run_radar ────────────────────────────


class TestRunTaskMock:
    def test_pair_task_result(self):
        from lmdiff.probes.loader import Probe
        from lmdiff.tasks.base import Task
        from lmdiff.tasks.evaluators import ExactMatch

        probes = ProbeSet([
            Probe(id="m1", text="1+1=", domain="math", expected="2"),
            Probe(id="k1", text="France is ", domain="knowledge", expected="Paris"),
        ])
        task = Task("test_task", probes, ExactMatch(), max_new_tokens=8)

        md = ModelDiff(Config(model="gpt2"), Config(model="distilgpt2"), probes)

        mock_a = MagicMock()
        mock_a.model_name = "gpt2"
        gen_a = MagicMock()
        gen_a.completions = [["2"], ["Paris"]]
        mock_a.generate.return_value = gen_a

        mock_b = MagicMock()
        mock_b.model_name = "distilgpt2"
        gen_b = MagicMock()
        gen_b.completions = [["3"], ["Paris"]]
        mock_b.generate.return_value = gen_b

        with patch.object(md, "_engine_a", mock_a), \
             patch.object(md, "_engine_b", mock_b):
            ptr = md.run_task(task)

        assert isinstance(ptr, PairTaskResult)
        assert ptr.task_name == "test_task"
        assert ptr.result_a.accuracy == 1.0
        assert ptr.result_b.accuracy == 0.5
        assert abs(ptr.delta_accuracy - (-0.5)) < 1e-6
        assert "math" in ptr.per_domain_delta
        assert "knowledge" in ptr.per_domain_delta
        assert ptr.metadata["evaluator"] == "exact_match"
        assert ptr.metadata["n_probes"] == 2

    def test_run_tasks_multiple(self):
        from lmdiff.probes.loader import Probe
        from lmdiff.tasks.base import Task
        from lmdiff.tasks.evaluators import ExactMatch

        probes1 = ProbeSet([
            Probe(id="a", text="x", domain="d1", expected="y"),
            Probe(id="b", text="z", domain="d2", expected="w"),
        ])
        probes2 = ProbeSet([
            Probe(id="c", text="q", domain="d1", expected="r"),
        ])
        task1 = Task("t1", probes1, ExactMatch(), max_new_tokens=8)
        task2 = Task("t2", probes2, ExactMatch(), max_new_tokens=8)

        md = ModelDiff(Config(model="gpt2"), Config(model="gpt2"), ["dummy"])

        mock_eng = MagicMock()
        mock_eng.model_name = "gpt2"

        def gen_side(texts, **kw):
            g = MagicMock()
            g.completions = [["wrong"]] * len(texts)
            return g

        mock_eng.generate.side_effect = gen_side

        with patch.object(md, "_engine_a", mock_eng), \
             patch.object(md, "_engine_b", mock_eng):
            fr = md.run_tasks([task1, task2])

        assert isinstance(fr, FullReport)
        assert len(fr.task_results) == 2
        assert fr.task_results[0].task_name == "t1"
        assert fr.task_results[1].task_name == "t2"
        assert fr.diff_report is None

    def test_run_radar_mock(self):
        from lmdiff.probes.loader import Probe
        from lmdiff.tasks.capability_radar import RadarResult

        multi_probes = ProbeSet([
            Probe(id="m1", text="1+1=", domain="math", expected="2"),
            Probe(id="k1", text="France", domain="knowledge", expected="Paris"),
        ])
        md = ModelDiff(Config(model="gpt2"), Config(model="distilgpt2"), multi_probes)

        fake_radar_result = MagicMock(spec=RadarResult)

        with patch("lmdiff.tasks.capability_radar.CapabilityRadar") as MockRadar:
            instance = MockRadar.return_value
            instance.run_pair.return_value = fake_radar_result

            with patch.object(md, "_engine_a", MagicMock()), \
                 patch.object(md, "_engine_b", MagicMock()):
                result = md.run_radar()

        MockRadar.assert_called_once()
        instance.run_pair.assert_called_once()
        assert result is fake_radar_result

    def test_run_radar_single_domain_raises(self):
        md = ModelDiff(
            Config(model="gpt2"), Config(model="distilgpt2"),
            ["hello", "world"],
        )
        with patch.object(md, "_engine_a", MagicMock()), \
             patch.object(md, "_engine_b", MagicMock()):
            with pytest.raises(ValueError, match="multi-domain ProbeSet"):
                md.run_radar()
