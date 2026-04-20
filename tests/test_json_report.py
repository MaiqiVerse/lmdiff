from __future__ import annotations

import json
import math

import numpy as np
import pytest

from lmdiff.config import Config
from lmdiff.diff import DiffReport, FullReport, PairTaskResult
from lmdiff.metrics.base import MetricLevel, MetricResult
from lmdiff.report.json_report import SCHEMA_VERSION, to_json, to_json_dict
from lmdiff.tasks.base import EvalResult, TaskResult
from lmdiff.tasks.capability_radar import DomainRadarResult, RadarResult


def _make_metric_result(**overrides) -> MetricResult:
    defaults = dict(
        name="test_metric",
        level=MetricLevel.OUTPUT,
        value=0.42,
        details={"a": 1.0, "b": [1, 2, 3]},
        metadata={"foo": "bar"},
    )
    defaults.update(overrides)
    return MetricResult(**defaults)


def _make_eval_result() -> EvalResult:
    return EvalResult(
        probe_id="m1", output="42", expected="42",
        correct=True, score=1.0, metadata={"pos": 0},
    )


def _make_task_result() -> TaskResult:
    return TaskResult(
        task_name="test", engine_name="gpt2", probe_set_name="v01",
        n_probes=1, n_correct=1, accuracy=1.0,
        per_probe=[_make_eval_result()],
        per_domain={"math": {"n": 1, "correct": 1, "accuracy": 1.0}},
    )


def _make_diff_report() -> DiffReport:
    return DiffReport(
        config_a=Config(model="gpt2"),
        config_b=Config(model="distilgpt2"),
        results=[_make_metric_result()],
        metadata={"level": "output", "n_probes": 1},
    )


def _make_radar_result() -> RadarResult:
    return RadarResult(
        engine_a_name="gpt2",
        engine_b_name="distilgpt2",
        domains=["code", "math"],
        a_by_domain={
            "code": DomainRadarResult("code", 10, 0.7, 0.5),
            "math": DomainRadarResult("math", 10, 0.5, 0.3),
        },
        b_by_domain={
            "code": DomainRadarResult("code", 10, 0.6, 0.5),
            "math": DomainRadarResult("math", 10, 0.4, 0.3),
        },
        bd_by_domain={"code": 0.5, "math": 0.3},
        bd_healthy_by_domain={"code": 0.4, "math": None},
        degeneracy_rates={"code": {"a": 0.0, "b": 0.2}, "math": {"a": 0.0, "b": 0.1}},
    )


# ── Round-trip tests ────────────────────────────────────────────────────


class TestMetricResultJson:
    def test_round_trip(self):
        r = _make_metric_result()
        d = to_json_dict(r)
        s = json.dumps(d)
        reloaded = json.loads(s)
        assert reloaded["name"] == "test_metric"
        assert reloaded["level"] == "output"
        assert reloaded["value"] == 0.42

    def test_enum_serialized_as_string(self):
        d = to_json_dict(_make_metric_result())
        assert d["level"] == "output"
        assert isinstance(d["level"], str)


class TestEvalResultJson:
    def test_round_trip(self):
        d = to_json_dict(_make_eval_result())
        s = json.dumps(d)
        reloaded = json.loads(s)
        assert reloaded["probe_id"] == "m1"
        assert reloaded["correct"] is True


class TestTaskResultJson:
    def test_round_trip(self):
        d = to_json_dict(_make_task_result())
        s = json.dumps(d)
        reloaded = json.loads(s)
        assert reloaded["task_name"] == "test"
        assert reloaded["accuracy"] == 1.0
        assert len(reloaded["per_probe"]) == 1


class TestDiffReportJson:
    def test_round_trip(self):
        d = to_json_dict(_make_diff_report())
        s = json.dumps(d)
        reloaded = json.loads(s)
        assert reloaded["schema_version"] == SCHEMA_VERSION
        assert "generated_at" in reloaded
        assert reloaded["config_a"]["model"] == "gpt2"
        assert reloaded["config_b"]["model"] == "distilgpt2"
        assert len(reloaded["results"]) == 1

    def test_config_includes_dtype_field(self):
        report = DiffReport(
            config_a=Config(model="gpt2", dtype="float16"),
            config_b=Config(model="distilgpt2"),
            results=[],
            metadata={},
        )
        d = to_json_dict(report)
        assert d["config_a"]["dtype"] == "float16"
        assert d["config_b"]["dtype"] is None


class TestRadarResultJson:
    def test_round_trip(self):
        d = to_json_dict(_make_radar_result())
        s = json.dumps(d)
        reloaded = json.loads(s)
        assert reloaded["engine_a_name"] == "gpt2"
        assert reloaded["domains"] == ["code", "math"]
        assert reloaded["a_by_domain"]["code"]["accuracy"] == 0.7
        assert reloaded["bd_healthy_by_domain"]["math"] is None


class TestPairTaskResultJson:
    def test_round_trip(self):
        ptr = PairTaskResult(
            task_name="t1",
            result_a=_make_task_result(),
            result_b=_make_task_result(),
            delta_accuracy=0.0,
            per_domain_delta={"math": 0.0},
        )
        d = to_json_dict(ptr)
        s = json.dumps(d)
        reloaded = json.loads(s)
        assert reloaded["task_name"] == "t1"
        assert "result_a" in reloaded
        assert "result_b" in reloaded


class TestFullReportJson:
    def test_round_trip(self):
        fr = FullReport(
            config_a=Config(model="gpt2"),
            config_b=Config(model="distilgpt2"),
            diff_report=_make_diff_report(),
            task_results=[],
            radar_result=_make_radar_result(),
        )
        d = to_json_dict(fr)
        s = json.dumps(d)
        reloaded = json.loads(s)
        assert reloaded["schema_version"] == SCHEMA_VERSION
        assert reloaded["diff_report"] is not None
        assert reloaded["radar_result"] is not None


# ── Edge cases ──────────────────────────────────────────────────────────


class TestNanInf:
    def test_nan_becomes_none(self):
        r = _make_metric_result(value=float("nan"))
        d = to_json_dict(r)
        assert d["value"] is None

    def test_inf_becomes_none(self):
        r = _make_metric_result(value=float("inf"))
        d = to_json_dict(r)
        assert d["value"] is None

    def test_nan_in_details(self):
        r = _make_metric_result(details={"x": float("nan"), "y": 1.0})
        d = to_json_dict(r)
        assert d["details"]["x"] is None
        assert d["details"]["y"] == 1.0


class TestNumpyArray:
    def test_array_to_list(self):
        arr = np.array([1.0, 2.0, 3.0])
        r = _make_metric_result(value=0.0, details={"arr": arr})
        d = to_json_dict(r)
        assert d["details"]["arr"] == [1.0, 2.0, 3.0]
        json.dumps(d)  # must not raise

    def test_numpy_scalar(self):
        r = _make_metric_result(value=0.0, details={"x": np.float64(3.14)})
        d = to_json_dict(r)
        assert isinstance(d["details"]["x"], float)


class TestConfigObjectModel:
    def test_object_model_no_crash(self):
        class FakeModel:
            pass

        r = DiffReport(
            config_a=Config(model=FakeModel()),
            config_b=Config(model="distilgpt2"),
            results=[],
        )
        d = to_json_dict(r)
        assert d["config_a"]["model"] == "<object>"
        json.dumps(d)  # must not raise


class TestDeterministic:
    def test_same_output_twice(self):
        r = _make_metric_result()
        s1 = to_json(r)
        s2 = to_json(r)
        assert s1 == s2

    def test_diff_report_keys_sorted(self):
        s = to_json(_make_diff_report())
        d = json.loads(s)
        keys = list(d.keys())
        assert keys == sorted(keys)


class TestSchemaVersion:
    def test_diff_report_has_schema_version(self):
        d = to_json_dict(_make_diff_report())
        assert d["schema_version"] == SCHEMA_VERSION

    def test_full_report_has_schema_version(self):
        fr = FullReport(
            config_a=Config(model="gpt2"),
            config_b=Config(model="distilgpt2"),
            diff_report=None,
            task_results=[],
        )
        d = to_json_dict(fr)
        assert d["schema_version"] == SCHEMA_VERSION


class TestUnsupportedType:
    def test_unsupported_raises(self):
        with pytest.raises(TypeError, match="does not support"):
            to_json_dict("not a dataclass")
