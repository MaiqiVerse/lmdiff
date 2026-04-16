"""Unit tests for lmdiff CLI (no model loading)."""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lmdiff.cli import app

runner = CliRunner()


class TestHelp:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "compare" in result.output
        assert "radar" in result.output
        assert "list-metrics" in result.output
        assert "run-task" in result.output

    def test_compare_help(self):
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "MODEL_A" in result.output
        assert "--verbose" in result.output

    def test_radar_help(self):
        result = runner.invoke(app, ["radar", "--help"])
        assert result.exit_code == 0

    def test_run_task_help(self):
        result = runner.invoke(app, ["run-task", "--help"])
        assert result.exit_code == 0

    def test_list_metrics_help(self):
        result = runner.invoke(app, ["list-metrics", "--help"])
        assert result.exit_code == 0


class TestListMetrics:
    def test_lists_three_metrics(self):
        result = runner.invoke(app, ["list-metrics"])
        assert result.exit_code == 0
        assert "behavioral_distance" in result.output
        assert "token_entropy" in result.output
        assert "token_kl" in result.output

    def test_filter_by_level(self):
        result = runner.invoke(app, ["list-metrics", "--level", "output"])
        assert result.exit_code == 0
        assert "behavioral_distance" in result.output

    def test_no_model_load(self):
        """list-metrics must not import transformers at top level."""
        import lmdiff.cli as cli_mod
        import inspect
        src = inspect.getsource(cli_mod)
        # Top-level (outside functions) should not have transformers
        # We check that 'import transformers' doesn't appear before any def
        lines_before_first_def = []
        for line in src.split("\n"):
            if line.strip().startswith("def ") or line.strip().startswith("class "):
                break
            lines_before_first_def.append(line)
        top_level = "\n".join(lines_before_first_def)
        assert "import transformers" not in top_level
        assert "from transformers" not in top_level


class TestCompareJsonMock:
    def test_json_output(self):
        from lmdiff.config import Config
        from lmdiff.diff import DiffReport
        from lmdiff.metrics.base import MetricLevel, MetricResult

        fake_report = DiffReport(
            config_a=Config(model="gpt2"),
            config_b=Config(model="distilgpt2"),
            results=[MetricResult(name="behavioral_distance", level=MetricLevel.OUTPUT, value=0.5)],
            metadata={"level": "output", "n_probes": 1, "name_a": "gpt2", "name_b": "distilgpt2"},
        )

        with patch("lmdiff.diff.ModelDiff") as MockMD, \
             patch("lmdiff.probes.loader.ProbeSet.from_json") as mock_from_json:
            mock_from_json.return_value = MagicMock()
            instance = MockMD.return_value
            instance.run.return_value = fake_report

            result = runner.invoke(app, ["compare", "gpt2", "distilgpt2", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["schema_version"] == "1"
        assert parsed["results"][0]["name"] == "behavioral_distance"

    def test_missing_probes_fails(self):
        result = runner.invoke(app, [
            "compare", "gpt2", "distilgpt2",
            "--probes", "nonexistent_probe_set_xyz",
        ])
        assert result.exit_code != 0

    def test_unsupported_level_fails(self):
        result = runner.invoke(app, [
            "compare", "gpt2", "distilgpt2",
            "--level", "representation",
        ])
        assert result.exit_code != 0
