"""E2E CLI tests that load real models."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lmdiff.cli import app

runner = CliRunner()


def _extract_json(output: str) -> dict:
    """Extract JSON object from output that may contain HF warnings."""
    start = output.find("{")
    assert start >= 0, f"No JSON found in output:\n{output[:200]}"
    return json.loads(output[start:])


@pytest.mark.slow
class TestCompareE2E:
    def test_compare_json_output(self, tmp_path):
        out_file = tmp_path / "result.json"
        result = runner.invoke(app, [
            "compare", "gpt2", "distilgpt2",
            "--probes", "v01",
            "--json",
            "--output", str(out_file),
            "--max-new-tokens", "16",
        ])
        assert result.exit_code == 0, result.output
        assert out_file.exists()

        data = json.loads(out_file.read_text())
        assert data["schema_version"] == "4"
        assert any(r["name"] == "behavioral_distance" for r in data["results"])
        assert data["config_a"]["model"] == "gpt2"
        assert data["config_b"]["model"] == "distilgpt2"

    def test_compare_terminal_output(self):
        result = runner.invoke(app, [
            "compare", "gpt2", "distilgpt2",
            "--probes", "v01",
            "--max-new-tokens", "16",
        ])
        assert result.exit_code == 0
        assert "behavioral_distance" in result.output


@pytest.mark.slow
class TestRadarE2E:
    def test_radar_json_output(self, tmp_path):
        out_file = tmp_path / "radar.json"
        result = runner.invoke(app, [
            "radar", "gpt2", "distilgpt2",
            "--probes", "v01",
            "--json",
            "--output", str(out_file),
            "--max-new-tokens", "16",
        ])
        assert result.exit_code == 0, result.output
        assert out_file.exists()

        data = json.loads(out_file.read_text())
        assert data["engine_a_name"] == "gpt2"
        assert "code" in data["domains"]
        assert data["bd_by_domain"]["code"] > 0


@pytest.mark.slow
class TestFamilyExperimentE2E:
    """Smoke test that CLI args reach run_family_experiment correctly.

    Patches the heavy library entry point so this stays in 'slow' purely
    because it exercises CLI plumbing for an experiment that would
    otherwise need a real model load.
    """

    def test_cli_args_reach_library(self, tmp_path):
        from unittest.mock import patch
        with patch("lmdiff.experiments.family.run_family_experiment") as mock_run:
            result = runner.invoke(app, [
                "family-experiment",
                "--base", "gpt2",
                "--variant", "small=distilgpt2",
                "--tasks", "hellaswag",
                "--limit-per-task", "2",
                "--output-dir", str(tmp_path),
                "--no-radars",
                "--skip-accuracy",
            ])
        assert result.exit_code == 0, result.output
        kw = mock_run.call_args.kwargs
        assert kw["base"] == "gpt2"
        assert kw["variants"] == {"small": "distilgpt2"}
        assert kw["tasks"] == ["hellaswag"]
        assert kw["limit_per_task"] == 2
        assert kw["skip_accuracy"] is True
        assert kw["render_radars"] is False


@pytest.mark.slow
class TestRunTaskE2E:
    def test_run_task_json(self):
        result = runner.invoke(app, [
            "run-task", "gpt2",
            "--probes", "v01",
            "--json",
            "--max-new-tokens", "16",
        ])
        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert data["engine_name"] == "gpt2"
        assert data["n_probes"] == 90
