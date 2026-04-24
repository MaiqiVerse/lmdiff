"""Unit tests for `lmdiff family-experiment` and `lmdiff plot-geometry`."""
from __future__ import annotations

import os
import re
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lmdiff.cli import app

os.environ.setdefault("COLUMNS", "200")
runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _plain(s: str) -> str:
    """Strip ANSI escape sequences that rich/typer inject into help output."""
    return _ANSI_RE.sub("", s)


# ── help / registration ───────────────────────────────────────────────


class TestRegistration:
    def test_main_help_lists_new_commands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        plain = _plain(result.output)
        assert "family-experiment" in plain
        assert "plot-geometry" in plain

    def test_family_experiment_help(self):
        result = runner.invoke(app, ["family-experiment", "--help"])
        assert result.exit_code == 0
        plain = _plain(result.output)
        assert "--variant" in plain
        assert "--base" in plain
        assert "--output-dir" in plain

    def test_plot_geometry_help(self):
        result = runner.invoke(app, ["plot-geometry", "--help"])
        assert result.exit_code == 0
        plain = _plain(result.output)
        assert "--output-dir" in plain


# ── --variant parsing (repeatable, NOT comma-separated) ───────────────


class TestVariantParsing:
    def test_repeatable_variants_collected(self, tmp_path):
        with patch("lmdiff.experiments.family.run_family_experiment") as mock_run:
            result = runner.invoke(
                app,
                [
                    "family-experiment",
                    "--base", "base-id",
                    "--variant", "yarn=path/yarn",
                    "--variant", "code=path/code",
                    "--tasks", "hellaswag",
                    "--output-dir", str(tmp_path),
                    "--no-radars",
                ],
            )
        assert result.exit_code == 0, result.output
        kwargs = mock_run.call_args.kwargs
        assert kwargs["base"] == "base-id"
        assert kwargs["variants"] == {"yarn": "path/yarn", "code": "path/code"}
        assert kwargs["tasks"] == ["hellaswag"]

    def test_variant_without_equals_rejected(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "family-experiment",
                "--base", "b",
                "--variant", "broken-no-equals",
                "--output-dir", str(tmp_path),
            ],
        )
        assert result.exit_code != 0
        assert "name=model_id" in _plain(result.output)

    def test_duplicate_variant_name_rejected(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "family-experiment",
                "--base", "b",
                "--variant", "v=a",
                "--variant", "v=b",
                "--output-dir", str(tmp_path),
            ],
        )
        assert result.exit_code != 0
        assert "Duplicate variant" in _plain(result.output)


# ── option forwarding ────────────────────────────────────────────────


class TestOptionForwarding:
    def test_default_tasks_used_when_omitted(self, tmp_path):
        with patch("lmdiff.experiments.family.run_family_experiment") as mock_run:
            result = runner.invoke(
                app,
                [
                    "family-experiment",
                    "--base", "b",
                    "--variant", "v=m",
                    "--output-dir", str(tmp_path),
                    "--no-radars",
                ],
            )
        assert result.exit_code == 0, result.output
        from lmdiff.experiments.family import DEFAULT_TASKS
        assert mock_run.call_args.kwargs["tasks"] == list(DEFAULT_TASKS)

    def test_skip_accuracy_and_dtype_forwarded(self, tmp_path):
        with patch("lmdiff.experiments.family.run_family_experiment") as mock_run:
            result = runner.invoke(
                app,
                [
                    "family-experiment",
                    "--base", "b",
                    "--variant", "v=m",
                    "--tasks", "hellaswag",
                    "--output-dir", str(tmp_path),
                    "--skip-accuracy",
                    "--dtype", "bfloat16",
                    "--limit-per-task", "10",
                    "--max-new-tokens", "32",
                    "--seed", "7",
                    "--output-prefix", "exp_xyz",
                    "--no-radars",
                ],
            )
        assert result.exit_code == 0, result.output
        kw = mock_run.call_args.kwargs
        assert kw["skip_accuracy"] is True
        assert kw["dtype"] == "bfloat16"
        assert kw["limit_per_task"] == 10
        assert kw["max_new_tokens"] == 32
        assert kw["seed"] == 7
        assert kw["output_prefix"] == "exp_xyz"
        assert kw["render_radars"] is False
        assert kw["write_outputs"] is True


# ── plot-geometry CLI ────────────────────────────────────────────────


class TestPlotGeometryCli:
    def test_missing_input_rejected(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "plot-geometry",
                str(tmp_path / "missing.json"),
                "--output-dir", str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
        plain = _plain(result.output).lower()
        assert "not found" in plain or "does not exist" in plain

    def test_calls_library_with_existing_input(self, tmp_path):
        fake_json = tmp_path / "geo.json"
        fake_json.write_text("{}", encoding="utf-8")
        out_dir = tmp_path / "figs"

        with patch("lmdiff.viz.family_figures.plot_family_figures") as mock_plot:
            mock_plot.return_value = {"cosine_raw": out_dir / "x.png"}
            result = runner.invoke(
                app,
                [
                    "plot-geometry",
                    str(fake_json),
                    "--output-dir", str(out_dir),
                ],
            )
        assert result.exit_code == 0, result.output
        mock_plot.assert_called_once()
        kwargs = mock_plot.call_args.kwargs
        # Default: which=None renders all 7 figures.
        assert kwargs["which"] is None
        assert kwargs["dpi"] == 200

    def test_figures_subset_flag_forwarded(self, tmp_path):
        fake_json = tmp_path / "geo.json"
        fake_json.write_text("{}", encoding="utf-8")
        with patch("lmdiff.viz.family_figures.plot_family_figures") as mock_plot:
            mock_plot.return_value = {"specialization": tmp_path / "x.png"}
            result = runner.invoke(
                app,
                [
                    "plot-geometry",
                    str(fake_json),
                    "--output-dir", str(tmp_path / "out"),
                    "--figures", "specialization,cosine_raw",
                ],
            )
        assert result.exit_code == 0, result.output
        assert mock_plot.call_args.kwargs["which"] == ["specialization", "cosine_raw"]

    def test_unknown_figure_key_rejected(self, tmp_path):
        fake_json = tmp_path / "geo.json"
        fake_json.write_text("{}", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "plot-geometry",
                str(fake_json),
                "--output-dir", str(tmp_path / "out"),
                "--figures", "nonexistent_figure",
            ],
        )
        assert result.exit_code != 0
        assert "Unknown figure key" in _plain(result.output)

    def test_variant_and_domain_order_flags_forwarded(self, tmp_path):
        fake_json = tmp_path / "geo.json"
        fake_json.write_text("{}", encoding="utf-8")
        with patch("lmdiff.viz.family_figures.plot_family_figures") as mock_plot:
            mock_plot.return_value = {"cosine_raw": tmp_path / "x.png"}
            result = runner.invoke(
                app,
                [
                    "plot-geometry",
                    str(fake_json),
                    "--output-dir", str(tmp_path / "out"),
                    "--variant-order", "yarn,long",
                    "--domain-order", "code,math",
                    "--dpi", "150",
                ],
            )
        assert result.exit_code == 0, result.output
        kw = mock_plot.call_args.kwargs
        assert kw["variant_order"] == ["yarn", "long"]
        assert kw["domain_order"] == ["code", "math"]
        assert kw["dpi"] == 150

    def test_empty_render_returns_nonzero(self, tmp_path):
        fake_json = tmp_path / "geo.json"
        fake_json.write_text("{}", encoding="utf-8")
        with patch("lmdiff.viz.family_figures.plot_family_figures") as mock_plot:
            mock_plot.return_value = {}
            result = runner.invoke(
                app,
                [
                    "plot-geometry",
                    str(fake_json),
                    "--output-dir", str(tmp_path / "out"),
                ],
            )
        assert result.exit_code == 1
