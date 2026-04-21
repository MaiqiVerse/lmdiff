"""Tests for scripts/discover_lm_eval_tasks.py — fully mocked, no real lm_eval."""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "discover_lm_eval_tasks.py"


@pytest.fixture
def discovery_module(monkeypatch):
    """Load the script as a module (it's not in a package)."""
    spec = importlib.util.spec_from_file_location(
        "discover_lm_eval_tasks", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Make re-imports deterministic per test.
    sys.modules.pop("discover_lm_eval_tasks", None)
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("discover_lm_eval_tasks", None)


# ── suggest_domain ─────────────────────────────────────────────────────

class TestSuggestDomain:
    def test_prefix_match_to_mmlu(self, discovery_module):
        # KNOWN_TASK_DOMAINS has "mmlu" entry with domain="knowledge".
        # Task name "mmlu_anything" splits to "mmlu" → knowledge.
        registered = set()  # suggest_domain doesn't actually use this arg
        domain, reason = discovery_module.suggest_domain(
            "mmlu_foo_bar", registered,
        )
        assert domain == "knowledge"
        assert reason == "prefix:mmlu"

    def test_longbench_like_name_reaches_long_context(self, discovery_module):
        # "longbench" is not itself a key in KNOWN_TASK_DOMAINS (only
        # longbench_2wikimqa etc are). Prefix lookup fails → keyword "long"
        # fires → domain "long-context". Result is still right, path is via
        # keyword not prefix.
        domain, reason = discovery_module.suggest_domain(
            "longbench_some_new_subset", set(),
        )
        assert domain == "long-context"
        assert reason == "keyword:long"

    def test_keyword_match_math(self, discovery_module):
        # No underscore prefix that's in the registry.
        domain, reason = discovery_module.suggest_domain(
            "newmathbench", set(),
        )
        assert domain == "math"
        assert reason == "keyword:math"

    def test_keyword_match_code(self, discovery_module):
        domain, reason = discovery_module.suggest_domain(
            "codebench_v2", set(),
        )
        # Note: first split of "codebench_v2" is "codebench", not in registry →
        # falls to keyword match "code".
        assert domain == "code"
        assert reason == "keyword:code"

    def test_fallback_unknown(self, discovery_module):
        domain, reason = discovery_module.suggest_domain(
            "zzbenchmark", set(),
        )
        assert domain == "unknown"
        assert reason == "none"

    def test_prefix_wins_over_keyword(self, discovery_module):
        # "mmlu_math_stuff" — prefix "mmlu" is in registry → knowledge.
        # (Keyword "math" would otherwise suggest "math".)
        domain, reason = discovery_module.suggest_domain(
            "mmlu_math_stuff", set(),
        )
        assert reason.startswith("prefix:")
        assert domain == "knowledge"


# ── ImportError path ───────────────────────────────────────────────────

class TestImportErrorExit:
    def test_missing_lm_eval_exits_2(self, discovery_module, monkeypatch, capsys):
        # Force `import lm_eval` inside the script to fail.
        monkeypatch.delitem(sys.modules, "lm_eval", raising=False)

        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "lm_eval" or name.startswith("lm_eval."):
                raise ImportError("no lm_eval")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(SystemExit) as exc_info:
            discovery_module._import_lm_eval()
        assert exc_info.value.code == 2

        captured = capsys.readouterr()
        assert "pip install lmdiff-kit[lm-eval]" in captured.err


# ── Mocked lm_eval path ────────────────────────────────────────────────

def _install_mock_lm_eval(monkeypatch, task_names: list[str], version: str = "0.4.99"):
    """Install a fake lm_eval into sys.modules whose TaskManager yields task_names."""
    fake_tm = MagicMock(name="TaskManager-instance")
    fake_tm.all_tasks = task_names

    # Shallow config lookup: return a minimal dict-config object per task.
    def fake_get_config(name: str) -> dict:
        return {
            "output_type": "multiple_choice",
            "metric_list": [{"metric": "acc"}],
        }
    fake_tm._get_config = fake_get_config

    TaskManagerFactory = MagicMock(return_value=fake_tm)

    fake_tasks_mod = types.SimpleNamespace(TaskManager=TaskManagerFactory)
    fake_module = types.SimpleNamespace(
        __version__=version,
        tasks=fake_tasks_mod,
    )
    monkeypatch.setitem(sys.modules, "lm_eval", fake_module)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", fake_tasks_mod)
    return fake_module


class TestBuildReport:
    def test_shallow_run_produces_report(self, discovery_module, monkeypatch):
        # Two registered tasks (in KNOWN_TASK_DOMAINS) + two unregistered.
        _install_mock_lm_eval(monkeypatch, [
            "hellaswag", "gsm8k",  # registered
            "fake_new_task_a", "mmlu_fake_subject",  # unregistered
        ])
        report = discovery_module.build_report(
            filter_pattern="*",
            deep=False,
            verbose=False,
        )
        assert report["lm_eval_version"] == "0.4.99"
        assert report["lm_eval_registry_size"] == 4
        assert "hellaswag" in report["registered_and_in_lm_eval"]
        assert "gsm8k" in report["registered_and_in_lm_eval"]
        unregistered_names = {e["task_name"] for e in report["unregistered"]}
        assert "fake_new_task_a" in unregistered_names
        assert "mmlu_fake_subject" in unregistered_names

    def test_filter_narrows_unregistered(self, discovery_module, monkeypatch):
        _install_mock_lm_eval(monkeypatch, [
            "fake_new_task_a", "mmlu_fake_subject",
        ])
        report = discovery_module.build_report(
            filter_pattern="mmlu_*",
            deep=False,
            verbose=False,
        )
        names = {e["task_name"] for e in report["unregistered"]}
        assert names == {"mmlu_fake_subject"}
        assert report["filter_matched"] == 1

    def test_missing_from_lm_eval(self, discovery_module, monkeypatch):
        # lm-eval reports only one task; KNOWN_TASK_DOMAINS has many more.
        _install_mock_lm_eval(monkeypatch, ["hellaswag"])
        report = discovery_module.build_report(
            filter_pattern="*",
            deep=False,
            verbose=False,
        )
        # Every registered task other than hellaswag should be flagged missing.
        from lmdiff.probes.adapters import KNOWN_TASK_DOMAINS
        expected_missing = set(KNOWN_TASK_DOMAINS.keys()) - {"hellaswag"}
        assert set(report["missing_from_lm_eval"]) == expected_missing


class TestMainCLI:
    def test_main_shallow_writes_json(self, discovery_module, monkeypatch, tmp_path, capsys):
        _install_mock_lm_eval(monkeypatch, [
            "hellaswag", "fake_new_task_a",
        ])
        out = tmp_path / "discovery.json"
        rc = discovery_module.main([
            "--filter", "*",
            "--output-json", str(out),
        ])
        assert rc == 0
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "lm_eval_version" in data
        assert "unregistered" in data
        assert any(
            e["task_name"] == "fake_new_task_a" for e in data["unregistered"]
        )
        captured = capsys.readouterr()
        assert "lm-eval task discovery" in captured.out

    def test_generate_entries_emits_parseable_lines(self, discovery_module, monkeypatch, capsys):
        _install_mock_lm_eval(monkeypatch, [
            "mmlu_fake_subject", "fake_math_bench",
        ])
        rc = discovery_module.main(["--filter", "*", "--generate-entries"])
        assert rc == 0
        out = capsys.readouterr().out
        # Extract lines that look like TaskInfo entries.
        lines = [
            line.strip() for line in out.splitlines()
            if re.match(r'^\s*"[^"]+":\s*TaskInfo\(', line)
        ]
        assert len(lines) >= 2
        # Each line should parse as valid Python if wrapped in a dict literal.
        # The lines end with "# comment" — ast parses through trailing comments.
        snippet = "{\n" + "\n".join(lines) + "\n}"
        # TaskInfo is not defined in the snippet's scope; use ast.parse which
        # verifies syntax without executing.
        ast.parse(snippet, mode="eval")

    def test_help_exits_zero(self, discovery_module, capsys):
        with pytest.raises(SystemExit) as exc_info:
            discovery_module.main(["--help"])
        assert exc_info.value.code == 0
