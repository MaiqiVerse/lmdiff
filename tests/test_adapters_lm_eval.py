"""Tests for lmdiff.probes.adapters.from_lm_eval.

Every test mocks the `lm_eval` module — no real task downloads. The
module-under-test imports `lm_eval` lazily inside the function, so
installing a MagicMock into sys.modules before import suffices.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from lmdiff.probes.adapters import (
    KNOWN_TASK_DOMAINS,
    TaskInfo,
    from_lm_eval,
)
from lmdiff.probes.loader import Probe, ProbeSet


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_fake_task(
    *,
    output_type: str,
    docs: list[dict],
    doc_to_text_fn=None,
    doc_to_target_fn=None,
    doc_to_choice_fn=None,
    fewshot_fn=None,
):
    """Return an object that quacks like an lm_eval task."""
    task = MagicMock(name=f"task-{output_type}")
    task.OUTPUT_TYPE = output_type
    # Materialize via test_docs() first, then validation, then training.
    task.test_docs.return_value = list(docs)
    task.validation_docs.return_value = []
    task.training_docs.return_value = []
    task.doc_to_text.side_effect = doc_to_text_fn or (
        lambda doc: doc.get("text", f"Q:{doc.get('id', '?')}")
    )
    task.doc_to_target.side_effect = doc_to_target_fn or (
        lambda doc: doc.get("target", "unknown")
    )
    if doc_to_choice_fn is not None:
        task.doc_to_choice.side_effect = doc_to_choice_fn
    else:
        # Ensure doc_to_choice isn't auto-created as MagicMock that returns
        # another MagicMock. Make it raise so _render_target falls through
        # to the dict-lookup branch.
        task.doc_to_choice.side_effect = AttributeError("no choices")
    if fewshot_fn is not None:
        task.fewshot_context.side_effect = fewshot_fn
    task.VERSION = "1.0"
    return task


def _install_mock_lm_eval(monkeypatch, task_dict_by_name: dict[str, object]):
    """Install a fake `lm_eval.tasks` with get_task_dict returning our mocks."""
    fake_tasks = types.SimpleNamespace()

    def get_task_dict(names):
        result = {}
        for n in names:
            if n in task_dict_by_name:
                result[n] = task_dict_by_name[n]
            else:
                raise KeyError(n)
        return result

    fake_tasks.get_task_dict = get_task_dict
    fake_module = types.SimpleNamespace(tasks=fake_tasks)
    monkeypatch.setitem(sys.modules, "lm_eval", fake_module)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", fake_tasks)


# ── ImportError path ───────────────────────────────────────────────────

class TestImportError:
    def test_missing_lm_eval_raises_with_install_hint(self, monkeypatch):
        # Force the import to fail deterministically.
        monkeypatch.delitem(sys.modules, "lm_eval", raising=False)
        monkeypatch.delitem(sys.modules, "lm_eval.tasks", raising=False)

        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "lm_eval" or name.startswith("lm_eval."):
                raise ImportError("no module named 'lm_eval'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match=r"pip install lmdiff-kit\[lm-eval\]"):
            from_lm_eval("hellaswag")


# ── KeyError path ──────────────────────────────────────────────────────

class TestKeyError:
    def test_unknown_task_name_raises(self, monkeypatch):
        _install_mock_lm_eval(monkeypatch, {})  # empty registry
        with pytest.raises(KeyError, match="not found in lm-eval task registry"):
            from_lm_eval("definitely_not_a_real_task_name")


# ── Unsupported output_type path ───────────────────────────────────────

class TestUnsupportedOutputType:
    def test_chat_output_type_raises(self, monkeypatch):
        chat_task = _make_fake_task(output_type="chat", docs=[{"id": "a"}])
        _install_mock_lm_eval(monkeypatch, {"madeup_chat_task": chat_task})
        with pytest.raises(NotImplementedError, match="output_type 'chat'"):
            from_lm_eval("madeup_chat_task")


# ── Multiple-choice happy path ─────────────────────────────────────────

class TestMultipleChoicePath:
    def _make_task(self):
        docs = [
            {"id": "d0", "text": "Q0?", "target": 1, "choices": ["A", "B", "C"]},
            {"id": "d1", "text": "Q1?", "target": 0, "choices": ["A", "B"]},
            {"id": "d2", "text": "Q2?", "target": 2, "choices": ["X", "Y", "Z"]},
        ]
        return _make_fake_task(
            output_type="multiple_choice",
            docs=docs,
            doc_to_choice_fn=lambda doc: doc["choices"],
        )

    def test_limit_truncates(self, monkeypatch):
        _install_mock_lm_eval(monkeypatch, {"hellaswag": self._make_task()})
        result = from_lm_eval("hellaswag", limit=2)
        assert isinstance(result, ProbeSet)
        assert len(result) == 2

    def test_metadata_fields_populated(self, monkeypatch):
        _install_mock_lm_eval(monkeypatch, {"hellaswag": self._make_task()})
        result = from_lm_eval("hellaswag", limit=3)
        probe = result[0]
        assert isinstance(probe, Probe)
        assert probe.metadata["task_name"] == "hellaswag"
        assert probe.metadata["native_metric"] == "acc_norm"
        assert probe.metadata["output_type"] == "multiple_choice"
        assert probe.metadata["requires_execution"] is False
        assert "doc_idx" in probe.metadata

    def test_domain_resolved_from_known_registry(self, monkeypatch):
        _install_mock_lm_eval(monkeypatch, {"hellaswag": self._make_task()})
        result = from_lm_eval("hellaswag", limit=1)
        assert result[0].domain == "commonsense"

    def test_multi_choice_target_becomes_choice_text(self, monkeypatch):
        _install_mock_lm_eval(monkeypatch, {"hellaswag": self._make_task()})
        result = from_lm_eval("hellaswag", limit=3)
        # Shuffle randomizes which original doc ends up at doc_idx=k. For
        # each probe, verify expected ∈ allowed choice-at-target set across
        # the three original docs. Also verify every probe's expected comes
        # from a chosen choice (not the raw int index).
        allowed = {"B", "A", "Z"}  # docs[0]→"B", docs[1]→"A", docs[2]→"Z"
        observed = {p.expected for p in result}
        assert observed == allowed


# ── generate_until happy path (gsm8k) ──────────────────────────────────

class TestGenerateUntilPath:
    def test_string_target(self, monkeypatch):
        task = _make_fake_task(
            output_type="generate_until",
            docs=[{"id": "g0", "text": "What is 17+25?", "target": "#### 42"}],
        )
        _install_mock_lm_eval(monkeypatch, {"gsm8k": task})
        result = from_lm_eval("gsm8k", limit=1)
        probe = result[0]
        assert probe.expected == "#### 42"
        assert probe.metadata["native_metric"] == "exact_match"
        assert probe.metadata["output_type"] == "generate_until"
        assert "aliases" not in probe.metadata

    def test_list_target_populates_aliases(self, monkeypatch):
        task = _make_fake_task(
            output_type="generate_until",
            docs=[{"id": "q0", "text": "Capital?", "target": ["Paris", "paris", "PARIS"]}],
        )
        _install_mock_lm_eval(monkeypatch, {"triviaqa": task})
        result = from_lm_eval("triviaqa", limit=1)
        probe = result[0]
        assert probe.expected == "Paris"
        assert probe.metadata["aliases"] == ["paris", "PARIS"]


# ── limit + seed determinism ───────────────────────────────────────────

class TestLimitSeedDeterminism:
    def _make_task_with_10_docs(self):
        docs = [
            {"id": f"d{i}", "text": f"Q{i}?", "target": f"A{i}"}
            for i in range(10)
        ]
        return _make_fake_task(output_type="generate_until", docs=docs)

    def test_same_seed_same_order(self, monkeypatch):
        _install_mock_lm_eval(monkeypatch, {"gsm8k": self._make_task_with_10_docs()})
        r1 = from_lm_eval("gsm8k", limit=5, seed=42)
        # second call — re-install because MagicMock sides are consumed
        _install_mock_lm_eval(monkeypatch, {"gsm8k": self._make_task_with_10_docs()})
        r2 = from_lm_eval("gsm8k", limit=5, seed=42)
        assert [p.text for p in r1] == [p.text for p in r2]
        assert [p.expected for p in r1] == [p.expected for p in r2]

    def test_different_seeds_different_order(self, monkeypatch):
        _install_mock_lm_eval(monkeypatch, {"gsm8k": self._make_task_with_10_docs()})
        r1 = from_lm_eval("gsm8k", limit=10, seed=42)
        _install_mock_lm_eval(monkeypatch, {"gsm8k": self._make_task_with_10_docs()})
        r2 = from_lm_eval("gsm8k", limit=10, seed=99)
        # Full permutation across 10 items under different seeds is
        # overwhelmingly likely to differ.
        assert [p.text for p in r1] != [p.text for p in r2]


# ── Domain fallback chain ──────────────────────────────────────────────

class TestDomainFallback:
    def test_prefix_fallback_to_mmlu(self, monkeypatch):
        # A made-up mmlu subset NOT pre-registered in KNOWN_TASK_DOMAINS.
        task = _make_fake_task(
            output_type="multiple_choice",
            docs=[{"id": "x", "text": "Q?", "target": 0, "choices": ["A", "B"]}],
            doc_to_choice_fn=lambda doc: doc["choices"],
        )
        fake_name = "mmlu_some_unregistered_subject"
        assert fake_name not in KNOWN_TASK_DOMAINS
        _install_mock_lm_eval(monkeypatch, {fake_name: task})
        result = from_lm_eval(fake_name, limit=1)
        # Falls back through prefix 'mmlu' → domain "knowledge"
        assert result[0].domain == "knowledge"
        assert result[0].metadata["native_metric"] is None  # no direct registry hit
        assert result[0].metadata["requires_execution"] is False

    def test_fully_unknown_task(self, monkeypatch):
        task = _make_fake_task(
            output_type="generate_until",
            docs=[{"id": "u", "text": "Q?", "target": "A"}],
        )
        name = "utterly_novel_benchmark_xyz"
        _install_mock_lm_eval(monkeypatch, {name: task})
        result = from_lm_eval(name, limit=1)
        assert result[0].domain == "unknown"
        assert result[0].metadata["native_metric"] is None


# ── Registry inspection ────────────────────────────────────────────────

class TestRegistryInvariants:
    def test_registry_size(self):
        assert len(KNOWN_TASK_DOMAINS) >= 28

    def test_registry_covers_expected_domains(self):
        domains = {info.domain for info in KNOWN_TASK_DOMAINS.values()}
        expected = {
            "commonsense", "reasoning", "math", "knowledge", "code",
            "reading", "language", "long-context", "safety",
        }
        assert expected.issubset(domains)

    def test_mmlu_college_cs_domain_is_code(self):
        info = KNOWN_TASK_DOMAINS["mmlu_college_computer_science"]
        assert info.domain == "code"

    def test_humaneval_requires_execution(self):
        info = KNOWN_TASK_DOMAINS["humaneval"]
        assert info.requires_execution is True

    def test_mbpp_requires_execution(self):
        info = KNOWN_TASK_DOMAINS["mbpp"]
        assert info.requires_execution is True

    def test_taskinfo_is_frozen(self):
        info = KNOWN_TASK_DOMAINS["hellaswag"]
        with pytest.raises((AttributeError, Exception)):
            info.domain = "other"  # type: ignore[misc]


# ── Probe contract from adapter ────────────────────────────────────────

class TestProbeShape:
    def test_probe_text_field_populated(self, monkeypatch):
        task = _make_fake_task(
            output_type="generate_until",
            docs=[{"id": "g", "text": "Prompt here", "target": "Answer"}],
        )
        _install_mock_lm_eval(monkeypatch, {"gsm8k": task})
        result = from_lm_eval("gsm8k", limit=1)
        probe = result[0]
        # Instruction correction 1: probe.text (not probe.prompt)
        assert probe.text == "Prompt here"
        # Instruction correction 1: domain at top level
        assert probe.domain == "math"
        # Instruction correction 1: metadata does NOT contain 'domain'
        assert "domain" not in probe.metadata

    def test_probe_id_is_task_and_idx(self, monkeypatch):
        task = _make_fake_task(
            output_type="generate_until",
            docs=[{"id": "anything", "text": "q", "target": "a"}],
        )
        _install_mock_lm_eval(monkeypatch, {"gsm8k": task})
        result = from_lm_eval("gsm8k", limit=1)
        assert result[0].id.startswith("gsm8k:")

    def test_probeset_name_and_version(self, monkeypatch):
        task = _make_fake_task(
            output_type="generate_until",
            docs=[{"id": "q", "text": "q", "target": "a"}],
        )
        _install_mock_lm_eval(monkeypatch, {"gsm8k": task})
        result = from_lm_eval("gsm8k", limit=1)
        assert result.name == "lm_eval:gsm8k"
        assert result.version == "1.0"  # task.VERSION
