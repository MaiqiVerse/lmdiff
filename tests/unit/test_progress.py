"""Unit tests for the v0.3.2-progress helper.

The progress helper sits in front of long-running engine loops in
``lmdiff/engine.py`` and ``lmdiff/geometry.py``. The contract:

  - ``enable=False`` is silent — never imports ``rich``, never writes
    to stdout, behaves like a plain ``yield from`` over the iterable.
  - ``enable=True`` (or auto-on with a tty) yields the same items in
    the same order — the bar wraps but does not modify them.
  - ``LMDIFF_PROGRESS=0`` / ``=1`` overrides the auto-detect default.
  - ``device_map_summary`` returns None for a non-sharded model and a
    short string when a model is sharded across multiple devices.

These tests do NOT exercise rich's rendering — that's rich's job. They
only verify the public contract of the wrapper.
"""
from __future__ import annotations

import io
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lmdiff._progress import device_map_summary, iterate, phase


class TestIterateContract:
    def test_silent_when_disabled_yields_in_order(self, capsys):
        out = list(iterate([1, 2, 3], desc="x", enable=False))
        assert out == [1, 2, 3]
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_silent_passes_through_generator(self):
        def gen():
            yield from "abc"

        out = list(iterate(gen(), desc="x", enable=False))
        assert out == ["a", "b", "c"]

    def test_enable_true_yields_same_items(self):
        # Even with the bar enabled, the wrapper must not drop or reorder
        # items. Capture into list and compare.
        out = list(iterate(range(5), desc="generate", enable=True))
        assert out == [0, 1, 2, 3, 4]

    def test_env_var_overrides_auto(self, monkeypatch):
        # With the env var pinned to 0 and no explicit enable, progress
        # is silent regardless of tty.
        monkeypatch.setenv("LMDIFF_PROGRESS", "0")
        # Make stdout look like a tty — env should still win.
        out = io.StringIO()
        out.isatty = lambda: True  # type: ignore[method-assign]
        with patch.object(sys, "stdout", out):
            list(iterate([1], desc="x"))
        assert out.getvalue() == ""

    def test_env_var_can_force_on(self, monkeypatch):
        monkeypatch.setenv("LMDIFF_PROGRESS", "1")
        # Even with stdout NOT a tty (StringIO returns False), the env
        # var forces progress on, so we'd render. We don't assert the
        # *content* (rich-specific) — only that the iterator still
        # yields the right items.
        out = list(iterate([10, 20], desc="x"))
        assert out == [10, 20]

    def test_default_silent_when_not_a_tty(self, monkeypatch, capsys):
        monkeypatch.delenv("LMDIFF_PROGRESS", raising=False)
        # capsys's stdout is a file-like, not a tty.
        out = list(iterate([1, 2], desc="x"))
        assert out == [1, 2]
        captured = capsys.readouterr()
        assert captured.out == ""


class TestPhaseContext:
    def test_silent_when_disabled(self, capsys):
        with phase("loading model", enable=False):
            pass
        assert capsys.readouterr().out == ""

    def test_prints_when_enabled(self, capsys):
        with phase("loading model", enable=True):
            pass
        out = capsys.readouterr().out
        # Format: "[HH:MM:SS] loading model ...\n[HH:MM:SS] loading model done in ..."
        assert "loading model ..." in out
        assert "loading model done in" in out

    def test_prints_even_on_exception(self, capsys):
        with pytest.raises(RuntimeError):
            with phase("crashy work", enable=True):
                raise RuntimeError("boom")
        out = capsys.readouterr().out
        # The "done in" line still fires (try/finally).
        assert "crashy work ..." in out
        assert "crashy work done in" in out


class TestDeviceMapSummary:
    def test_none_when_no_device_map_attr(self):
        model = SimpleNamespace()
        assert device_map_summary(model) is None

    def test_none_when_empty_device_map(self):
        model = SimpleNamespace(hf_device_map={})
        assert device_map_summary(model) is None

    def test_none_when_single_device(self):
        # All layers on cuda:0 — no shard, no warning.
        model = SimpleNamespace(hf_device_map={
            "model.embed_tokens": "cuda:0",
            "model.layers.0": "cuda:0",
            "model.layers.1": "cuda:0",
            "lm_head": "cuda:0",
        })
        assert device_map_summary(model) is None

    def test_warns_when_sharded_cpu_and_cuda(self):
        # The exact failure mode that wasted 8h of wall-clock time:
        # accelerate spilled some layers to CPU, forward runs partly on
        # CPU at 0% GPU util. We want this surfaced before run start.
        model = SimpleNamespace(hf_device_map={
            "model.embed_tokens": "cuda:0",
            "model.layers.0": "cuda:0",
            "model.layers.1": "cuda:0",
            "model.layers.30": "cpu",
            "model.layers.31": "cpu",
            "lm_head": "cpu",
        })
        msg = device_map_summary(model)
        assert msg is not None
        assert "cpu" in msg
        assert "cuda:0" in msg
        assert "sharded" in msg.lower()

    def test_handles_non_string_device_keys(self):
        # accelerate sometimes uses torch.device objects; str() coerces.
        class FakeDev:
            def __init__(self, name): self.name = name
            def __str__(self): return self.name

        model = SimpleNamespace(hf_device_map={
            "a": FakeDev("cuda:0"),
            "b": FakeDev("cpu"),
        })
        msg = device_map_summary(model)
        assert msg is not None
        assert "cuda:0" in msg and "cpu" in msg


class TestEngineProgressKwarg:
    """Smoke: the new ``progress=`` kwargs on InferenceEngine.generate /
    .score don't break the existing call shape. We don't actually load
    a model here — we just confirm the signatures accept the new kwarg
    via inspect."""

    def test_inference_engine_generate_accepts_progress_kwarg(self):
        import inspect
        from lmdiff.engine import InferenceEngine

        sig = inspect.signature(InferenceEngine.generate)
        assert "progress" in sig.parameters
        assert "progress_desc" in sig.parameters

    def test_inference_engine_score_accepts_progress_kwarg(self):
        import inspect
        from lmdiff.engine import InferenceEngine

        sig = inspect.signature(InferenceEngine.score)
        assert "progress" in sig.parameters
        assert "progress_desc" in sig.parameters

    def test_change_geometry_analyze_accepts_progress_kwarg(self):
        import inspect
        from lmdiff.geometry import ChangeGeometry

        sig = inspect.signature(ChangeGeometry.analyze)
        assert "progress" in sig.parameters

    def test_compare_and_family_accept_progress_kwarg(self):
        import inspect
        from lmdiff._api import compare, family

        assert "progress" in inspect.signature(compare).parameters
        assert "progress" in inspect.signature(family).parameters
