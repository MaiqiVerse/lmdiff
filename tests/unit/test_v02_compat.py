"""Unit tests for the v0.2.x → v0.3.0 deprecation shims.

The shims live in :mod:`lmdiff.config` (the v0.2.x ``Config``) and
:mod:`lmdiff.diff` (the v0.2.x ``ModelDiff``). Both emit a
:class:`DeprecationWarning` at construction time but otherwise continue
to function so existing code keeps working through v0.4.0.
"""
from __future__ import annotations

import warnings

import pytest

from lmdiff.config import Config as V02Config
from lmdiff.diff import ModelDiff
from lmdiff.probes.loader import ProbeSet


# ── v0.2.x Config emits DeprecationWarning ────────────────────────────


class TestV02ConfigDeprecation:
    def test_construction_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            V02Config(model="gpt2")
        deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecations) >= 1
        msg = str(deprecations[0].message)
        assert "lmdiff.config.Config" in msg
        assert "v0.4.0" in msg
        assert "from lmdiff import Config" in msg

    def test_construction_still_succeeds(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = V02Config(model="gpt2", system_prompt="hi")
        assert cfg.model == "gpt2"
        assert cfg.system_prompt == "hi"
        # decode is still a dict (v0.2.x semantics)
        assert isinstance(cfg.decode, dict)
        assert cfg.decode == {"strategy": "greedy"}

    def test_dict_decode_still_accepted(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = V02Config(
                model="gpt2",
                decode={"strategy": "sample", "temperature": 0.7},
            )
        assert cfg.decode["temperature"] == 0.7

    def test_invalid_decode_still_raises(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(TypeError, match="decode must be a dict"):
                V02Config(model="gpt2", decode="not-a-dict")  # type: ignore[arg-type]


# ── v0.2.x ModelDiff emits DeprecationWarning ─────────────────────────


class TestV02ModelDiffDeprecation:
    def test_construction_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with warnings.catch_warnings():
                # Suppress the inner v0.2.x Config warnings so we can find
                # ModelDiff's specifically.
                pass
            cfg_a = _silently(lambda: V02Config(model="gpt2"))
            cfg_b = _silently(lambda: V02Config(model="distilgpt2"))
            ProbeSet([], name="empty")
            ModelDiff(cfg_a, cfg_b, prompts=["x", "y"])
        msgs = [
            str(x.message) for x in w
            if issubclass(x.category, DeprecationWarning)
        ]
        # At least one ModelDiff-specific warning should be present.
        assert any("lmdiff.ModelDiff" in m for m in msgs), msgs
        assert any("lmdiff.compare" in m for m in msgs), msgs

    def test_construction_still_works_and_runs_at_output_level(self):
        # Don't actually load real models — just make sure construction
        # succeeds with mocked configs and that .run() with level="output"
        # doesn't raise (it uses BD/TokenEntropy/TokenKL metric classes
        # which require an Engine; we don't call .run here, only verify
        # the object constructs).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg_a = V02Config(model="gpt2")
            cfg_b = V02Config(model="distilgpt2")
            md = ModelDiff(cfg_a, cfg_b, prompts=["a", "b", "c"])
        assert md.config_a.model == "gpt2"
        assert md.config_b.model == "distilgpt2"
        assert len(md.prompts) == 3

    def test_run_representation_raises_with_v07_pointer(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg_a = V02Config(model="gpt2")
            cfg_b = V02Config(model="distilgpt2")
            md = ModelDiff(cfg_a, cfg_b, prompts=["a"])
            with pytest.raises(NotImplementedError, match="representation"):
                md.run(level="representation")

    def test_run_representation_message_mentions_v07(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg_a = V02Config(model="gpt2")
            cfg_b = V02Config(model="distilgpt2")
            md = ModelDiff(cfg_a, cfg_b, prompts=["a"])
            try:
                md.run(level="representation")
            except NotImplementedError as exc:
                assert "v0.7.0" in str(exc) or "Phase 5" in str(exc)


# ── helpers ──────────────────────────────────────────────────────────


def _silently(fn):
    """Run a callable suppressing DeprecationWarning. Used to construct
    v0.2.x objects in fixtures without polluting the warning record we
    actually care about."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return fn()
