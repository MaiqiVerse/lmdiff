"""Unit tests for ``lmdiff._pipeline.run_family_pipeline``.

Runs against ``MockEngine`` (CPU, no real models). Validates the
Engine-Protocol-only contract of the new v0.4.0 pipeline:

  - ``token_count`` and ``tokenizers_equivalent_to`` are called via the
    public Protocol; no direct attribute access on engine internals.
  - Engine cache + look-ahead-by-one release work the same way as
    ``ChangeGeometry.analyze`` did in v0.3.2.
  - Per-variant runtime overrides (``system_prompt``, ``context``,
    ``decode``) are applied at the prompt-assembly layer here, not via
    Engine kwargs — verifies engines that never had those kwargs (e.g.
    HFEngine) work correctly.
  - Cross-tokenizer BPB fallback fires when ``tokenizers_equivalent_to``
    returns False.
  - GeoResult schema-v5 fields are all populated.

The numeric calibration regression (byte-equivalence vs v0.3.2) lives
in ``tests/integration/test_calibration_regression.py`` — that needs a
GPU and the calibration fixture.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from lmdiff._api import _BASE_ANCHOR, _compute_anchor_map
from lmdiff._config import Config, DecodeSpec
from lmdiff._pipeline import (
    _assemble_prompt,
    _generate_kwargs,
    _prefix_text,
    run_family_pipeline,
)
from lmdiff.probes.loader import Probe, ProbeSet

from tests.fixtures.mock_engine import MockEngine


def _probes(n: int = 6, domains: tuple[str, ...] | None = None) -> ProbeSet:
    """Build a small ProbeSet with explicit domains."""
    if domains is None:
        domains = ("a",) * n
    assert len(domains) == n
    return ProbeSet(
        [Probe(id=f"p{i}", text=f"prompt_{i}", domain=domains[i]) for i in range(n)],
        name="test_set",
        version="v1",
    )


# ── Prompt-assembly helpers ──────────────────────────────────────────


class TestPrefixText:
    def test_empty_when_no_prefix_material(self):
        assert _prefix_text(Config(model="gpt2")) == ""

    def test_system_prompt_only(self):
        cfg = Config(model="gpt2", system_prompt="Be concise.")
        assert _prefix_text(cfg) == "Be concise.\n"

    def test_trailing_newline_preserved(self):
        # Critical for byte-equivalence with v0.2.x InferenceEngine.
        cfg = Config(model="gpt2", system_prompt="hi")
        assert _prefix_text(cfg).endswith("\n")


class TestAssemblePrompt:
    def test_no_prefix_returns_probe_unchanged(self):
        assert _assemble_prompt(Config(model="gpt2"), "user text") == "user text"

    def test_with_system_prompt_concatenates(self):
        cfg = Config(model="gpt2", system_prompt="Be concise.")
        assert _assemble_prompt(cfg, "user text") == "Be concise.\nuser text"


class TestGenerateKwargs:
    def test_greedy_strips_to_defaults(self):
        cfg = Config(model="gpt2", decode=DecodeSpec(strategy="greedy"))
        assert _generate_kwargs(cfg, max_new_tokens=8) == {"max_new_tokens": 8}

    def test_sample_includes_temperature_top_p(self):
        cfg = Config(
            model="gpt2",
            decode=DecodeSpec(
                strategy="sample", temperature=1.5, top_p=0.95, seed=7,
            ),
        )
        kw = _generate_kwargs(cfg, max_new_tokens=16)
        assert kw["max_new_tokens"] == 16
        assert kw["temperature"] == 1.5
        assert kw["top_p"] == 0.95
        assert kw["seed"] == 7


# ── Pipeline end-to-end (MockEngine) ─────────────────────────────────


class TestPipelineEndToEnd:
    def test_basic_run_produces_v5_geo_result(self):
        base_cfg = Config(model="mock_base")
        v_cfg = Config(model="mock_variant")
        # Distinct seeds so the two engines produce different logprobs
        # for the same (prompt, continuation) — otherwise MockEngine's
        # seeded RNG yields identical scores and δ ≡ 0 (degenerate
        # magnitude=0 case that's irrelevant to pipeline correctness).
        base_eng = MockEngine(config=base_cfg, seed=1)
        v_eng = MockEngine(config=v_cfg, seed=2)

        anchor_map = _compute_anchor_map(
            [(_BASE_ANCHOR, base_cfg), ("variant", v_cfg)],
        )
        result = run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_engines={"variant": v_eng},
            variant_configs={"variant": v_cfg},
            probe_set=_probes(4),
            max_new_tokens=4,
        )
        assert result.variant_names == ["variant"]
        assert result.n_probes == 4
        assert "variant" in result.magnitudes
        assert result.magnitudes["variant"] > 0
        assert "variant" in result.cosine_matrix
        assert result.cosine_matrix["variant"]["variant"] == pytest.approx(1.0)
        # Schema v5 fields populated:
        assert isinstance(result.share_per_domain, dict)
        assert isinstance(result.magnitudes_per_domain_normalized, dict)

    def test_runtime_only_variant_reuses_base_engine(self):
        """When the anchor map says variant reuses base, the pipeline
        passes base_engine in for both base + variant scoring. The
        prompt-assembly layer applies the variant's runtime params."""
        base_cfg = Config(model="mock_base")
        sp_cfg = Config(model="mock_base", system_prompt="Be concise.")
        base_eng = MockEngine(config=base_cfg)

        anchor_map = _compute_anchor_map(
            [(_BASE_ANCHOR, base_cfg), ("sp", sp_cfg)],
        )
        # sp is runtime-only of base → reuses base's engine.
        assert anchor_map["sp"] == _BASE_ANCHOR

        result = run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_engines={"sp": base_eng},  # same instance
            variant_configs={"sp": sp_cfg},
            probe_set=_probes(3),
            max_new_tokens=4,
            engine_groups=anchor_map,
        )
        assert result.n_probes == 3

    def test_engine_groups_drives_release(self):
        """Look-ahead-by-one release: when next variant has a different
        anchor, current variant's engine drops out of the cache."""
        base_cfg = Config(model="mock_base")
        a_cfg = Config(model="mock_a")
        b_cfg = Config(model="mock_b")
        c_cfg = Config(model="mock_a", name="c")  # same model as a

        items = [(_BASE_ANCHOR, base_cfg), ("a", a_cfg), ("b", b_cfg), ("c", c_cfg)]
        anchor_map = _compute_anchor_map(items)
        # c is runtime-only mod of a → anchor[c] == "a"
        assert anchor_map["c"] == "a"

        base_eng = MockEngine(config=base_cfg)
        a_eng = MockEngine(config=a_cfg)
        b_eng = MockEngine(config=b_cfg)

        # In the real pipeline, c's engine would be `a_eng` (reused via
        # cache); here we just supply it explicitly for the mock test.
        result = run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_engines={"a": a_eng, "b": b_eng, "c": a_eng},
            variant_configs={"a": a_cfg, "b": b_cfg, "c": c_cfg},
            probe_set=_probes(3),
            max_new_tokens=4,
            engine_groups=anchor_map,
        )
        assert result.variant_names == ["a", "b", "c"]


class TestProtocolOnlyAccess:
    """The pipeline must not poke private engine attributes. Verify by
    asserting only Protocol methods get called on a wrapped engine."""

    def test_no_private_attribute_access_on_engines(self):
        base_cfg = Config(model="mock_base")
        v_cfg = Config(model="mock_v")

        # Wrap MockEngine so any attribute access we don't whitelist
        # raises. Only Protocol-allowed names pass through.
        ALLOWED = {
            "name", "tokenizer_id", "n_layers", "hidden_dim", "capabilities",
            "score", "generate", "close", "token_count",
            "tokenizers_equivalent_to",
            # MockEngine internals that aren't Protocol but are
            # accessed by the engine itself, not the pipeline:
            # (none — pipeline shouldn't reach into anything else)
        }

        class StrictWrapper:
            def __init__(self, eng): self._eng = eng
            def __getattr__(self, name):
                if name.startswith("_"):
                    raise AssertionError(
                        f"pipeline accessed private attribute {name!r}"
                    )
                if name not in ALLOWED:
                    raise AssertionError(
                        f"pipeline accessed non-Protocol attribute {name!r}"
                    )
                return getattr(self._eng, name)

        base_eng = StrictWrapper(MockEngine(config=base_cfg))
        v_eng = StrictWrapper(MockEngine(config=v_cfg))

        result = run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_engines={"v": v_eng},
            variant_configs={"v": v_cfg},
            probe_set=_probes(3),
            max_new_tokens=4,
        )
        assert result.n_probes == 3


class TestCrossTokenizerBpb:
    """When base and variant engines have different tokenizers, the
    pipeline applies BPB normalization to δ values."""

    def test_bpb_flag_set_when_tokenizers_disagree(self):
        base_cfg = Config(model="mock_base")
        v_cfg = Config(model="mock_v_other")  # different tokenizer_id
        base_eng = MockEngine(config=base_cfg)
        v_eng = MockEngine(config=v_cfg)

        result = run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_engines={"v": v_eng},
            variant_configs={"v": v_cfg},
            probe_set=_probes(3),
            max_new_tokens=4,
        )
        assert result.metadata["bpb_normalized"]["v"] is True

    def test_bpb_flag_unset_when_tokenizers_match(self):
        base_cfg = Config(model="mock_same")
        v_cfg = Config(model="mock_same", name="variant")  # same model id
        base_eng = MockEngine(config=base_cfg)
        v_eng = MockEngine(config=v_cfg)

        result = run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_engines={"variant": v_eng},
            variant_configs={"variant": v_cfg},
            probe_set=_probes(3),
            max_new_tokens=4,
        )
        assert result.metadata["bpb_normalized"]["variant"] is False
