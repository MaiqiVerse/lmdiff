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

    def test_sample_does_not_include_seed(self):
        """Fix 3 contract — seed is applied once-per-variant by
        ``_delta_for_variant`` (probe 0 only), NOT per-probe via
        gen_kwargs. Putting seed in gen_kwargs would reset RNG
        on every probe and force every probe in a sampling variant
        to see the same RNG state — wrong granularity. The
        end-to-end "seed reaches engine.generate" contract is in
        ``test_seed_plumbing.py``."""
        cfg = Config(
            model="gpt2",
            decode=DecodeSpec(strategy="sample", temperature=1.5, seed=7),
        )
        kw = _generate_kwargs(cfg, max_new_tokens=16)
        assert "seed" not in kw, (
            "seed must not be in gen_kwargs (would reset RNG per probe); "
            "it is plumbed via _delta_for_variant's seed= kwarg instead"
        )

    def test_sample_passes_top_k_explicitly(self):
        """Fix 1 regression test — caught the temp_1.5 share collapse
        from 34% → 5% on the GPU 7-variant demo. HF's model.generate
        defaults top_k=50 when omitted, silently truncating the
        sample distribution. _generate_kwargs MUST forward
        DecodeSpec.top_k so the engine passes 0 (no filtering) for
        configs that don't set top_k explicitly."""
        cfg = Config(
            model="gpt2",
            decode=DecodeSpec(strategy="sample", temperature=1.5),  # top_k defaults to 0
        )
        kw = _generate_kwargs(cfg, max_new_tokens=16)
        assert kw["top_k"] == 0, (
            "top_k must be explicitly forwarded — without it, HF's "
            "model.generate defaults to top_k=50 and sample-decode "
            "outputs diverge from v0.3.2"
        )

    def test_sample_passes_user_specified_top_k(self):
        cfg = Config(
            model="gpt2",
            decode=DecodeSpec(
                strategy="sample", temperature=0.7, top_k=40,
            ),
        )
        kw = _generate_kwargs(cfg, max_new_tokens=16)
        assert kw["top_k"] == 40

    def test_greedy_does_not_pass_top_k(self):
        # Greedy decoding doesn't sample, top_k irrelevant. Don't
        # pollute the kwargs dict with it.
        cfg = Config(model="gpt2", decode=DecodeSpec(strategy="greedy"))
        kw = _generate_kwargs(cfg, max_new_tokens=8)
        assert "top_k" not in kw


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


class TestHFEngineSelfScoreSignature:
    """The pipeline calls ``v_engine.score(prompt, continuation_ids=…)``
    on the self-score path. HFEngine's validator requires *exactly one*
    of (continuation, continuation_ids) and raises ``ValueError`` if
    both are passed. This test simulates HFEngine's strict signature
    against MockEngine to catch the regression where ``continuation=""``
    was being passed alongside ``continuation_ids``.

    Without this test, the bug only surfaced on the GPU-only calibration
    test (where MockEngine isn't used)."""

    def test_pipeline_uses_continuation_ids_only_on_self_score(self):
        """Mirror HFEngine.score's strict signature: exactly one of
        (continuation, continuation_ids) must be non-None. If the
        pipeline regresses to passing both — even with continuation=""
        — this engine raises ValueError and the test fails loudly.

        Also asserts the self-score call shape: at least one call uses
        continuation_ids with continuation=None (HFEngine's preferred
        path; avoids decode→retokenize round-trip drift)."""
        from lmdiff._engine import ScoreResult

        recorded_calls: list[dict] = []

        class StrictScoreEngine(MockEngine):
            def score(
                self,
                prompt: str,
                continuation=None,
                *,
                continuation_ids=None,
                prefix_text: str = "",
            ) -> ScoreResult:
                recorded_calls.append({
                    "prompt": prompt,
                    "continuation": continuation,
                    "continuation_ids": continuation_ids,
                    "prefix_text": prefix_text,
                })
                if (continuation is None) == (continuation_ids is None):
                    raise ValueError(
                        "pass exactly one of `continuation` or `continuation_ids`",
                    )
                # Delegate to the parent's text-based score for the math.
                if continuation is None:
                    return super().score(prompt, "self_score_stub")
                return super().score(prompt, continuation)

        base_cfg = Config(model="mock_base")
        v_cfg = Config(model="mock_variant")
        base_eng = StrictScoreEngine(config=base_cfg, seed=1)
        v_eng = StrictScoreEngine(config=v_cfg, seed=2)

        result = run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_engines={"v": v_eng},
            variant_configs={"v": v_cfg},
            probe_set=_probes(2),
            max_new_tokens=2,
        )
        assert result.n_probes == 2

        ids_only_calls = [
            c for c in recorded_calls
            if c["continuation"] is None and c["continuation_ids"] is not None
        ]
        assert len(ids_only_calls) > 0, (
            "expected at least one score(continuation_ids=…) call from the "
            "self-score path; recorded calls: " + repr(recorded_calls)
        )


class TestPrefixTextThreading:
    """Fix 2 regression tests — verify _prefix_text(v_config) reaches
    the engine via ``prefix_text=`` kwarg instead of being concatenated
    into the prompt. The GPU-only system_prompt variant share collapse
    (60% → 94%) was caused by single-tokenize boundary effects on
    Llama SentencePiece; engine-side split-tokenize is the fix."""

    def test_pipeline_passes_prefix_text_kwarg_to_engine(self):
        from lmdiff._engine import ScoreResult

        prefix_calls: list[str] = []

        class PrefixRecordingEngine(MockEngine):
            def score(
                self,
                prompt: str,
                continuation=None,
                *,
                continuation_ids=None,
                prefix_text: str = "",
            ) -> ScoreResult:
                prefix_calls.append(prefix_text)
                if continuation is None:
                    return super().score(prompt, "stub")
                return super().score(prompt, continuation)

            def generate(
                self,
                prompt: str,
                *,
                max_new_tokens: int = 16,
                temperature: float = 1.0,
                top_p: float = 1.0,
                top_k: int = 0,
                seed=None,
                prefix_text: str = "",
            ):
                prefix_calls.append(prefix_text)
                return super().generate(prompt, max_new_tokens=max_new_tokens)

        base_cfg = Config(model="mock_base")
        sp_cfg = Config(model="mock_sp",
                        system_prompt="You are concise.")
        base_eng = PrefixRecordingEngine(config=base_cfg, seed=1)
        sp_eng = PrefixRecordingEngine(config=sp_cfg, seed=2)

        run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_engines={"sp": sp_eng},
            variant_configs={"sp": sp_cfg},
            probe_set=_probes(2),
            max_new_tokens=4,
        )
        # Variant engine receives "You are concise.\n" — never empty.
        sp_prefixes = [p for p in prefix_calls if p == "You are concise.\n"]
        assert len(sp_prefixes) > 0, (
            f"variant engine never received system_prompt prefix; "
            f"prefix_text values seen: {prefix_calls!r}"
        )

    def test_pipeline_passes_empty_prefix_when_config_has_none(self):
        """Calibration variants (yarn/long/code/math/chat — no
        system_prompt, no context) must receive empty prefix_text so
        the engine's tokenization stays byte-identical to the v0.4.0-
        pre-fix path. Otherwise the calibration regression breaks."""
        from lmdiff._engine import ScoreResult

        prefix_calls: list[str] = []

        class PrefixRecordingEngine(MockEngine):
            def score(
                self,
                prompt: str,
                continuation=None,
                *,
                continuation_ids=None,
                prefix_text: str = "",
            ) -> ScoreResult:
                prefix_calls.append(prefix_text)
                if continuation is None:
                    return super().score(prompt, "stub")
                return super().score(prompt, continuation)

        base_cfg = Config(model="mock_base")
        v_cfg = Config(model="mock_v")  # no system_prompt, no context
        base_eng = PrefixRecordingEngine(config=base_cfg, seed=1)
        v_eng = PrefixRecordingEngine(config=v_cfg, seed=2)

        run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_engines={"v": v_eng},
            variant_configs={"v": v_cfg},
            probe_set=_probes(2),
            max_new_tokens=4,
        )
        # Every prefix_text passed must be "" — the calibration default.
        non_empty = [p for p in prefix_calls if p != ""]
        assert non_empty == [], (
            f"engine received non-empty prefix_text for a config without "
            f"system_prompt/context: {non_empty!r}"
        )

    def test_pipeline_falls_back_when_engine_lacks_prefix_text(self):
        """Engines that don't accept ``prefix_text`` (legacy MockEngine
        in test_geometry.py, custom user backends) get the
        concatenated-prefix call as fallback. Pipeline doesn't crash."""
        from lmdiff._engine import ScoreResult

        class LegacyEngine(MockEngine):
            def score(
                self,
                prompt: str,
                continuation=None,
                *,
                continuation_ids=None,
            ) -> ScoreResult:
                # No prefix_text kwarg.
                if continuation is None:
                    return super().score(prompt, "stub")
                return super().score(prompt, continuation)

            def generate(self, prompt: str, *, max_new_tokens: int = 16,
                         temperature: float = 1.0, top_p: float = 1.0,
                         top_k: int = 0, seed=None):
                return super().generate(prompt, max_new_tokens=max_new_tokens)

        base_cfg = Config(model="mock_base")
        sp_cfg = Config(model="mock_sp", system_prompt="hi.")
        base_eng = LegacyEngine(config=base_cfg, seed=1)
        sp_eng = LegacyEngine(config=sp_cfg, seed=2)

        # Should not crash even though LegacyEngine has no prefix_text
        # kwarg — the pipeline catches TypeError and falls back.
        result = run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_engines={"sp": sp_eng},
            variant_configs={"sp": sp_cfg},
            probe_set=_probes(2),
            max_new_tokens=4,
        )
        assert result.n_probes == 2


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
