"""Integration test for v0.3.2 engine reuse + look-ahead release.

Counts InferenceEngine instantiations across realistic family() orderings
to verify:
  1. Configs runtime-compatible with an earlier config share an engine
  2. The base engine is loaded once and never released
  3. Variant engines are released when the next variant doesn't reuse
     them, and re-loaded on later need (memory > reload-time tradeoff)

We patch ``InferenceEngine`` at the geometry layer with a counting stub,
so no actual model weights load.
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

from lmdiff._api import _BASE_ANCHOR, _compute_anchor_map
from lmdiff._config import Config, DecodeSpec

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from lmdiff.config import Config as V02Config


# ── _compute_anchor_map alone ────────────────────────────────────────


class TestComputeAnchorMap:
    def test_identical_configs_all_share_anchor(self):
        a = Config(model="gpt2")
        b = Config(model="gpt2")
        c = Config(model="gpt2")
        m = _compute_anchor_map([("base", a), ("v1", b), ("v2", c)])
        assert m == {"base": "base", "v1": "base", "v2": "base"}

    def test_runtime_only_diff_reuses_base(self):
        base = Config(model="gpt2")
        only_sp = Config(model="gpt2", system_prompt="hi")
        only_decode = Config(
            model="gpt2", decode=DecodeSpec(strategy="sample", temperature=0.7),
        )
        m = _compute_anchor_map(
            [("base", base), ("sp", only_sp), ("dec", only_decode)],
        )
        assert m == {"base": "base", "sp": "base", "dec": "base"}

    def test_different_model_starts_new_group(self):
        base = Config(model="gpt2")
        other = Config(model="distilgpt2")
        m = _compute_anchor_map([("base", base), ("other", other)])
        assert m == {"base": "base", "other": "other"}

    def test_users_7_variant_pattern(self):
        """Exact ordering from the user's run.py demo."""
        base = Config(model="meta-llama/Llama-2-7b-hf")
        items = [
            (_BASE_ANCHOR, base),
            ("yarn", Config(model="NousResearch/Yarn-Llama-2-7b-128k")),
            ("long", Config(model="togethercomputer/LLaMA-2-7B-32K")),
            ("code", Config(model="codellama/CodeLlama-7b-hf")),
            ("math", Config(model="EleutherAI/llemma_7b")),
            ("chat", Config(model="meta-llama/Llama-2-7b-chat-hf")),
            ("temp_1.5", Config(
                model="meta-llama/Llama-2-7b-hf",
                decode=DecodeSpec(strategy="sample", temperature=1.5),
            )),
            ("system_prompt", Config(
                model="meta-llama/Llama-2-7b-hf",
                system_prompt="You are concise.",
            )),
        ]
        m = _compute_anchor_map(items)

        # 5 variants get their own anchor (different model from base):
        assert m["yarn"] == "yarn"
        assert m["long"] == "long"
        assert m["code"] == "code"
        assert m["math"] == "math"
        assert m["chat"] == "chat"
        # 2 variants reuse base (runtime-only diffs):
        assert m["temp_1.5"] == _BASE_ANCHOR
        assert m["system_prompt"] == _BASE_ANCHOR

        # Distinct anchors → 6 unique engines for 8 configs.
        unique_anchors = set(m.values())
        assert len(unique_anchors) == 6

    def test_chained_runtime_only_resolves_to_earliest(self):
        # B is runtime-only of A; C is runtime-only of A AND of B.
        # C's anchor should be A (the earliest representative), not B.
        a = Config(model="gpt2")
        b = Config(model="gpt2", system_prompt="hi")
        c = Config(model="gpt2", name="c")
        m = _compute_anchor_map([("a", a), ("b", b), ("c", c)])
        assert m["b"] == "a"
        assert m["c"] == "a"


# ── Cache + look-ahead release in analyze() ──────────────────────────


def _build_cg_with_counting_engine_class(variant_v02_configs, base_v02):
    """Returns (cg, init_counter, release_counter, patcher)."""
    from lmdiff.geometry import ChangeGeometry

    init_log: list[str] = []
    release_log: list[str] = []

    class _CountingEngine:
        """Counting stand-in for InferenceEngine. Records every load
        keyed by config.display_name. ``score`` and ``generate`` return
        constant fakes so the geometry math runs to completion."""

        def __init__(self, config):
            init_log.append(config.display_name)
            self.config = config
            self.device = "cpu"
            self._model = None
            # Tokenizer that returns ids = list of integers based on text.
            self.tokenizer = MagicMock()
            self.tokenizer.encode.side_effect = lambda txt, **_: list(range(8))

        def generate(self, prompts, n_samples=1, max_new_tokens=16, **_):
            n = len(prompts)
            return MagicMock(
                completions=[[f"out{i}"] for i in range(n)],
                token_ids=[[[1, 2, 3]] for _ in range(n)],
            )

        def score(self, prompts, continuations=None, continuation_ids=None, **_):
            n = len(prompts)
            return MagicMock(
                cross_entropies=[1.0 + 0.01 * i for i in range(n)],
                token_ids=[[1, 2, 3] for _ in range(n)],
            )

        def __del__(self):
            # Best-effort log of garbage collection (used as a proxy
            # for "released"; not deterministic across Python versions
            # so the assertion below uses the cache size instead).
            try:
                release_log.append(self.config.display_name)
            except Exception:
                pass

    # Patch BOTH the constructor (geometry's `InferenceEngine(...)`) and
    # the lazy `self.base_engine` property which also constructs one.
    ie_patcher = patch("lmdiff.geometry.InferenceEngine", _CountingEngine)
    ie_patcher.start()

    cg = ChangeGeometry(
        base=base_v02,
        variants=variant_v02_configs,
        prompts=["p0", "p1", "p2"],
    )
    return cg, init_log, release_log, ie_patcher


def _v02_from_v03(v03_cfg):
    """Translate a v0.3 Config to v0.2 the same way ``_to_v02_config`` does."""
    return V02Config(
        model=v03_cfg.model,
        system_prompt=v03_cfg.system_prompt,
        decode={"strategy": v03_cfg.decode.strategy},
        name=v03_cfg.name or v03_cfg.model,
    )


class TestAnalyzeReusesAnchorEngine:
    def test_two_runtime_only_variants_load_one_engine(self):
        # base + 2 variants, both runtime-only mods of base. Total
        # InferenceEngine inits should be 1 (just base).
        base = Config(model="gpt2")
        v_cfgs_v03 = {
            "v_sp": Config(model="gpt2", system_prompt="hi"),
            "v_decode": Config(model="gpt2", decode=DecodeSpec(temperature=0.5)),
        }
        anchor_map = _compute_anchor_map(
            [(_BASE_ANCHOR, base), *v_cfgs_v03.items()],
        )
        v_v02 = {n: _v02_from_v03(c) for n, c in v_cfgs_v03.items()}
        cg, init_log, _, patcher = _build_cg_with_counting_engine_class(
            v_v02, _v02_from_v03(base),
        )
        try:
            cg.analyze(max_new_tokens=4, engine_groups=anchor_map)
        finally:
            patcher.stop()
        # Exactly one engine init: the base.
        assert len(init_log) == 1, init_log

    def test_unique_models_each_load_once(self):
        base = Config(model="gpt2")
        v_cfgs_v03 = {
            "v1": Config(model="distilgpt2"),
            "v2": Config(model="gpt2-medium"),
        }
        anchor_map = _compute_anchor_map(
            [(_BASE_ANCHOR, base), *v_cfgs_v03.items()],
        )
        v_v02 = {n: _v02_from_v03(c) for n, c in v_cfgs_v03.items()}
        cg, init_log, _, patcher = _build_cg_with_counting_engine_class(
            v_v02, _v02_from_v03(base),
        )
        try:
            cg.analyze(max_new_tokens=4, engine_groups=anchor_map)
        finally:
            patcher.stop()
        # 1 base + 2 variants = 3 inits.
        assert len(init_log) == 3, init_log

    def test_user_7_variant_pattern_loads_six_engines(self):
        base = Config(model="meta-llama/Llama-2-7b-hf")
        v_cfgs_v03 = {
            "yarn": Config(model="NousResearch/Yarn-Llama-2-7b-128k"),
            "long": Config(model="togethercomputer/LLaMA-2-7B-32K"),
            "code": Config(model="codellama/CodeLlama-7b-hf"),
            "math": Config(model="EleutherAI/llemma_7b"),
            "chat": Config(model="meta-llama/Llama-2-7b-chat-hf"),
            "temp_1.5": Config(
                model="meta-llama/Llama-2-7b-hf",
                decode=DecodeSpec(strategy="sample", temperature=1.5),
            ),
            "system_prompt": Config(
                model="meta-llama/Llama-2-7b-hf",
                system_prompt="You are concise.",
            ),
        }
        anchor_map = _compute_anchor_map(
            [(_BASE_ANCHOR, base), *v_cfgs_v03.items()],
        )
        v_v02 = {n: _v02_from_v03(c) for n, c in v_cfgs_v03.items()}
        cg, init_log, _, patcher = _build_cg_with_counting_engine_class(
            v_v02, _v02_from_v03(base),
        )
        try:
            cg.analyze(max_new_tokens=4, engine_groups=anchor_map)
        finally:
            patcher.stop()
        # 1 base + 5 distinct-model variants = 6 inits.
        # temp_1.5 and system_prompt reuse base, no extra loads.
        assert len(init_log) == 6, init_log


class TestLookAheadRelease:
    def test_aabbaa_ordering_releases_and_reloads(self):
        """[same_a, different_b, same_c] where a and c share an anchor
        but b breaks the run. After variant a's iteration, the next
        variant (b) doesn't reuse a's engine → release it. When we
        reach c (which has the same anchor as a) we reload."""
        base = Config(model="gpt2")
        v_cfgs_v03 = {
            "a": Config(model="distilgpt2"),
            "b": Config(model="gpt2-medium"),
            "c": Config(model="distilgpt2", name="c"),  # same model as a
        }
        anchor_map = _compute_anchor_map(
            [(_BASE_ANCHOR, base), *v_cfgs_v03.items()],
        )
        # Confirm the test setup: c's anchor is a (same model + only name diff)
        assert anchor_map["c"] == "a"
        v_v02 = {n: _v02_from_v03(c) for n, c in v_cfgs_v03.items()}
        cg, init_log, _, patcher = _build_cg_with_counting_engine_class(
            v_v02, _v02_from_v03(base),
        )
        try:
            cg.analyze(max_new_tokens=4, engine_groups=anchor_map)
        finally:
            patcher.stop()
        # 1 (base) + 1 (a) + 1 (b) + 1 (a, reloaded for c) = 4 inits.
        # If the look-ahead release didn't fire, "a" would stay cached
        # and there'd be only 3 inits. If reuse-after-release didn't
        # work, there'd be a crash trying to use a freed engine.
        assert len(init_log) == 4, init_log
        # The reload should re-create "a" — distilgpt2 appears twice.
        a_loads = sum(1 for n in init_log if n == "distilgpt2")
        assert a_loads == 2, f"expected 2 loads of 'a', got {a_loads}: {init_log}"

    def test_consecutive_same_anchor_keeps_engine(self):
        """[v1, v1_dup] — both reuse the same anchor (v1). v1's engine
        loads once, doesn't get released between v1 and v1_dup."""
        base = Config(model="gpt2")
        v_cfgs_v03 = {
            "v1": Config(model="distilgpt2"),
            "v1_dup": Config(model="distilgpt2", name="v1_dup"),
        }
        anchor_map = _compute_anchor_map(
            [(_BASE_ANCHOR, base), *v_cfgs_v03.items()],
        )
        assert anchor_map["v1_dup"] == "v1"
        v_v02 = {n: _v02_from_v03(c) for n, c in v_cfgs_v03.items()}
        cg, init_log, _, patcher = _build_cg_with_counting_engine_class(
            v_v02, _v02_from_v03(base),
        )
        try:
            cg.analyze(max_new_tokens=4, engine_groups=anchor_map)
        finally:
            patcher.stop()
        # 1 (base) + 1 (v1; reused for v1_dup) = 2 inits.
        assert len(init_log) == 2, init_log

    def test_base_never_released_even_when_no_variant_reuses_it(self):
        """User's 4-non-base-reusing-variants pattern.  Base engine is
        loaded once (lazily on first delta call) and stays. Each variant
        loads + releases its own engine."""
        base = Config(model="gpt2")
        v_cfgs_v03 = {
            "x1": Config(model="distilgpt2"),
            "x2": Config(model="gpt2-medium"),
        }
        anchor_map = _compute_anchor_map(
            [(_BASE_ANCHOR, base), *v_cfgs_v03.items()],
        )
        v_v02 = {n: _v02_from_v03(c) for n, c in v_cfgs_v03.items()}
        cg, init_log, _, patcher = _build_cg_with_counting_engine_class(
            v_v02, _v02_from_v03(base),
        )
        try:
            cg.analyze(max_new_tokens=4, engine_groups=anchor_map)
        finally:
            patcher.stop()
        # Base loads once + each variant loads once. The base init
        # appears at any point (lazy), so just verify total count = 3
        # and the base-config name appears exactly once.
        assert len(init_log) == 3, init_log
        assert init_log.count("gpt2") == 1, init_log


class TestLegacyDefaultBehaviour:
    def test_no_engine_groups_loads_each_variant_fresh(self):
        # When engine_groups is None (e.g. user-built ChangeGeometry
        # called directly without going through _api), every variant
        # loads its own engine. This preserves v0.3.0 / v0.3.1 behavior.
        from lmdiff.geometry import ChangeGeometry

        v_cfgs_v03 = {
            "a": Config(model="gpt2"),  # same model as base
            "b": Config(model="gpt2", system_prompt="hi"),  # runtime-only
        }
        v_v02 = {n: _v02_from_v03(c) for n, c in v_cfgs_v03.items()}
        cg, init_log, _, patcher = _build_cg_with_counting_engine_class(
            v_v02, V02Config(model="gpt2"),
        )
        try:
            cg.analyze(max_new_tokens=4)  # no engine_groups
        finally:
            patcher.stop()
        # Legacy: base + each variant separately = 3 inits, even
        # though a and b would share with base under engine_groups.
        assert len(init_log) == 3, init_log
