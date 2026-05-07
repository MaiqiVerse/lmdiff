"""Plumbing regression — Fix 3 (v0.4.0 PR #15).

Asserts that ``family(seed=…)`` actually reaches ``engine.generate``,
which it didn't before Fix 3 (``_api.compare/family`` accepted ``seed``
but its docstring said *"Reserved for future randomized metrics; v0.3.0
ignores it."* and the kwarg was never plumbed). The 7-variant
calibration regression surfaced the consequence: ``temp_1.5`` sampling
produced different probe sets across runs (497 vs 500), because RNG
state at temp_1.5's generate step depended on cumulative prior work.

This is **plumbing only**: validates seed propagation through the API
boundary. It does NOT validate that sampling under a pinned seed is
byte-reproducible — that lives in
``tests/integration/test_calibration_regression_7variant.py`` and
requires a real model on GPU.

Precedence contract (Fix 3, v0.4.0):
    DecodeSpec.seed (per variant) > family seed (top-level) > unpinned (None)

Granularity contract: once-per-variant (seed pinned at probe 0 of each
variant; subsequent probes pass seed=None so RNG advances naturally).
"""
from __future__ import annotations

from typing import Any, Optional

from lmdiff._api import _BASE_ANCHOR, _compute_anchor_map
from lmdiff._config import Config, DecodeSpec
from lmdiff._engine import GenerateResult, ScoreResult
from lmdiff._pipeline import run_family_pipeline
from lmdiff.probes.loader import Probe, ProbeSet

from tests.fixtures.mock_engine import MockEngine


# ── Helpers ──────────────────────────────────────────────────────────


class SeedTrackingMockEngine(MockEngine):
    """MockEngine subclass that records every ``seed`` value passed to
    ``generate``. Each call appends ``(prompt, seed)`` to
    ``seed_log``. Used by the plumbing tests to verify the family-seed
    → engine-seed wire is connected.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.seed_log: list[tuple[str, Optional[int]]] = []

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: Optional[int] = None,
    ) -> GenerateResult:
        self.seed_log.append((prompt, seed))
        return super().generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )


def _probes(n: int = 4) -> ProbeSet:
    return ProbeSet(
        [Probe(id=f"p{i}", text=f"prompt_{i}", domain="x") for i in range(n)],
        name="seed_test",
        version="v1",
    )


# ── 1. Family seed reaches engine.generate ───────────────────────────


class TestFamilySeedReachesEngine:
    """``family(seed=42)`` → ``engine.generate(seed=42)`` on probe 0
    of each variant."""

    def test_family_seed_lands_on_probe_zero(self):
        base_cfg = Config(model="m_base")
        v_cfg = Config(model="m_v", decode=DecodeSpec(strategy="sample"))
        base_eng = SeedTrackingMockEngine(config=base_cfg, seed=1)
        v_eng = SeedTrackingMockEngine(config=v_cfg, seed=2)

        run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_engines={"v": v_eng},
            variant_configs={"v": v_cfg},
            probe_set=_probes(4),
            max_new_tokens=4,
            seed=42,
        )

        # v_eng.generate is called exactly once per probe (4 times).
        # Probe 0 must see seed=42; probes 1..3 must see seed=None.
        assert len(v_eng.seed_log) == 4
        assert v_eng.seed_log[0][1] == 42, (
            "family(seed=42) must reach engine.generate on probe 0; "
            f"got seed={v_eng.seed_log[0][1]}"
        )
        for i in range(1, 4):
            assert v_eng.seed_log[i][1] is None, (
                f"probe {i} must receive seed=None (RNG advances "
                f"naturally after probe 0); got seed={v_eng.seed_log[i][1]}"
            )

    def test_family_seed_none_means_no_seed_anywhere(self):
        """No top-level seed and no DecodeSpec.seed → engine.generate
        receives ``seed=None`` for every probe (PyTorch convention:
        no seeding by default)."""
        base_cfg = Config(model="m_base")
        v_cfg = Config(model="m_v", decode=DecodeSpec(strategy="sample"))
        base_eng = SeedTrackingMockEngine(config=base_cfg, seed=1)
        v_eng = SeedTrackingMockEngine(config=v_cfg, seed=2)

        run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_engines={"v": v_eng},
            variant_configs={"v": v_cfg},
            probe_set=_probes(3),
            max_new_tokens=4,
            seed=None,
        )

        assert all(s is None for _, s in v_eng.seed_log), (
            f"seed=None should produce all-None seed_log; "
            f"got {[s for _, s in v_eng.seed_log]}"
        )


# ── 2. Precedence: DecodeSpec.seed overrides family seed ─────────────


class TestSeedPrecedence:
    def test_decode_seed_overrides_family_seed_per_variant(self):
        """When variant_a has DecodeSpec(seed=99) and variant_b has
        DecodeSpec(seed=None), family(seed=42) → a sees 99, b sees 42."""
        base_cfg = Config(model="m_base")
        a_cfg = Config(
            model="m_a",
            decode=DecodeSpec(strategy="sample", seed=99),
        )
        b_cfg = Config(
            model="m_b",
            decode=DecodeSpec(strategy="sample"),  # seed=None
        )
        base_eng = SeedTrackingMockEngine(config=base_cfg, seed=1)
        a_eng = SeedTrackingMockEngine(config=a_cfg, seed=2)
        b_eng = SeedTrackingMockEngine(config=b_cfg, seed=3)

        run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_engines={"a": a_eng, "b": b_eng},
            variant_configs={"a": a_cfg, "b": b_cfg},
            probe_set=_probes(2),
            max_new_tokens=4,
            seed=42,
        )

        assert a_eng.seed_log[0][1] == 99, (
            "DecodeSpec.seed=99 must win over family seed=42; "
            f"got {a_eng.seed_log[0][1]}"
        )
        assert b_eng.seed_log[0][1] == 42, (
            "variant b has DecodeSpec.seed=None; family seed=42 must "
            f"be used; got {b_eng.seed_log[0][1]}"
        )

    def test_decode_seed_used_when_family_seed_is_none(self):
        """DecodeSpec.seed=99, family(seed=None) → engine sees 99."""
        base_cfg = Config(model="m_base")
        v_cfg = Config(
            model="m_v",
            decode=DecodeSpec(strategy="sample", seed=99),
        )
        base_eng = SeedTrackingMockEngine(config=base_cfg, seed=1)
        v_eng = SeedTrackingMockEngine(config=v_cfg, seed=2)

        run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_engines={"v": v_eng},
            variant_configs={"v": v_cfg},
            probe_set=_probes(2),
            max_new_tokens=4,
            seed=None,
        )

        assert v_eng.seed_log[0][1] == 99


# ── 3. Reproducibility — same seed twice produces same seed sequence ──


class TestSeedSequenceReproducibility:
    def test_back_to_back_same_seed_same_sequence(self):
        """Running the pipeline twice with the same family seed must
        produce identical seed sequences at the engine boundary. This
        is the plumbing-level reproducibility guarantee — the
        downstream "same seed → same outputs" guarantee depends on the
        engine's own determinism (HFEngine on a real model under
        BF16 attention may still have hardware-level non-determinism
        — that's not what this test catches)."""
        base_cfg = Config(model="m_base")
        v_cfg = Config(model="m_v", decode=DecodeSpec(strategy="sample"))

        def _run() -> list[Optional[int]]:
            base_eng = SeedTrackingMockEngine(config=base_cfg, seed=1)
            v_eng = SeedTrackingMockEngine(config=v_cfg, seed=2)
            run_family_pipeline(
                base_engine=base_eng,
                base_config=base_cfg,
                variant_engines={"v": v_eng},
                variant_configs={"v": v_cfg},
                probe_set=_probes(3),
                max_new_tokens=4,
                seed=42,
            )
            return [s for _, s in v_eng.seed_log]

        seq1 = _run()
        seq2 = _run()
        assert seq1 == seq2, (
            f"identical inputs + seed=42 must produce identical seed "
            f"sequences at engine boundary; got {seq1} vs {seq2}"
        )
        assert seq1 == [42, None, None]


# ── 4. Public API plumbing — _api.family/compare reach run_pipeline ──


class TestApiSeedPlumbing:
    """Validates the outer-most boundary: ``lmdiff.family(seed=42)``
    eventually reaches ``run_family_pipeline(seed=42)``."""

    def test_api_family_passes_seed_to_pipeline(self, monkeypatch):
        from lmdiff import _api as api_mod
        from lmdiff import _pipeline as pipeline_mod

        captured: dict[str, Any] = {}

        def _fake_run_family_pipeline(*args: Any, **kwargs: Any):
            captured.update(kwargs)
            # Return a minimally-valid GeoResult-shaped object so the
            # caller's `result.metadata.update(probe_info)` doesn't blow
            # up. We only care about the seed kwarg — short-circuit.
            from types import SimpleNamespace
            return SimpleNamespace(metadata={})

        # Patch where _api looks it up — _api does a local import inside
        # family(), so we patch on the pipeline module.
        monkeypatch.setattr(
            pipeline_mod, "run_family_pipeline", _fake_run_family_pipeline,
        )
        # _api.family imports run_family_pipeline lazily; ensure the
        # late-bound lookup sees the patched version.
        monkeypatch.setattr(
            "lmdiff._pipeline.run_family_pipeline",
            _fake_run_family_pipeline,
            raising=True,
        )

        # Build a no-op engine factory so _api doesn't try to load HF.
        class _NopEng(MockEngine):
            pass

        base_eng = _NopEng(config=Config(model="m_base"), seed=1)

        # _build_engine_for_config takes an engine template. With a
        # template, _api won't try to construct HFEngine.
        api_mod.family(
            base="m_base",
            variants={"v": "m_v"},
            probes=_probes(2),
            n_probes=2,
            metrics=[],
            engine=base_eng,
            seed=42,
        )

        assert captured.get("seed") == 42, (
            f"_api.family(seed=42) must plumb seed=42 to "
            f"run_family_pipeline; got {captured.get('seed')!r}"
        )
