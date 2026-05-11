"""Lazy-load regression — Fix 4 (v0.4.0 PR #15).

Asserts that ``run_family_pipeline(engine_factory=…)`` constructs at
most ONE variant engine at any moment (peak resident = base + 1
variant = 2). Without Fix 4, ``_api.family()`` pre-built every
unique-model variant before the pipeline started — for the 7-variant
Llama family that meant 6 model loads sitting in memory simultaneously,
silently spilling through ``device_map="auto"`` to CPU and tanking
throughput. v0.3.2's ``ChangeGeometry.analyze`` had peak = 2; the
v0.4.0 cutover lost it; this test gates against losing it again.

Companion to L-029 (release aggressively) and L-032 (porting
orchestration logic must preserve resource-lifetime contract, not
just the function signature).
"""
from __future__ import annotations

from typing import Any, Optional

import pytest

from lmdiff._api import _BASE_ANCHOR, _compute_anchor_map
from lmdiff._config import Config
from lmdiff._engine import GenerateResult, ScoreResult
from lmdiff._pipeline import run_family_pipeline
from lmdiff.probes.loader import Probe, ProbeSet

from tests.fixtures.mock_engine import MockEngine


# ── Counting MockEngine — tracks construct/close lifecycle ──────────


class CountingMockEngine(MockEngine):
    """MockEngine subclass that records construct + close as events on
    a shared monotonic timeline. Lets the test compute peak resident
    engines = max of (constructed - closed) at any single moment."""

    # Class-level shared timeline — every event is (tick, delta, model, iid)
    # where delta is +1 for construct, -1 for close. Reset by the fixture
    # below at the start of each test.
    timeline: list[tuple[int, int, str, int]] = []
    _next_id: int = 0
    _next_tick: int = 0

    @classmethod
    def reset(cls) -> None:
        cls.timeline = []
        cls._next_id = 0
        cls._next_tick = 0

    @classmethod
    def _stamp(cls, delta: int, model: str, iid: int) -> None:
        cls.timeline.append((cls._next_tick, delta, model, iid))
        cls._next_tick += 1

    @classmethod
    def constructed_models(cls) -> list[str]:
        return [model for _, delta, model, _ in cls.timeline if delta == +1]

    @classmethod
    def closed_models(cls) -> list[str]:
        return [model for _, delta, model, _ in cls.timeline if delta == -1]

    @classmethod
    def peak_resident(cls) -> int:
        """Walk the timeline; return max (constructed - closed) at any
        single moment. Real wall-time order is preserved by the
        monotonic ``_next_tick`` counter."""
        peak = 0
        cur = 0
        for _, delta, _, _ in cls.timeline:
            cur += delta
            peak = max(peak, cur)
        return peak

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cls = type(self)
        self._iid = cls._next_id
        cls._next_id += 1
        cls._stamp(+1, self._config.model, self._iid)

    def close(self) -> None:
        cls = type(self)
        cls._stamp(-1, self._config.model, self._iid)
        super().close()

    def with_config(self, config: Config) -> "CountingMockEngine":
        return CountingMockEngine(
            config=config,
            capabilities=self._capabilities,
            seed=self._seed,
            n_layers=self._n_layers,
            hidden_dim=self._hidden_dim,
        )


@pytest.fixture(autouse=True)
def _reset_counter():
    CountingMockEngine.reset()
    yield
    CountingMockEngine.reset()


def _probes(n: int = 3) -> ProbeSet:
    return ProbeSet(
        [Probe(id=f"p{i}", text=f"prompt_{i}", domain="x") for i in range(n)],
        name="lazy_test",
        version="v1",
    )


# ── Fix 4: factory-mode peak resident is 2 ──────────────────────────


class TestLazyEngineLoading:
    """The contract: with ``engine_factory=…``, the pipeline holds at
    most 2 variant-or-base engines in the cache simultaneously."""

    def test_peak_resident_is_two_for_three_unique_variants(self):
        """3 unique-model variants + base = 4 distinct engines total.
        Lazy mode must keep peak resident at 2."""
        base_cfg = Config(model="m_base")
        a_cfg = Config(model="m_a")
        b_cfg = Config(model="m_b")
        c_cfg = Config(model="m_c")

        anchor_map = _compute_anchor_map(
            [(_BASE_ANCHOR, base_cfg), ("a", a_cfg), ("b", b_cfg), ("c", c_cfg)],
        )
        # All distinct models → each variant is its own anchor.
        assert anchor_map["a"] == "a"
        assert anchor_map["b"] == "b"
        assert anchor_map["c"] == "c"

        base_eng = CountingMockEngine(config=base_cfg, seed=1)

        def factory(cfg: Config):
            # Pipeline-owned (True) — pipeline must close on release.
            return CountingMockEngine(config=cfg, seed=hash(cfg.model) % 100), True

        run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_configs={"a": a_cfg, "b": b_cfg, "c": c_cfg},
            probe_set=_probes(2),
            engine_factory=factory,
            max_new_tokens=2,
            engine_groups=anchor_map,
        )

        # All 4 engines were constructed (1 base + 3 variants).
        assert len(CountingMockEngine.constructed_models()) == 4
        # All 3 lazy-built variants were closed (base is caller-owned).
        assert len(CountingMockEngine.closed_models()) == 3
        # Peak resident = 2 (base + active variant).
        assert CountingMockEngine.peak_resident() <= 2, (
            f"lazy mode must keep peak resident ≤ 2; got "
            f"{CountingMockEngine.peak_resident()} "
            f"(timeline: {CountingMockEngine.timeline})"
        )

    def test_runtime_only_variants_share_base_no_extra_load(self):
        """Variants whose anchor is base must NOT trigger the factory —
        they reuse base_engine directly (peak stays at 1 base)."""
        base_cfg = Config(model="m_base")
        # Both variants are runtime-only mods of base (different
        # system_prompt) — anchor map collapses them to __base__.
        sp1_cfg = Config(model="m_base", system_prompt="hi.")
        sp2_cfg = Config(model="m_base", system_prompt="bye.")

        anchor_map = _compute_anchor_map(
            [(_BASE_ANCHOR, base_cfg), ("sp1", sp1_cfg), ("sp2", sp2_cfg)],
        )
        assert anchor_map["sp1"] == _BASE_ANCHOR
        assert anchor_map["sp2"] == _BASE_ANCHOR

        base_eng = CountingMockEngine(config=base_cfg, seed=1)
        factory_calls: list[Config] = []

        def factory(cfg: Config):
            factory_calls.append(cfg)
            return CountingMockEngine(config=cfg, seed=2), True

        run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_configs={"sp1": sp1_cfg, "sp2": sp2_cfg},
            probe_set=_probes(2),
            engine_factory=factory,
            max_new_tokens=2,
            engine_groups=anchor_map,
        )

        assert factory_calls == [], (
            f"runtime-only variants must reuse base; factory was "
            f"called for {[c.display_name for c in factory_calls]}"
        )
        assert len(CountingMockEngine.constructed_models()) == 1, (
            "only base should have been constructed"
        )
        assert len(CountingMockEngine.closed_models()) == 0, (
            "base is caller-owned; pipeline must not close it"
        )

    def test_anchor_reuse_avoids_extra_load(self):
        """Two variants sharing an anchor (e.g. same model, different
        runtime params) trigger ONE factory call, not two — and the
        engine survives between them, then releases together."""
        base_cfg = Config(model="m_base")
        a_cfg = Config(model="m_a")
        # ``a2`` is runtime-only mod of ``a`` (different name only).
        a2_cfg = Config(model="m_a", name="a2")

        anchor_map = _compute_anchor_map(
            [(_BASE_ANCHOR, base_cfg), ("a", a_cfg), ("a2", a2_cfg)],
        )
        assert anchor_map["a2"] == "a", anchor_map

        base_eng = CountingMockEngine(config=base_cfg, seed=1)
        factory_calls: list[Config] = []

        def factory(cfg: Config):
            factory_calls.append(cfg)
            return CountingMockEngine(config=cfg, seed=2), True

        run_family_pipeline(
            base_engine=base_eng,
            base_config=base_cfg,
            variant_configs={"a": a_cfg, "a2": a2_cfg},
            probe_set=_probes(2),
            engine_factory=factory,
            max_new_tokens=2,
            engine_groups=anchor_map,
        )

        assert len(factory_calls) == 1, (
            f"anchor reuse must collapse two variants to one factory "
            f"call; got {len(factory_calls)} calls"
        )
        # 1 base + 1 anchor-shared variant = 2 constructed.
        assert len(CountingMockEngine.constructed_models()) == 2
        # Lazy-built engine closed after both variants finished using it.
        assert len(CountingMockEngine.closed_models()) == 1


# ── API requires exactly one of variant_engines / engine_factory ─────


class TestEngineSupplyValidation:
    def test_neither_supplied_raises(self):
        base_cfg = Config(model="m_base")
        v_cfg = Config(model="m_v")
        base_eng = CountingMockEngine(config=base_cfg)

        with pytest.raises(ValueError, match="exactly one"):
            run_family_pipeline(
                base_engine=base_eng,
                base_config=base_cfg,
                variant_configs={"v": v_cfg},
                probe_set=_probes(2),
                # neither variant_engines nor engine_factory
            )

    def test_both_supplied_raises(self):
        base_cfg = Config(model="m_base")
        v_cfg = Config(model="m_v")
        base_eng = CountingMockEngine(config=base_cfg)
        v_eng = CountingMockEngine(config=v_cfg)

        def factory(cfg: Config):
            return CountingMockEngine(config=cfg), True

        with pytest.raises(ValueError, match="exactly one"):
            run_family_pipeline(
                base_engine=base_eng,
                base_config=base_cfg,
                variant_configs={"v": v_cfg},
                probe_set=_probes(2),
                variant_engines={"v": v_eng},
                engine_factory=factory,
            )


# ── Lifecycle: factory exception during build doesn't leak engines ──


class TestFactoryExceptionDoesNotLeak:
    def test_factory_raises_does_not_leak_partial_engines(self):
        """If the factory raises mid-loop, any engines the pipeline
        already built must be closed in the finally block — the
        exception propagates but no weights leak."""
        base_cfg = Config(model="m_base")
        a_cfg = Config(model="m_a")
        b_cfg = Config(model="m_b")

        anchor_map = _compute_anchor_map(
            [(_BASE_ANCHOR, base_cfg), ("a", a_cfg), ("b", b_cfg)],
        )

        base_eng = CountingMockEngine(config=base_cfg, seed=1)
        call_count = [0]

        def factory(cfg: Config):
            call_count[0] += 1
            if call_count[0] == 2:  # second factory call (for "b") fails
                raise RuntimeError("synthetic factory failure")
            return CountingMockEngine(config=cfg, seed=2), True

        with pytest.raises(RuntimeError, match="synthetic factory failure"):
            run_family_pipeline(
                base_engine=base_eng,
                base_config=base_cfg,
                variant_configs={"a": a_cfg, "b": b_cfg},
                probe_set=_probes(2),
                engine_factory=factory,
                max_new_tokens=2,
                engine_groups=anchor_map,
            )

        # ``a`` was built and used; look-ahead-by-one would've closed
        # it before ``b`` started. ``b`` factory raised before producing
        # an engine. Net: 1 base (caller-owned) + 1 lazy (pipeline-owned,
        # closed) = 2 constructed, 1 closed. No leak.
        assert len(CountingMockEngine.constructed_models()) == 2, (
            f"expected 2 constructed (base + a); got "
            f"{len(CountingMockEngine.constructed_models())}: "
            f"{CountingMockEngine.constructed_models()}"
        )
        assert len(CountingMockEngine.closed_models()) == 1, (
            f"expected 1 closed (lazy ``a``); got "
            f"{len(CountingMockEngine.closed_models())}: "
            f"{CountingMockEngine.closed_models()}"
        )
