"""Unit tests for the v0.3.0 Config + sub-specs.

Targets ``lmdiff._config``. Covers construction validation, immutability,
hashability, ``differs_in``, ``to_dict`` / ``from_dict`` / pickle
round-trips (including numpy arrays), the truncated repr, list→tuple
coercion, and the lazy-import contract (no torch on
``import lmdiff._config``).
"""
from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from lmdiff import (
    AdapterSpec,
    Config,
    DecodeSpec,
    ICLExample,
    KVCacheSpec,
    Message,
    PruneSpec,
    QuantSpec,
    SteeringSpec,
)


# ── Basic ─────────────────────────────────────────────────────────────


class TestConfigBasic:
    """Smoke tests for Config construction and access."""

    def test_minimal_config(self):
        cfg = Config(model="gpt2")
        assert cfg.model == "gpt2"
        assert cfg.name is None
        assert cfg.adapter is None

    def test_display_name_defaults_to_model(self):
        cfg = Config(model="meta-llama/Llama-2-7b-hf")
        assert cfg.display_name == "meta-llama/Llama-2-7b-hf"

    def test_display_name_explicit_overrides(self):
        cfg = Config(model="meta-llama/Llama-2-7b-hf", name="llama-base")
        assert cfg.display_name == "llama-base"

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="model is required"):
            Config(model="")

    def test_immutable(self):
        cfg = Config(model="gpt2")
        with pytest.raises(FrozenInstanceError):
            cfg.model = "other"  # type: ignore[misc]

    def test_hashable_in_set(self):
        cfg = Config(model="gpt2")
        s = {cfg}
        assert cfg in s

    def test_hashable_as_dict_key(self):
        cfg = Config(model="gpt2")
        d = {cfg: "value"}
        assert d[cfg] == "value"

    def test_default_decode_is_decodespec(self):
        cfg = Config(model="gpt2")
        assert isinstance(cfg.decode, DecodeSpec)
        assert cfg.decode.strategy == "greedy"


# ── Equality / differs_in ─────────────────────────────────────────────


class TestConfigEquality:
    def test_equal_configs(self):
        c1 = Config(model="gpt2")
        c2 = Config(model="gpt2")
        assert c1 == c2
        assert hash(c1) == hash(c2)

    def test_unequal_models(self):
        assert Config(model="gpt2") != Config(model="distilgpt2")

    def test_differs_in_single_field(self):
        c1 = Config(model="gpt2", system_prompt="a")
        c2 = Config(model="gpt2", system_prompt="b")
        assert c1.differs_in(c2) == ("system_prompt",)

    def test_differs_in_multiple_fields(self):
        c1 = Config(model="gpt2")
        c2 = Config(
            model="gpt2",
            system_prompt="hi",
            decode=DecodeSpec(strategy="sample", temperature=0.7),
        )
        diffs = c2.differs_in(c1)
        assert "system_prompt" in diffs
        assert "decode" in diffs

    def test_differs_in_returns_empty_for_equal(self):
        c1 = Config(model="gpt2")
        c2 = Config(model="gpt2")
        assert c1.differs_in(c2) == ()


# ── AdapterSpec ────────────────────────────────────────────────────────


class TestAdapterSpec:
    def test_lora_basic(self):
        spec = AdapterSpec(type="lora", path="path/to/lora", rank=16)
        assert spec.rank == 16
        assert spec.path == "path/to/lora"

    def test_lora_requires_rank(self):
        with pytest.raises(ValueError, match="requires `rank`"):
            AdapterSpec(type="lora", path="path/to/lora")

    def test_qlora_requires_rank(self):
        with pytest.raises(ValueError, match="requires `rank`"):
            AdapterSpec(type="qlora", path="path/to/qlora")

    def test_ia3_no_rank_required(self):
        spec = AdapterSpec(type="ia3", path="path/to/ia3")
        assert spec.rank is None

    def test_prefix_no_rank_required(self):
        spec = AdapterSpec(type="prefix", path="path/to/prefix")
        assert spec.rank is None

    def test_requires_path(self):
        with pytest.raises(ValueError, match="requires `path`"):
            AdapterSpec(type="lora", rank=16)

    def test_target_modules_list_coerced_to_tuple(self):
        spec = AdapterSpec(
            type="lora", path="p", rank=8,
            target_modules=["q_proj", "v_proj"],
        )
        assert isinstance(spec.target_modules, tuple)

    def test_immutable(self):
        spec = AdapterSpec(type="lora", path="p", rank=8)
        with pytest.raises(FrozenInstanceError):
            spec.rank = 16  # type: ignore[misc]


# ── QuantSpec ──────────────────────────────────────────────────────────


class TestQuantSpec:
    def test_int4_default(self):
        spec = QuantSpec()
        assert spec.method == "int4"
        assert spec.compute_dtype == "bf16"

    def test_gptq_requires_path(self):
        with pytest.raises(ValueError, match="requires `config_path`"):
            QuantSpec(method="gptq")

    def test_awq_requires_path(self):
        with pytest.raises(ValueError, match="requires `config_path`"):
            QuantSpec(method="awq")

    def test_int8_no_path_needed(self):
        spec = QuantSpec(method="int8")
        assert spec.config_path is None

    def test_fp8_no_path_needed(self):
        spec = QuantSpec(method="fp8")
        assert spec.config_path is None


# ── PruneSpec ──────────────────────────────────────────────────────────


class TestPruneSpec:
    def test_unstructured_requires_sparsity(self):
        with pytest.raises(ValueError, match="requires `sparsity`"):
            PruneSpec(type="unstructured")

    def test_unstructured_sparsity_above_one(self):
        with pytest.raises(ValueError, match="must be in"):
            PruneSpec(type="unstructured", sparsity=1.5)

    def test_unstructured_sparsity_negative(self):
        with pytest.raises(ValueError, match="must be in"):
            PruneSpec(type="unstructured", sparsity=-0.1)

    def test_unstructured_sparsity_at_bounds(self):
        # 0.0 and 1.0 are inclusive.
        PruneSpec(type="unstructured", sparsity=0.0)
        PruneSpec(type="unstructured", sparsity=1.0)

    def test_structured_requires_pattern(self):
        with pytest.raises(ValueError, match="requires `pattern`"):
            PruneSpec(type="structured")

    def test_preloaded_requires_path(self):
        with pytest.raises(ValueError, match="requires `config_path`"):
            PruneSpec(type="preloaded")


# ── DecodeSpec ─────────────────────────────────────────────────────────


class TestDecodeSpec:
    def test_default_greedy(self):
        spec = DecodeSpec()
        assert spec.strategy == "greedy"
        assert spec.max_new_tokens == 16

    def test_best_of_n_requires_samples_2(self):
        with pytest.raises(ValueError, match="requires `num_samples`"):
            DecodeSpec(strategy="best_of_n", num_samples=1)

    def test_self_consistency_requires_samples_2(self):
        with pytest.raises(ValueError, match="requires `num_samples`"):
            DecodeSpec(strategy="self_consistency", num_samples=1)

    def test_best_of_n_succeeds_with_samples(self):
        spec = DecodeSpec(strategy="best_of_n", num_samples=8)
        assert spec.num_samples == 8

    def test_greedy_temperature_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DecodeSpec(strategy="greedy", temperature=0.7)
            assert any("ignores temperature" in str(warning.message) for warning in w)

    def test_invalid_top_p_above_one(self):
        with pytest.raises(ValueError, match="top_p"):
            DecodeSpec(top_p=1.5)

    def test_invalid_top_p_negative(self):
        with pytest.raises(ValueError, match="top_p"):
            DecodeSpec(top_p=-0.1)

    def test_invalid_top_k_negative(self):
        with pytest.raises(ValueError, match="top_k"):
            DecodeSpec(top_k=-1)

    def test_top_k_zero_allowed(self):
        DecodeSpec(top_k=0)


# ── ICLExample / Message ──────────────────────────────────────────────


class TestICLExample:
    def test_basic(self):
        ex = ICLExample(user="What is 2+2?", assistant="4")
        assert ex.user == "What is 2+2?"

    def test_immutable(self):
        ex = ICLExample(user="a", assistant="b")
        with pytest.raises(FrozenInstanceError):
            ex.user = "c"  # type: ignore[misc]

    def test_metadata_dict_coerced_to_tuple_of_pairs(self):
        ex = ICLExample(user="a", assistant="b", metadata={"src": "synthetic", "id": 7})
        # Dict was coerced to a hashable, sorted tuple of pairs.
        assert isinstance(ex.metadata, tuple)
        # Hashable now:
        hash(ex)
        assert dict(ex.metadata) == {"src": "synthetic", "id": 7}

    def test_metadata_order_invariant(self):
        e1 = ICLExample(user="a", assistant="b", metadata={"x": 1, "y": 2})
        e2 = ICLExample(user="a", assistant="b", metadata={"y": 2, "x": 1})
        assert e1 == e2
        assert hash(e1) == hash(e2)


class TestMessage:
    def test_basic(self):
        m = Message(role="user", content="hi")
        assert m.role == "user"

    def test_metadata_dict_coerced(self):
        m = Message(role="user", content="hi", metadata={"k": "v"})
        assert isinstance(m.metadata, tuple)


# ── KVCacheSpec ────────────────────────────────────────────────────────


class TestKVCacheSpec:
    def test_default_none(self):
        spec = KVCacheSpec()
        assert spec.method == "none"

    def test_h2o_requires_keep_ratio(self):
        with pytest.raises(ValueError, match="requires `keep_ratio`"):
            KVCacheSpec(method="h2o")

    def test_keep_ratio_above_one(self):
        with pytest.raises(ValueError, match="keep_ratio"):
            KVCacheSpec(method="h2o", keep_ratio=1.5)

    def test_keep_ratio_zero_excluded(self):
        with pytest.raises(ValueError, match="keep_ratio"):
            KVCacheSpec(method="h2o", keep_ratio=0.0)

    def test_keep_ratio_one_inclusive(self):
        KVCacheSpec(method="h2o", keep_ratio=1.0)


# ── SteeringSpec ───────────────────────────────────────────────────────


class TestSteeringSpec:
    def test_requires_vectors_empty_dict(self):
        with pytest.raises(ValueError, match="requires at least one vector"):
            SteeringSpec(vectors={})

    def test_requires_vectors_none(self):
        with pytest.raises(ValueError, match="requires at least one vector"):
            SteeringSpec(vectors=None)

    def test_basic(self):
        spec = SteeringSpec(vectors={5: np.array([1.0, 2.0, 3.0])})
        assert 5 in spec.vectors

    def test_default_scale_one(self):
        spec = SteeringSpec(vectors={0: np.array([1.0])})
        assert spec.scale == 1.0
        assert spec.application == "add"
        assert spec.positions == "all"


# ── Serialization ─────────────────────────────────────────────────────


class TestSerialization:
    def test_to_dict_minimal_round_trip(self):
        cfg = Config(model="gpt2")
        d = cfg.to_dict()
        restored = Config.from_dict(d)
        assert restored == cfg

    def test_to_dict_with_subspec_round_trip(self):
        cfg = Config(
            model="gpt2",
            adapter=AdapterSpec(type="lora", path="p", rank=8),
            decode=DecodeSpec(strategy="sample", temperature=0.7),
        )
        d = cfg.to_dict()
        restored = Config.from_dict(d)
        assert restored == cfg

    def test_json_round_trip(self):
        cfg = Config(model="gpt2", system_prompt="hi")
        s = json.dumps(cfg.to_dict())
        restored = Config.from_dict(json.loads(s))
        assert restored == cfg

    def test_json_with_numpy_round_trip(self):
        cfg = Config(
            model="gpt2",
            soft_prompts=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        )
        s = json.dumps(cfg.to_dict())
        restored = Config.from_dict(json.loads(s))
        np.testing.assert_array_equal(restored.soft_prompts, cfg.soft_prompts)
        assert restored.soft_prompts.dtype == cfg.soft_prompts.dtype

    def test_json_with_steering_round_trip(self):
        cfg = Config(
            model="gpt2",
            steering=SteeringSpec(
                vectors={10: np.array([1.0, 2.0]), 15: np.array([3.0, 4.0])},
                scale=2.0,
            ),
        )
        s = json.dumps(cfg.to_dict())
        restored = Config.from_dict(json.loads(s))
        assert restored.steering.scale == 2.0
        assert set(restored.steering.vectors.keys()) == {10, 15}
        np.testing.assert_array_equal(restored.steering.vectors[10], np.array([1.0, 2.0]))

    def test_json_with_icl_examples_round_trip(self):
        cfg = Config(
            model="gpt2",
            icl_examples=(
                ICLExample(user="Q1", assistant="A1"),
                ICLExample(user="Q2", assistant="A2"),
            ),
        )
        s = json.dumps(cfg.to_dict())
        restored = Config.from_dict(json.loads(s))
        assert restored == cfg
        assert isinstance(restored.icl_examples, tuple)

    def test_json_with_context_round_trip(self):
        cfg = Config(
            model="gpt2",
            context=(
                Message(role="user", content="hi"),
                Message(role="assistant", content="hello"),
            ),
        )
        s = json.dumps(cfg.to_dict())
        restored = Config.from_dict(json.loads(s))
        assert restored == cfg

    def test_json_deterministic_for_frozenset(self):
        """``frozenset`` must serialize as sorted list for stable JSON."""
        cfg1 = Config(model="m", capabilities_required=frozenset({"a", "b", "c"}))
        cfg2 = Config(model="m", capabilities_required=frozenset({"c", "b", "a"}))
        assert cfg1.to_dict() == cfg2.to_dict()

    def test_pickle_round_trip(self):
        cfg = Config(
            model="gpt2",
            adapter=AdapterSpec(type="lora", path="p", rank=8),
        )
        restored = pickle.loads(pickle.dumps(cfg))
        assert restored == cfg

    def test_pickle_round_trip_with_numpy(self):
        cfg = Config(model="gpt2", soft_prompts=np.array([1.0, 2.0]))
        restored = pickle.loads(pickle.dumps(cfg))
        np.testing.assert_array_equal(restored.soft_prompts, cfg.soft_prompts)

    def test_from_dict_rejects_unknown_field(self):
        d = Config(model="gpt2").to_dict()
        d["nonexistent"] = "boom"
        with pytest.raises(ValueError, match="Unknown field"):
            Config.from_dict(d)


# ── Repr ──────────────────────────────────────────────────────────────


class TestRepr:
    def test_minimal(self):
        cfg = Config(model="gpt2")
        r = repr(cfg)
        assert r == "Config(model='gpt2')"

    def test_truncates_long_system_prompt(self):
        long_prompt = "a" * 100
        cfg = Config(model="gpt2", system_prompt=long_prompt)
        r = repr(cfg)
        assert "..." in r
        assert len(r) < 200

    def test_shows_adapter(self):
        cfg = Config(
            model="gpt2",
            adapter=AdapterSpec(type="lora", path="p", rank=16),
        )
        r = repr(cfg)
        assert "adapter" in r

    def test_default_decode_hidden(self):
        cfg = Config(model="gpt2")
        r = repr(cfg)
        assert "decode=" not in r

    def test_non_default_decode_shown(self):
        cfg = Config(model="gpt2", decode=DecodeSpec(strategy="sample", temperature=0.7))
        r = repr(cfg)
        assert "decode=" in r

    def test_truncates_long_recipe_summary(self):
        recipe = "x" * 100
        cfg = Config(model="gpt2", training_recipe_summary=recipe)
        r = repr(cfg)
        assert "training_recipe_summary=" in r
        assert "..." in r


# ── Coercion ──────────────────────────────────────────────────────────


class TestCoercion:
    def test_list_icl_coerced_to_tuple(self):
        cfg = Config(
            model="gpt2",
            icl_examples=[ICLExample(user="a", assistant="b")],
        )
        assert isinstance(cfg.icl_examples, tuple)

    def test_list_context_coerced_to_tuple(self):
        cfg = Config(
            model="gpt2",
            context=[Message(role="user", content="hi")],
        )
        assert isinstance(cfg.context, tuple)

    def test_set_capabilities_coerced_to_frozenset(self):
        cfg = Config(model="gpt2", capabilities_required={"score", "generate"})
        assert isinstance(cfg.capabilities_required, frozenset)

    def test_list_capabilities_coerced_to_frozenset(self):
        cfg = Config(model="gpt2", capabilities_required=["score"])  # type: ignore[arg-type]
        assert isinstance(cfg.capabilities_required, frozenset)


# ── Fully-loaded smoke ────────────────────────────────────────────────


class TestFullyLoadedConfig:
    """Smoke test: construct a Config with every field populated."""

    def test_construction_succeeds(self):
        cfg = Config(
            model="meta-llama/Llama-2-7b-hf",
            name="llama2-fully-loaded",
            adapter=AdapterSpec(type="lora", path="adapter/", rank=16),
            quantization=QuantSpec(method="int4"),
            pruning=PruneSpec(type="preloaded", config_path="pruned/"),
            system_prompt="You are a helpful assistant.",
            icl_examples=(ICLExample(user="Q: 2+2?", assistant="4"),),
            context=(Message(role="user", content="prior turn"),),
            soft_prompts=np.array([[0.1, 0.2]]),
            kv_cache_compression=KVCacheSpec(method="h2o", keep_ratio=0.5),
            decode=DecodeSpec(strategy="sample", temperature=0.7),
            steering=SteeringSpec(vectors={10: np.array([1.0])}),
            tokenizer_id_override="custom_tokenizer_hash",
            capabilities_required=frozenset({"score", "generate"}),
            training_recipe_summary="LoRA r=16 on 100k tokens of math, lr=2e-5",
        )
        assert cfg.model == "meta-llama/Llama-2-7b-hf"

    def test_fully_loaded_round_trip_no_numpy(self):
        # Numpy round-trips exercised in TestSerialization separately.
        cfg = Config(
            model="meta-llama/Llama-2-7b-hf",
            name="x",
            adapter=AdapterSpec(type="lora", path="p", rank=16),
            quantization=QuantSpec(method="int4"),
            pruning=PruneSpec(type="preloaded", config_path="pruned/"),
            system_prompt="hi",
            kv_cache_compression=KVCacheSpec(method="h2o", keep_ratio=0.5),
            decode=DecodeSpec(strategy="sample", temperature=0.7),
            tokenizer_id_override="h",
            capabilities_required=frozenset({"score"}),
            training_recipe_summary="r",
        )
        d = cfg.to_dict()
        s = json.dumps(d)
        restored = Config.from_dict(json.loads(s))
        assert restored == cfg


# ── Lazy import contract ───────────────────────────────────────────────


class TestLazyImport:
    """Critical: importing _config must not load torch."""

    def test_no_torch_at_import(self):
        """Vacuously passes if torch is already loaded by an earlier test
        in the same process; the dedicated CI invocation runs it isolated."""
        import sys
        if "torch" in sys.modules:
            pytest.skip("torch already loaded (run-in-isolation guard)")
        import importlib
        import lmdiff._config as cfg_mod
        importlib.reload(cfg_mod)
        assert "torch" not in sys.modules
