"""Tests for ``Config.is_runtime_only_modification_of`` (v0.3.2 OOM fix).

The predicate decides whether one loaded ``InferenceEngine`` can serve
two Configs. Wrong-positive (saying yes when weights actually differ)
silently corrupts results; wrong-negative just costs reload time. So
the test scope is paranoid — every field flipped one at a time, plus
the explicit allow/deny list.
"""
from __future__ import annotations

import numpy as np
import pytest

from lmdiff._config import (
    AdapterSpec,
    Config,
    DecodeSpec,
    ICLExample,
    KVCacheSpec,
    Message,
    MODEL_SPECIFIC_COMPARATORS,
    PruneSpec,
    QuantSpec,
    RUNTIME_ONLY_FIELDS,
    SteeringSpec,
)


# ── Field classification audit ───────────────────────────────────────


def test_runtime_only_fields_set_is_what_we_documented():
    # Lock in the exact membership so a future PR adding a field
    # accidentally going in/out of the set fails this test loudly.
    assert RUNTIME_ONLY_FIELDS == frozenset({
        "name",
        "system_prompt",
        "icl_examples",
        "context",
        "decode",
        "tokenizer_id_override",
        "capabilities_required",
        "training_recipe_summary",
    })


def test_model_field_is_NOT_runtime_only():
    # Sanity — different model ALWAYS forces a separate engine.
    assert "model" not in RUNTIME_ONLY_FIELDS


def test_weight_modifying_fields_are_NOT_runtime_only():
    # Adapter / quantization / pruning all transform weights at load.
    for field in ("adapter", "quantization", "pruning"):
        assert field not in RUNTIME_ONLY_FIELDS, field


def test_hook_installing_fields_are_NOT_runtime_only():
    # Defensive: kv-cache compression and steering install module
    # forward hooks; not safe to swap per-call.
    for field in ("kv_cache_compression", "steering", "soft_prompts"):
        assert field not in RUNTIME_ONLY_FIELDS, field


# ── Identity + symmetry ──────────────────────────────────────────────


class TestReflexivityAndSymmetry:
    def test_identical_configs_compatible(self):
        a = Config(model="gpt2")
        b = Config(model="gpt2")
        assert a.is_runtime_only_modification_of(b)
        assert b.is_runtime_only_modification_of(a)

    def test_reflexive_with_self(self):
        c = Config(model="gpt2", system_prompt="hi")
        assert c.is_runtime_only_modification_of(c)


# ── Different model ALWAYS rejects ───────────────────────────────────


class TestModelMismatch:
    def test_different_model_string_rejected(self):
        a = Config(model="gpt2")
        b = Config(model="distilgpt2")
        assert not a.is_runtime_only_modification_of(b)

    def test_different_model_even_when_other_fields_match(self):
        # Two configs that are byte-identical except for ``model`` —
        # the model field difference must still cause rejection.
        a = Config(model="gpt2", system_prompt="hi", name="a")
        b = Config(model="distilgpt2", system_prompt="hi", name="a")
        assert not a.is_runtime_only_modification_of(b)


# ── Each runtime-only field, one at a time ───────────────────────────


class TestRuntimeOnlyFieldsAcceptDifferences:
    """Vary ONE runtime-only field; predicate must return True."""

    BASE = Config(model="gpt2")

    def test_name_diff_accepted(self):
        a = Config(model="gpt2", name="alpha")
        b = Config(model="gpt2", name="beta")
        assert a.is_runtime_only_modification_of(b)

    def test_system_prompt_diff_accepted(self):
        a = Config(model="gpt2", system_prompt="be helpful")
        b = Config(model="gpt2", system_prompt="be concise")
        assert a.is_runtime_only_modification_of(b)

    def test_decode_diff_accepted(self):
        a = Config(model="gpt2", decode=DecodeSpec(strategy="greedy"))
        b = Config(model="gpt2", decode=DecodeSpec(strategy="sample", temperature=1.5))
        assert a.is_runtime_only_modification_of(b)

    def test_icl_examples_diff_accepted(self):
        a = Config(model="gpt2", icl_examples=(ICLExample(user="2+2", assistant="4"),))
        b = Config(model="gpt2", icl_examples=(ICLExample(user="3+3", assistant="6"),))
        assert a.is_runtime_only_modification_of(b)

    def test_context_diff_accepted(self):
        a = Config(model="gpt2", context=(Message(role="user", content="hi"),))
        b = Config(model="gpt2", context=(Message(role="user", content="bye"),))
        assert a.is_runtime_only_modification_of(b)

    def test_tokenizer_id_override_diff_accepted(self):
        a = Config(model="gpt2", tokenizer_id_override="custom-id-1")
        b = Config(model="gpt2", tokenizer_id_override="custom-id-2")
        assert a.is_runtime_only_modification_of(b)

    def test_capabilities_required_diff_accepted(self):
        a = Config(model="gpt2", capabilities_required=frozenset({"score"}))
        b = Config(model="gpt2", capabilities_required=frozenset({"score", "generate"}))
        assert a.is_runtime_only_modification_of(b)

    def test_training_recipe_summary_diff_accepted(self):
        a = Config(model="gpt2", training_recipe_summary="continued pretrain on math")
        b = Config(model="gpt2", training_recipe_summary="instruct-tuned on chat")
        assert a.is_runtime_only_modification_of(b)


# ── Each weight-affecting field, one at a time ───────────────────────


class TestWeightAffectingFieldsRejectDifferences:
    """Vary ONE weight-affecting field; predicate must return False."""

    def test_adapter_diff_rejected(self):
        a = Config(model="gpt2")
        b = Config(model="gpt2", adapter=AdapterSpec(type="lora", path="/x", rank=8))
        assert not a.is_runtime_only_modification_of(b)

    def test_quantization_diff_rejected(self):
        a = Config(model="gpt2")
        b = Config(model="gpt2", quantization=QuantSpec(method="int4"))
        assert not a.is_runtime_only_modification_of(b)

    def test_pruning_diff_rejected(self):
        a = Config(model="gpt2")
        b = Config(
            model="gpt2",
            pruning=PruneSpec(type="preloaded", config_path="/p"),
        )
        assert not a.is_runtime_only_modification_of(b)

    def test_soft_prompts_diff_rejected(self):
        a = Config(model="gpt2")
        b = Config(model="gpt2", soft_prompts=np.zeros((4, 8)))
        assert not a.is_runtime_only_modification_of(b)

    def test_kv_cache_compression_diff_rejected(self):
        a = Config(model="gpt2")
        b = Config(
            model="gpt2",
            kv_cache_compression=KVCacheSpec(method="h2o", keep_ratio=0.5),
        )
        assert not a.is_runtime_only_modification_of(b)

    def test_steering_diff_rejected(self):
        # SteeringSpec requires non-empty vectors via __post_init__.
        a = Config(model="gpt2")
        b = Config(model="gpt2", steering=SteeringSpec(vectors={0: np.zeros(8)}))
        assert not a.is_runtime_only_modification_of(b)


# ── Combinations ─────────────────────────────────────────────────────


class TestCombinedDifferences:
    def test_multiple_runtime_only_fields_still_compatible(self):
        a = Config(model="gpt2", name="a", system_prompt="hi")
        b = Config(
            model="gpt2",
            name="b",
            system_prompt="bye",
            decode=DecodeSpec(strategy="sample", temperature=0.7),
        )
        assert a.is_runtime_only_modification_of(b)

    def test_runtime_plus_weight_diff_rejected(self):
        a = Config(model="gpt2", name="a")
        b = Config(model="gpt2", name="b", quantization=QuantSpec(method="int8"))
        assert not a.is_runtime_only_modification_of(b)


# ── Model-specific comparator hook ───────────────────────────────────


class TestModelSpecificComparator:
    def test_registered_comparator_overrides_default(self, monkeypatch):
        monkeypatch.setitem(
            MODEL_SPECIFIC_COMPARATORS,
            "test_model",
            lambda a, b: False,
        )
        a = Config(model="test_model")
        b = Config(model="test_model")
        # Default would say True (identical), but the comparator vetoes.
        assert not a.is_runtime_only_modification_of(b)

    def test_registered_comparator_can_be_more_lax(self, monkeypatch):
        monkeypatch.setitem(
            MODEL_SPECIFIC_COMPARATORS,
            "test_model",
            lambda a, b: True,
        )
        # Default would reject (different adapter); custom comparator
        # OKs reuse anyway.
        a = Config(model="test_model")
        b = Config(
            model="test_model",
            adapter=AdapterSpec(type="lora", path="/x", rank=8),
        )
        assert a.is_runtime_only_modification_of(b)

    def test_comparator_keyed_by_model_string_only(self, monkeypatch):
        # Comparator under "model_a" doesn't affect "model_b".
        monkeypatch.setitem(
            MODEL_SPECIFIC_COMPARATORS,
            "model_a",
            lambda a, b: False,
        )
        a = Config(model="model_b")
        b = Config(model="model_b", name="renamed")
        assert a.is_runtime_only_modification_of(b)  # default behavior

    def test_comparator_default_dict_is_empty(self):
        # The shipped registry has nothing in it — pure default behavior
        # for everyone unless they register something.
        assert MODEL_SPECIFIC_COMPARATORS == {}


# ── Realistic family() pattern ───────────────────────────────────────


class TestUserFamilyPattern:
    """The exact pattern from the user's run.py — base + 7 variants
    where 2 variants share base's model. The predicate should let those
    2 reuse base's engine and force separate engines for the other 5."""

    def test_users_temp_15_variant_reuses_base(self):
        base = Config(model="meta-llama/Llama-2-7b-hf")
        temp_15 = Config(
            model="meta-llama/Llama-2-7b-hf",
            decode=DecodeSpec(strategy="sample", temperature=1.5),
            name="temp_1.5",
        )
        assert temp_15.is_runtime_only_modification_of(base)

    def test_users_system_prompt_variant_reuses_base(self):
        base = Config(model="meta-llama/Llama-2-7b-hf")
        sp = Config(
            model="meta-llama/Llama-2-7b-hf",
            system_prompt="You are concise.",
            name="system_prompt",
        )
        assert sp.is_runtime_only_modification_of(base)

    def test_users_chat_variant_does_NOT_reuse_base(self):
        # ``Llama-2-7b-chat-hf`` is a different model id from the base.
        base = Config(model="meta-llama/Llama-2-7b-hf")
        chat = Config(model="meta-llama/Llama-2-7b-chat-hf", name="chat")
        assert not chat.is_runtime_only_modification_of(base)
