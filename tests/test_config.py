import pytest

from modeldiff.config import Config


class TestConfigCreation:
    def test_minimal(self):
        c = Config(model="gpt2")
        assert c.model == "gpt2"
        assert c.context is None
        assert c.system_prompt is None
        assert c.decode == {"strategy": "greedy"}
        assert c.adapter is None
        assert c.ttt is None
        assert c.agent is None
        assert c.name is None

    def test_full(self):
        c = Config(
            model="gpt2",
            context=[{"role": "user", "content": "hi"}],
            system_prompt="Be concise.",
            decode={"strategy": "sample", "temperature": 0.7},
            adapter="lora-v1",
            ttt={"lr": 1e-4},
            agent="react",
            name="my-config",
        )
        assert c.context == [{"role": "user", "content": "hi"}]
        assert c.system_prompt == "Be concise."
        assert c.decode["temperature"] == 0.7
        assert c.adapter == "lora-v1"
        assert c.ttt == {"lr": 1e-4}
        assert c.agent == "react"
        assert c.name == "my-config"

    def test_model_none_raises(self):
        with pytest.raises(ValueError, match="model must not be None"):
            Config(model=None)

    def test_decode_non_dict_raises(self):
        with pytest.raises(TypeError, match="decode must be a dict"):
            Config(model="gpt2", decode="greedy")

    def test_model_can_be_object(self):
        sentinel = object()
        c = Config(model=sentinel)
        assert c.model is sentinel


class TestDisplayName:
    def test_uses_name_if_set(self):
        c = Config(model="gpt2", name="my-name")
        assert c.display_name == "my-name"

    def test_uses_model_string(self):
        c = Config(model="gpt2")
        assert c.display_name == "gpt2"

    def test_uses_class_name_for_object(self):
        c = Config(model=42)
        assert c.display_name == "int"


class TestSharesToknizerWith:
    def test_same_model(self):
        a = Config(model="gpt2")
        b = Config(model="gpt2")
        assert a.shares_tokenizer_with(b)

    def test_different_model_returns_none(self):
        a = Config(model="gpt2")
        b = Config(model="llama-7b")
        assert a.shares_tokenizer_with(b) is None

    def test_non_string_model_returns_none(self):
        a = Config(model="gpt2")
        b = Config(model=object())
        assert a.shares_tokenizer_with(b) is None


class TestWithOverride:
    def test_override_name(self):
        original = Config(model="gpt2", name="orig")
        new = original.with_override(name="new")
        assert new.name == "new"
        assert original.name == "orig"

    def test_override_decode(self):
        original = Config(model="gpt2")
        new = original.with_override(decode={"strategy": "sample", "temperature": 0.9})
        assert new.decode["temperature"] == 0.9
        assert original.decode == {"strategy": "greedy"}

    def test_override_model(self):
        new = Config(model="gpt2").with_override(model="llama-7b")
        assert new.model == "llama-7b"


class TestUseChatTemplate:
    def test_default_false(self):
        c = Config(model="gpt2")
        assert c.use_chat_template is False

    def test_override(self):
        c = Config(model="gpt2").with_override(use_chat_template=True)
        assert c.use_chat_template is True


class TestFixtures:
    def test_gpt2_config(self, gpt2_config):
        assert gpt2_config.model == "gpt2"

    def test_gpt2_config_with_context(self, gpt2_config_with_context):
        assert gpt2_config_with_context.context is not None
        assert gpt2_config_with_context.display_name == "gpt2-ctx"

    def test_sample_probes(self, sample_probes):
        assert len(sample_probes) == 3
