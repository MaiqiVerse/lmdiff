import numpy as np
import torch
import pytest

from modeldiff.config import Config
from modeldiff.engine import (
    InferenceEngine,
    GenerationResult,
    ForwardResult,
    HiddenStatesResult,
)


class TestEngineInit:
    def test_loads_model(self, tiny_model):
        assert tiny_model._model is not None
        assert tiny_model._tokenizer is not None

    def test_pad_token_set(self, tiny_model):
        assert tiny_model.tokenizer.pad_token is not None

    def test_model_name(self, tiny_model):
        assert tiny_model.model_name == "gpt2"

    def test_eval_mode(self, tiny_model):
        assert not tiny_model._model.training


class TestGenerate:
    def test_single_prompt_single_sample(self, tiny_model):
        result = tiny_model.generate(["Hello world"], n_samples=1, max_new_tokens=10)
        assert isinstance(result, GenerationResult)
        assert len(result.prompts) == 1
        assert len(result.completions) == 1
        assert len(result.completions[0]) == 1
        assert isinstance(result.completions[0][0], str)
        assert len(result.completions[0][0]) > 0

    def test_multiple_prompts(self, tiny_model):
        prompts = ["The sky is", "One plus one equals"]
        result = tiny_model.generate(prompts, n_samples=1, max_new_tokens=10)
        assert len(result.completions) == 2

    def test_multiple_samples(self, tiny_model):
        result = tiny_model.generate(["Hello"], n_samples=3, max_new_tokens=10)
        assert len(result.completions[0]) == 3

    def test_token_ids_populated(self, tiny_model):
        result = tiny_model.generate(["Test"], n_samples=1, max_new_tokens=10)
        assert result.token_ids is not None
        assert len(result.token_ids[0][0]) > 0

    def test_context_prepended(self, gpt2_config_with_context):
        engine = InferenceEngine(gpt2_config_with_context)
        built = engine._build_prompt("Tell me more")
        assert "You are helpful." in built
        assert "Hello" in built
        assert "Tell me more" in built


class TestScore:
    def test_basic_score(self, tiny_model):
        result = tiny_model.score(
            prompts=["The capital of France is"],
            continuations=[" Paris"],
        )
        assert isinstance(result, ForwardResult)
        assert len(result.log_probs) == 1
        assert len(result.cross_entropies) == 1
        assert result.cross_entropies[0] > 0

    def test_log_probs_are_negative(self, tiny_model):
        result = tiny_model.score(
            prompts=["Hello"],
            continuations=[" world, this is a test"],
        )
        lp = result.log_probs[0]
        assert isinstance(lp, np.ndarray)
        assert np.all(lp <= 0)

    def test_multiple_pairs(self, tiny_model):
        result = tiny_model.score(
            prompts=["2 + 2 =", "The sky is"],
            continuations=[" 4", " blue"],
        )
        assert len(result.cross_entropies) == 2
        assert all(ce > 0 for ce in result.cross_entropies)

    def test_self_entropy_lower_than_cross(self, tiny_model):
        prompt = "The quick brown fox"
        result_self = tiny_model.generate([prompt], n_samples=1, max_new_tokens=20)
        own_continuation = result_self.completions[0][0]

        score_self = tiny_model.score([prompt], [own_continuation])
        score_random = tiny_model.score([prompt], [" xylophone tractor purple elephant"])

        assert score_self.cross_entropies[0] < score_random.cross_entropies[0]


class TestGetLogits:
    def test_basic(self, tiny_model):
        result = tiny_model.get_logits(["Hello world"], topk=50)
        assert isinstance(result, ForwardResult)
        assert len(result.logits) == 1
        assert result.logits[0].shape[-1] == 50

    def test_full_vocab(self, tiny_model):
        result = tiny_model.get_logits(["Test"], topk=0)
        vocab_size = tiny_model.tokenizer.vocab_size
        assert result.logits[0].shape[-1] >= vocab_size


class TestForwardWithHidden:
    def test_returns_hidden_states(self, tiny_model):
        result = tiny_model.forward_with_hidden(["Hello"], layers=[0, 1])
        assert isinstance(result, HiddenStatesResult)
        assert 0 in result.hidden_states
        assert 1 in result.hidden_states

    def test_all_layers(self, tiny_model):
        result = tiny_model.forward_with_hidden(["Hello"])
        n_layers = tiny_model._model.config.num_hidden_layers + 1
        assert len(result.hidden_states) == n_layers

    def test_hidden_shape(self, tiny_model):
        result = tiny_model.forward_with_hidden(["Hello"], layers=[0])
        h = result.hidden_states[0][0]
        assert isinstance(h, torch.Tensor)
        assert h.dim() == 2  # (seq_len, hidden_dim)
