import math

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
    def test_loads_model(self, engine):
        assert engine._model is not None
        assert engine._tokenizer is not None

    def test_pad_token_set(self, engine):
        assert engine.tokenizer.pad_token is not None

    def test_eval_mode(self, engine):
        assert not engine._model.training


class TestGenerate:
    def test_single_prompt_single_sample(self, engine):
        result = engine.generate(["Hello world"], n_samples=1, max_new_tokens=10)
        assert isinstance(result, GenerationResult)
        assert len(result.prompts) == 1
        assert len(result.completions) == 1
        assert len(result.completions[0]) == 1
        assert isinstance(result.completions[0][0], str)
        assert len(result.completions[0][0]) > 0

    def test_multiple_prompts(self, engine):
        prompts = ["The sky is", "One plus one equals"]
        result = engine.generate(prompts, n_samples=1, max_new_tokens=10)
        assert len(result.completions) == 2

    def test_multiple_samples_with_sampling(self, engine):
        config = engine.config.with_override(
            decode={"strategy": "sample", "temperature": 0.7},
        )
        eng = InferenceEngine(config)
        result = eng.generate(["Hello"], n_samples=3, max_new_tokens=10)
        assert len(result.completions[0]) == 3

    def test_multiple_samples_greedy_raises(self, engine):
        with pytest.raises(ValueError, match="n_samples > 1"):
            engine.generate(["Hello"], n_samples=3, max_new_tokens=10)

    def test_token_ids_populated(self, engine):
        result = engine.generate(["Test"], n_samples=1, max_new_tokens=10)
        assert result.token_ids is not None
        assert len(result.token_ids[0][0]) > 0


class TestScore:
    def test_basic_score(self, engine):
        result = engine.score(
            prompts=["The capital of France is"],
            continuations=[" Paris"],
        )
        assert isinstance(result, ForwardResult)
        assert len(result.log_probs) == 1
        assert len(result.cross_entropies) == 1
        assert result.cross_entropies[0] > 0

    def test_log_probs_are_negative(self, engine):
        result = engine.score(
            prompts=["Hello"],
            continuations=[" world, this is a test"],
        )
        lp = result.log_probs[0]
        assert isinstance(lp, np.ndarray)
        assert np.all(lp <= 0)

    def test_multiple_pairs(self, engine):
        result = engine.score(
            prompts=["2 + 2 =", "The sky is"],
            continuations=[" 4", " blue"],
        )
        assert len(result.cross_entropies) == 2
        assert all(ce > 0 for ce in result.cross_entropies)

    def test_self_entropy_lower_than_cross(self, engine):
        prompt = "The quick brown fox"
        gen = engine.generate([prompt], n_samples=1, max_new_tokens=20)
        own_continuation = gen.completions[0][0]

        score_self = engine.score([prompt], [own_continuation])
        score_random = engine.score([prompt], [" xylophone tractor purple elephant"])

        assert score_self.cross_entropies[0] < score_random.cross_entropies[0]


class TestScoreWithIds:
    def test_ids_vs_string_match(self, engine):
        prompt = "The quick brown fox"
        gen = engine.generate([prompt], n_samples=1, max_new_tokens=16)
        ids = gen.token_ids[0][0]
        text = gen.completions[0][0]

        score_ids = engine.score([prompt], continuation_ids=[ids])
        score_str = engine.score([prompt], continuations=[text])

        assert abs(score_ids.cross_entropies[0] - score_str.cross_entropies[0]) < 1e-4

    def test_no_args_raises(self, engine):
        with pytest.raises(ValueError, match="exactly one"):
            engine.score(["test"])

    def test_both_args_raises(self, engine):
        with pytest.raises(ValueError, match="exactly one"):
            engine.score(["test"], continuations=["x"], continuation_ids=[[1]])

    def test_empty_continuation_returns_nan(self, engine):
        result = engine.score(["Hello"], continuations=[""])
        assert math.isnan(result.cross_entropies[0])
        assert result.log_probs[0].shape == (0,)
        assert result.token_ids[0] == []

    def test_empty_ids_returns_nan(self, engine):
        result = engine.score(["Hello"], continuation_ids=[[]])
        assert math.isnan(result.cross_entropies[0])


class TestGetLogits:
    def test_basic(self, engine):
        result = engine.get_logits(["Hello world"], topk=50)
        assert isinstance(result, ForwardResult)
        assert len(result.logits) == 1
        assert result.logits[0].shape[-1] == 50

    def test_topk_token_ids_shape(self, engine):
        topk = 50
        result = engine.get_logits(["Hello world"], topk=topk)
        seq_len = result.logits[0].shape[0]
        assert len(result.token_ids[0]) == seq_len
        assert len(result.token_ids[0][0]) == topk

    def test_full_vocab(self, engine):
        result = engine.get_logits(["Test"], topk=0)
        vocab_size = engine.tokenizer.vocab_size
        assert result.logits[0].shape[-1] >= vocab_size


class TestForwardWithHidden:
    def test_returns_hidden_states(self, engine):
        result = engine.forward_with_hidden(["Hello"], layers=[0, 1])
        assert isinstance(result, HiddenStatesResult)
        assert 0 in result.hidden_states
        assert 1 in result.hidden_states

    def test_all_layers(self, engine):
        result = engine.forward_with_hidden(["Hello"])
        n_layers = engine._model.config.num_hidden_layers + 1
        assert len(result.hidden_states) == n_layers

    def test_hidden_shape(self, engine):
        result = engine.forward_with_hidden(["Hello"], layers=[0])
        h = result.hidden_states[0][0]
        assert isinstance(h, torch.Tensor)
        assert h.dim() == 2


class TestContextPrepend:
    def test_context_prepended(self, gpt2_config_with_context):
        eng = InferenceEngine(gpt2_config_with_context)
        built = eng._build_prompt("Tell me more")
        assert "You are helpful." in built
        assert "Hello" in built
        assert "Tell me more" in built


@pytest.mark.slow
class TestCrossModel:
    """Tests that compare behavior across gpt2 and llama2."""

    def test_different_tokenizers(self, tiny_model, llama_engine):
        from modeldiff.tokenizer_utils import tokenizers_equivalent
        assert tiny_model.config.shares_tokenizer_with(llama_engine.config) is None
        assert not tokenizers_equivalent(tiny_model.tokenizer, llama_engine.tokenizer)

    def test_both_generate(self, tiny_model, llama_engine):
        for eng in [tiny_model, llama_engine]:
            result = eng.generate(["The meaning of life is"], n_samples=1, max_new_tokens=10)
            assert len(result.completions[0][0]) > 0

    def test_both_score(self, tiny_model, llama_engine):
        for eng in [tiny_model, llama_engine]:
            result = eng.score(["The sky is"], [" blue"])
            assert result.cross_entropies[0] > 0
