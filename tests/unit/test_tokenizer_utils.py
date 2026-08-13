import math

import pytest

from lmdiff.tokenizer_utils import bpb_from_ce, tokenizers_equivalent, utf8_byte_count


class TestUtf8ByteCount:
    def test_ascii(self):
        assert utf8_byte_count("hello") == 5

    def test_cjk(self):
        assert utf8_byte_count("日本") == 6

    def test_empty(self):
        assert utf8_byte_count("") == 0

    def test_mixed(self):
        assert utf8_byte_count("a日") == 1 + 3

    def test_emoji(self):
        assert utf8_byte_count("🎉") == 4


class TestBpbFromCe:
    def test_known_value(self):
        # CE=0.693 nats, 1 token, text="ab" (2 bytes)
        # bpb = (0.693 * 1 / log(2)) / 2 = 1.0 / 2 = 0.5
        bpb = bpb_from_ce(0.693, 1, "ab")
        expected = (0.693 / math.log(2)) / 2
        assert abs(bpb - expected) < 1e-4
        assert abs(bpb - 0.5) < 0.001

    def test_multiple_tokens(self):
        # CE=2.0 nats/token, 5 tokens, text="hello" (5 bytes)
        # bpb = (2.0 * 5 / log(2)) / 5 = 2.0 / log(2) ≈ 2.885
        bpb = bpb_from_ce(2.0, 5, "hello")
        expected = 2.0 / math.log(2)
        assert abs(bpb - expected) < 1e-4

    def test_cjk_text(self):
        # CE=1.0, 2 tokens, text="日本" (6 bytes)
        # bpb = (1.0 * 2 / log(2)) / 6
        bpb = bpb_from_ce(1.0, 2, "日本")
        expected = (1.0 * 2 / math.log(2)) / 6
        assert abs(bpb - expected) < 1e-6

    def test_zero_ce(self):
        assert bpb_from_ce(0.0, 10, "hello") == 0.0

    def test_empty_text(self):
        assert bpb_from_ce(1.0, 1, "") == 0.0


class TestTokenizersEquivalent:
    def test_gpt2_vs_gpt2(self):
        from transformers import AutoTokenizer
        tok_a = AutoTokenizer.from_pretrained("gpt2")
        tok_b = AutoTokenizer.from_pretrained("gpt2")
        assert tokenizers_equivalent(tok_a, tok_b)

    def test_gpt2_vs_distilgpt2(self):
        from transformers import AutoTokenizer
        tok_a = AutoTokenizer.from_pretrained("gpt2")
        tok_b = AutoTokenizer.from_pretrained("distilgpt2")
        assert tokenizers_equivalent(tok_a, tok_b)

    def test_gpt2_vs_tiny_gpt2(self):
        from transformers import AutoTokenizer
        tok_a = AutoTokenizer.from_pretrained("gpt2")
        tok_b = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        assert tokenizers_equivalent(tok_a, tok_b)

    def test_same_object(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        assert tokenizers_equivalent(tok, tok)

    @pytest.mark.slow
    def test_gpt2_vs_llama2(self):
        from transformers import AutoTokenizer
        tok_a = AutoTokenizer.from_pretrained("gpt2")
        tok_b = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        assert not tokenizers_equivalent(tok_a, tok_b)

    @pytest.mark.slow
    def test_llama2_slow_vs_fast(self):
        """Slow and fast variants of the same tokenizer are equivalent (L-011)."""
        from transformers import AutoTokenizer
        tok_slow = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
        tok_fast = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
        # Sanity: the two loads genuinely produce different Python classes.
        assert type(tok_slow).__name__ != type(tok_fast).__name__
        assert tokenizers_equivalent(tok_slow, tok_fast)
