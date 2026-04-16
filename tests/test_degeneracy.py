from modeldiff.metrics.output._degeneracy import is_degenerate_tokens


class TestIsDegenerateTokens:
    def test_high_repetition(self):
        assert is_degenerate_tokens([1, 1, 1, 1, 1, 1, 2]) is True  # 6/7 > 0.8

    def test_diverse(self):
        assert is_degenerate_tokens([1, 2, 3, 4, 5]) is False

    def test_empty(self):
        assert is_degenerate_tokens([]) is True

    def test_single_token(self):
        assert is_degenerate_tokens([1]) is True  # 1/1 = 100%

    def test_exact_threshold(self):
        # 8/10 = 0.8, threshold default 0.8 → True
        assert is_degenerate_tokens([1]*8 + [2, 3]) is True

    def test_just_below_threshold(self):
        # 7/10 = 0.7 < 0.8 → False
        assert is_degenerate_tokens([1]*7 + [2, 3, 4]) is False

    def test_custom_threshold(self):
        # 5/10 = 0.5, threshold 0.5 → True
        assert is_degenerate_tokens([1]*5 + [2]*5, threshold=0.5) is True
        # 4/10 = 0.4 < 0.5 → False
        assert is_degenerate_tokens([1]*4 + list(range(10, 16)), threshold=0.5) is False

    def test_newline_spam(self):
        # token 198 = \n in gpt2
        assert is_degenerate_tokens([198]*14 + [1, 2]) is True  # 14/16 = 0.875

    def test_all_same(self):
        assert is_degenerate_tokens([220]*16) is True
