"""HFEngine ↔ InferenceEngine score-equivalence test (v0.3.1 commit 2c).

The v0.3.0 backend cutover (planned for v0.4.0 internals) requires that
``HFEngine.score`` produces logprob arrays equivalent to v0.2.x
``InferenceEngine.score`` on the same inputs. This test pins that
contract using a tiny model that runs on CPU.

Tokenization invariant under verification (lm-eval convention):
    full_ids = [bos] + tokenize(prompt, add_special_tokens=False)
                     + tokenize(continuation, add_special_tokens=False)

If the new ``HFEngine.score`` ever drifts from this convention, the
calibration fixture for the future backend cutover will fail
byte-identity. This test catches such drift on every CI run, on CPU,
in a few seconds.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

# Marked as `slow` (not `gpu`): the model is small enough to run on CPU
# in ~10s but it does load real weights and import torch+transformers.
pytestmark = pytest.mark.slow


_TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


# A spread of (prompt, continuation) pairs that exercise different
# tokenization quirks (spaces, punctuation, repeats, multi-token continuations,
# leading-space sub-word merging risk).
_PAIRS = [
    ("Hello", " world"),
    ("Hello", "world"),                       # no leading space
    ("The capital of France is", " Paris"),
    ("2 + 2 =", " 4"),
    ("def fibonacci(n):", " return"),
    ("Q: What color is the sky?\nA:", " blue"),
    ("Once upon a", " time, there was a kingdom"),
    ("import numpy as", " np"),
    ("longest common subsequence", " problem"),
    ("the quick brown", " fox jumps over the lazy dog"),
    ("Mary had a little", " lamb"),
    ("A_B_C_D", "_E_F"),
    ("123", "456789"),
    ("\n\n", "\n\n\n"),
    ("hello world hello world", " hello world"),    # repeats
    # NOTE: empty-prompt and empty-continuation cases are excluded from
    # this test because they crash v0.2.x InferenceEngine on tokenizers
    # without an empty-prefix BOS (e.g. GPT-2 returns [] for tokenize("")).
    # HFEngine inherits the same constraint by design — see the
    # short-circuit in score() for empty continuations and the slice
    # arithmetic (prompt_len - 1) for empty prompts.
]


@pytest.fixture(scope="module")
def both_engines():
    """Construct one HFEngine + one InferenceEngine on the same model."""
    from lmdiff._config import Config as NewConfig
    from lmdiff._engine import HFEngine
    from lmdiff.config import Config as V02Config
    from lmdiff.engine import InferenceEngine

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # Force cpu + fp32 for both engines so the equivalence test is
        # independent of host CUDA/bf16 defaults — otherwise the v0.2.x
        # engine quietly auto-picks cuda+bf16 on a GPU host while HFEngine
        # sticks with cpu+fp32, and per-token logprobs differ at the
        # 0.01 level (bf16 quantization).
        v02_cfg = V02Config(model=_TINY_MODEL, dtype="float32")
    new_cfg = NewConfig(model=_TINY_MODEL)

    hf = HFEngine(new_cfg, device="cpu", dtype="fp32")
    inf = InferenceEngine(v02_cfg, device="cpu")
    yield hf, inf
    hf.close()


# ── Tokenization equivalence (cheap, no model forward) ───────────────


def test_tokenization_matches_v02_convention():
    """HFEngine.score must use the same prefix+continuation tokenization
    as v0.2.x InferenceEngine._encode_for_model + InferenceEngine.score
    continuation tokenization.
    """
    from lmdiff._engine import HFEngine
    from lmdiff._config import Config as NewConfig
    from lmdiff.config import Config as V02Config
    from lmdiff.engine import InferenceEngine

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        v02_cfg = V02Config(model=_TINY_MODEL, dtype="float32")
    hf = HFEngine(NewConfig(model=_TINY_MODEL), device="cpu", dtype="fp32")
    inf = InferenceEngine(v02_cfg, device="cpu")

    try:
        for prompt, cont in _PAIRS:
            if cont == "":
                continue  # empty-continuation tested separately
            v02_full_ids, _ = inf._encode_for_model(prompt)
            v02_cont_ids = inf._tokenizer(cont, add_special_tokens=False)["input_ids"]
            v02_full = list(v02_full_ids) + list(v02_cont_ids)

            # Reproduce HFEngine's internal tokenization without running
            # the forward pass.
            empty_prefix = hf._tokenizer(
                "", add_special_tokens=True,
            )["input_ids"]
            prompt_token_ids = hf._tokenizer(
                prompt, add_special_tokens=False,
            )["input_ids"]
            cont_ids = hf._tokenizer(
                cont, add_special_tokens=False,
            )["input_ids"]
            hf_full = list(empty_prefix) + list(prompt_token_ids) + list(cont_ids)

            assert hf_full == v02_full, (
                f"Tokenization diverged for prompt={prompt!r}, cont={cont!r}\n"
                f"  v02:     {v02_full}\n"
                f"  HFEngine: {hf_full}"
            )
    finally:
        hf.close()


# ── Per-token logprob equivalence (real forward) ────────────────────


def test_logprobs_match_within_1e_5(both_engines):
    """For every (prompt, continuation) pair, HFEngine.score and
    InferenceEngine.score must produce logprobs equal to within 1e-5."""
    hf, inf = both_engines
    import torch as _torch

    # Determinism: both engines load the same model fresh on CPU; same
    # weights; same forward; results should be bitwise equal modulo BLAS
    # nondeterminism. Loosen tolerance to 1e-5 for floating-point safety.
    _torch.manual_seed(0)

    for prompt, cont in _PAIRS:
        if cont == "":
            # InferenceEngine.score crashes on empty continuation
            # (it special-cases it as NaN, not an early empty return).
            continue
        hf_r = hf.score(prompt, cont)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inf_r = inf.score([prompt], continuations=[cont])

        # InferenceEngine returns lists; we want the first element.
        v02_logprobs = inf_r.log_probs[0]
        v02_tokens = inf_r.token_ids[0]

        # Same continuation tokens.
        assert hf_r.tokens == v02_tokens, (
            f"Continuation tokens differ for prompt={prompt!r}, cont={cont!r}\n"
            f"  v02:     {v02_tokens}\n"
            f"  HFEngine: {hf_r.tokens}"
        )

        # Same per-token logprobs.
        np.testing.assert_allclose(
            hf_r.logprobs, v02_logprobs, atol=1e-5, rtol=1e-5,
            err_msg=(
                f"Per-token logprobs diverged for prompt={prompt!r}, "
                f"cont={cont!r}"
            ),
        )

        # And therefore the same average (within rounding).
        # v02 reports cross_entropy = -mean(logprobs); HFEngine reports
        # avg_logprob = mean(logprobs). They differ in sign.
        v02_ce = inf_r.cross_entropies[0]
        assert abs(hf_r.avg_logprob - (-v02_ce)) < 1e-5


def test_continuation_ids_path_matches_str_path(both_engines):
    """Passing continuation_ids should produce the same logprobs as
    passing the string when the IDs were obtained from tokenizing that
    string with add_special_tokens=False (the v0.2.x convention)."""
    hf, _ = both_engines
    for prompt, cont in _PAIRS:
        if cont == "":
            continue
        cont_ids = hf._tokenizer(cont, add_special_tokens=False)["input_ids"]
        r_str = hf.score(prompt, cont)
        r_ids = hf.score(prompt, continuation_ids=list(cont_ids))
        np.testing.assert_array_equal(r_str.tokens, r_ids.tokens)
        np.testing.assert_allclose(r_str.logprobs, r_ids.logprobs, atol=1e-9)


def test_score_signature_validation(both_engines):
    """``score()`` must accept exactly one of continuation/continuation_ids."""
    hf, _ = both_engines
    with pytest.raises(ValueError, match="exactly one"):
        hf.score("hello", "world", continuation_ids=[1, 2])
    with pytest.raises(ValueError, match="exactly one"):
        hf.score("hello")
