from __future__ import annotations

import math
from typing import Any

_CANARY_STRINGS = [
    "hello world",
    "2+2=4",
    "日本語",
    "def f():",
]


def utf8_byte_count(text: str) -> int:
    """Return the number of UTF-8 bytes in a string."""
    return len(text.encode("utf-8"))


def bpb_from_ce(cross_entropy: float, n_tokens: int, text: str) -> float:
    """Convert per-token cross-entropy (nats) to bits-per-byte.

    Formula: bpb = (ce * n_tokens / log(2)) / utf8_byte_count(text)
    """
    byte_count = utf8_byte_count(text)
    if byte_count == 0:
        return 0.0
    return (cross_entropy * n_tokens / math.log(2)) / byte_count


def tokenizers_equivalent(tok_a: Any, tok_b: Any) -> bool:
    """Check if two tokenizers are substantively equivalent.

    Compares vocab_size and the id sequences produced on a set of canary
    strings with special tokens disabled — so slow/fast variants of the
    same tokenizer (e.g. LlamaTokenizer vs LlamaTokenizerFast) are treated
    as equivalent, and tokenizers whose `encode()` default differs on
    `add_special_tokens` are not falsely flagged as different (L-011).
    """
    if tok_a is tok_b:
        return True
    if tok_a.vocab_size != tok_b.vocab_size:
        return False
    for text in _CANARY_STRINGS:
        ids_a = tok_a(text, add_special_tokens=False)["input_ids"]
        ids_b = tok_b(text, add_special_tokens=False)["input_ids"]
        if ids_a != ids_b:
            return False
    return True
