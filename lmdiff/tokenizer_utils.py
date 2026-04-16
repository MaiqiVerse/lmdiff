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

    Compares vocab_size, class name, and encoding of canary strings.
    """
    if tok_a is tok_b:
        return True
    if tok_a.vocab_size != tok_b.vocab_size:
        return False
    if type(tok_a).__name__ != type(tok_b).__name__:
        return False
    for text in _CANARY_STRINGS:
        if tok_a.encode(text) != tok_b.encode(text):
            return False
    return True
