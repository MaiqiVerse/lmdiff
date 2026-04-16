from __future__ import annotations

from collections import Counter


def is_degenerate_tokens(token_ids: list[int], threshold: float = 0.8) -> bool:
    """True if >=threshold of tokens are a single repeated id.

    Also True for empty sequences.
    """
    if not token_ids:
        return True
    counts = Counter(token_ids)
    most_common_freq = counts.most_common(1)[0][1] / len(token_ids)
    return most_common_freq >= threshold
