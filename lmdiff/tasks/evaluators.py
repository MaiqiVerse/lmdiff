from __future__ import annotations

import math
import re
import string
from typing import ClassVar

from lmdiff.tasks.base import BaseEvaluator

__all__ = [
    "ExactMatch",
    "ContainsAnswer",
    "MultipleChoice",
    "F1",
    "Gsm8kNumberMatch",
]


class ExactMatch(BaseEvaluator):
    """Strict equality after stripping whitespace."""
    name = "exact_match"

    def __init__(self, case_sensitive: bool = True, strip: bool = True) -> None:
        self.case_sensitive = case_sensitive
        self.strip = strip

    def evaluate(self, output, expected, probe_metadata=None):
        if expected is None:
            return False, 0.0, {"reason": "no_expected"}
        o = output.strip() if self.strip else output
        e = expected.strip() if self.strip else expected
        if not self.case_sensitive:
            o, e = o.lower(), e.lower()
        correct = o == e
        return correct, float(correct), {}


class ContainsAnswer(BaseEvaluator):
    """Checks whether expected string appears anywhere in output."""
    name = "contains_answer"

    def __init__(self, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive

    def evaluate(self, output, expected, probe_metadata=None):
        if expected is None:
            return False, 0.0, {"reason": "no_expected"}
        if expected == "":
            return False, 0.0, {"reason": "empty_expected"}
        o = output if self.case_sensitive else output.lower()
        e = expected if self.case_sensitive else expected.lower()
        pos = o.find(e)
        correct = pos >= 0
        return correct, float(correct), {"position": pos}


class MultipleChoice(BaseEvaluator):
    """For MC probes with letter/number choices.

    Expects probe_metadata with 'correct_index': int.
    Parses first letter A-Z or first integer from output.
    """
    name = "multiple_choice"

    def evaluate(self, output, expected, probe_metadata=None):
        if probe_metadata is None or "correct_index" not in probe_metadata:
            return False, 0.0, {"reason": "missing_mc_metadata"}

        text = output.strip().upper()
        letter_match = re.search(r"\b([A-Z])\b", text)
        if letter_match:
            predicted = ord(letter_match.group(1)) - ord("A")
        else:
            num_match = re.search(r"\b(\d+)\b", text)
            if num_match:
                predicted = int(num_match.group(1))
            else:
                return False, 0.0, {"reason": "no_choice_parsed"}

        correct_idx = probe_metadata["correct_index"]
        correct = predicted == correct_idx
        return correct, float(correct), {"predicted_index": predicted}


class F1(BaseEvaluator):
    """SQuAD-style token-overlap F1.

    Normalization: lowercase, strip articles (a/an/the), strip punctuation,
    collapse whitespace. Tokenize by whitespace.

    Multi-target: if probe_metadata contains 'aliases' (list[str]), score
    against [expected] + aliases and take the max.

    `correct` = score >= F1_THRESHOLD (default 0.5).
    """

    name = "f1"
    F1_THRESHOLD: ClassVar[float] = 0.5
    _ARTICLES: ClassVar[frozenset[str]] = frozenset({"a", "an", "the"})
    _PUNCT_TABLE: ClassVar[dict[int, None]] = str.maketrans("", "", string.punctuation)

    def evaluate(self, output, expected, probe_metadata=None):
        if expected is None:
            return False, 0.0, {"reason": "no_expected"}

        targets: list[str] = [expected]
        if probe_metadata and isinstance(probe_metadata.get("aliases"), list):
            targets.extend(probe_metadata["aliases"])

        pred_tokens = self._tokenize(output)
        per_target = [
            (t, self._f1_pair(pred_tokens, self._tokenize(t))) for t in targets
        ]
        best_target, best_score = max(per_target, key=lambda x: x[1])
        correct = best_score >= self.F1_THRESHOLD
        return correct, float(best_score), {
            "f1": float(best_score),
            "best_target": best_target,
            "n_targets": len(targets),
        }

    @classmethod
    def _normalize(cls, s: str) -> str:
        lowered = s.lower()
        stripped = lowered.translate(cls._PUNCT_TABLE)
        tokens = [tok for tok in stripped.split() if tok not in cls._ARTICLES]
        return " ".join(tokens)

    @classmethod
    def _tokenize(cls, s: str) -> list[str]:
        return cls._normalize(s).split()

    @staticmethod
    def _f1_pair(pred: list[str], gold: list[str]) -> float:
        if not pred and not gold:
            return 1.0
        if not pred or not gold:
            return 0.0
        # SQuAD official: token-level overlap (multiset intersection).
        from collections import Counter
        pred_counts = Counter(pred)
        gold_counts = Counter(gold)
        overlap = sum((pred_counts & gold_counts).values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred)
        recall = overlap / len(gold)
        return 2 * precision * recall / (precision + recall)


class Gsm8kNumberMatch(BaseEvaluator):
    """Extract the last numeric answer and exact-match it against gold.

    Extraction order (lm-eval convention):
      1. Number following the last '####' marker.
      2. Otherwise, the last number-like token in the string.
      3. Thousands separators stripped ('1,234' → '1234').
    Comparison via math.isclose (rel_tol=1e-6, abs_tol=1e-9).
    `correct` is True iff pred and gold both extract successfully and match.
    `score` is 0.0 or 1.0.
    """

    name = "gsm8k_number_match"
    _NUMBER_RE: ClassVar = re.compile(r"-?\d[\d,]*\.?\d*")
    _HASH_RE: ClassVar = re.compile(r"####\s*(-?\d[\d,]*\.?\d*)")

    def evaluate(self, output, expected, probe_metadata=None):
        if expected is None:
            return False, 0.0, {"reason": "no_expected"}
        pred_num = self._extract(output)
        gold_num = self._extract(expected)
        if pred_num is None or gold_num is None:
            return False, 0.0, {
                "pred_num": pred_num,
                "gold_num": gold_num,
                "reason": "extraction_failed",
            }
        match = math.isclose(pred_num, gold_num, rel_tol=1e-6, abs_tol=1e-9)
        return match, 1.0 if match else 0.0, {
            "pred_num": pred_num,
            "gold_num": gold_num,
        }

    @classmethod
    def _extract(cls, s: str) -> float | None:
        if not isinstance(s, str) or not s:
            return None
        hash_matches = cls._HASH_RE.findall(s)
        if hash_matches:
            raw = hash_matches[-1]
        else:
            num_matches = cls._NUMBER_RE.findall(s)
            if not num_matches:
                return None
            raw = num_matches[-1]
        cleaned = raw.replace(",", "").rstrip(".")
        if not cleaned or cleaned in ("-", "."):
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
