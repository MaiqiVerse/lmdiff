from __future__ import annotations

import re

from lmdiff.tasks.base import BaseEvaluator


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

        text = output.strip()
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
