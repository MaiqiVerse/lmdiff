from pathlib import Path

import pytest

from lmdiff.probes.loader import ProbeSet
from lmdiff.tasks.base import Task
from lmdiff.tasks.evaluators import ContainsAnswer

V01_PATH = Path(__file__).parent.parent / "lmdiff" / "probes" / "v01.json"


@pytest.mark.slow
class TestTaskE2E:
    def test_gpt2_contains_answer(self, tiny_model):
        ps = ProbeSet.from_json(V01_PATH)
        task = Task("v01-contains", ps, ContainsAnswer(), max_new_tokens=16)
        result = task.run(tiny_model)

        assert result.n_probes == 90
        assert 0.0 <= result.accuracy <= 1.0
        assert set(result.per_domain.keys()) == {"math", "knowledge", "code"}

        print(f"\n=== gpt2 ContainsAnswer on v01 ===")
        print(f"  overall: {result.accuracy:.1%} ({result.n_correct}/{result.n_probes})")
        for d in sorted(result.per_domain):
            info = result.per_domain[d]
            print(f"  {d:12s}: {info['accuracy']:.1%} ({info['correct']}/{info['n']})")

    def test_distilgpt2_contains_answer(self, distil_engine):
        ps = ProbeSet.from_json(V01_PATH)
        task = Task("v01-contains", ps, ContainsAnswer(), max_new_tokens=16)
        result = task.run(distil_engine)

        assert result.n_probes == 90
        assert 0.0 <= result.accuracy <= 1.0

        print(f"\n=== distilgpt2 ContainsAnswer on v01 ===")
        print(f"  overall: {result.accuracy:.1%} ({result.n_correct}/{result.n_probes})")
        for d in sorted(result.per_domain):
            info = result.per_domain[d]
            print(f"  {d:12s}: {info['accuracy']:.1%} ({info['correct']}/{info['n']})")
