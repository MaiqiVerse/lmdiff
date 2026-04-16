import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modeldiff.config import Config
from modeldiff.metrics.base import MetricLevel, MetricResult
from modeldiff.metrics.output.behavioral_distance import BehavioralDistance


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_mock_engine(
    gen_outputs: list[str],
    score_map: dict[str, list[float]],
    token_ids_map: dict[str, list[list[int]]] | None = None,
    config: Config | None = None,
    tokenizer: MagicMock | None = None,
) -> MagicMock:
    """Build a mock engine with controlled generate/score outputs.

    score_map keys are "self" or "cross" — the mock uses call order to decide.
    """
    engine = MagicMock()
    engine.config = config or Config(model="mock-model")

    gen_result = MagicMock()
    gen_result.completions = [[o] for o in gen_outputs]
    engine.generate.return_value = gen_result

    score_results = []
    for key in score_map:
        sr = MagicMock()
        sr.cross_entropies = score_map[key]
        if token_ids_map and key in token_ids_map:
            sr.token_ids = token_ids_map[key]
        else:
            sr.token_ids = [[1, 2, 3]] * len(score_map[key])
        score_results.append(sr)

    engine.score.side_effect = score_results

    if tokenizer is not None:
        engine.tokenizer = tokenizer
    else:
        engine.tokenizer = MagicMock()

    return engine


class TestBDFormulaMock:
    """Hand-computed BD and Asymmetry from known CE values."""

    def test_known_values_single_probe(self):
        # ce_aa=2.0, ce_ab=3.0 (=CE(B,A)), ce_ba=2.8 (=CE(A,B)), ce_bb=1.9
        # BD = 0.5*(ce_ab-ce_bb) + 0.5*(ce_ba-ce_aa) = 0.5*(3.0-1.9) + 0.5*(2.8-2.0) = 0.95
        # Asym = (ce_ab-ce_aa) - (ce_ba-ce_bb) = (3.0-2.0) - (2.8-1.9) = 1.0 - 0.9 = 0.1
        probes = ["test probe"]

        engine_a = _make_mock_engine(
            gen_outputs=["output_a"],
            score_map={"aa": [2.0], "ba": [2.8]},
        )
        engine_b = _make_mock_engine(
            gen_outputs=["output_b"],
            score_map={"ab": [3.0], "bb": [1.9]},
        )

        # BD.compute calls:
        #   engine_a.generate(probes) -> outputs_a
        #   engine_b.generate(probes) -> outputs_b
        #   engine_a.score(probes, outputs_a) -> score_aa  (1st call on engine_a)
        #   engine_b.score(probes, outputs_a) -> score_ab  (1st call on engine_b)
        #   engine_a.score(probes, outputs_b) -> score_ba  (2nd call on engine_a)
        #   engine_b.score(probes, outputs_b) -> score_bb  (2nd call on engine_b)
        metric = BehavioralDistance()
        result = metric.compute(engine_a, engine_b, probes)

        assert isinstance(result, MetricResult)
        assert result.name == "behavioral_distance"
        assert result.level == MetricLevel.OUTPUT
        assert abs(result.value - 0.95) < 1e-6
        assert abs(result.details["asymmetry"] - 0.1) < 1e-6
        assert abs(result.details["ce_aa"] - 2.0) < 1e-6
        assert abs(result.details["ce_ab"] - 3.0) < 1e-6
        assert abs(result.details["ce_ba"] - 2.8) < 1e-6
        assert abs(result.details["ce_bb"] - 1.9) < 1e-6

    def test_multiple_probes_averaged(self):
        # probe 0: ce_aa=1.0, ce_ab=2.0, ce_ba=1.5, ce_bb=0.8
        #   bd_0 = 0.5*(2.0-0.8) + 0.5*(1.5-1.0) = 0.6 + 0.25 = 0.85
        # probe 1: ce_aa=3.0, ce_ab=4.0, ce_ba=3.5, ce_bb=2.5
        #   bd_1 = 0.5*(4.0-2.5) + 0.5*(3.5-3.0) = 0.75 + 0.25 = 1.0
        # BD = (0.85 + 1.0) / 2 = 0.925
        probes = ["p0", "p1"]

        engine_a = _make_mock_engine(
            gen_outputs=["out_a0", "out_a1"],
            score_map={"aa": [1.0, 3.0], "ba": [1.5, 3.5]},
        )
        engine_b = _make_mock_engine(
            gen_outputs=["out_b0", "out_b1"],
            score_map={"ab": [2.0, 4.0], "bb": [0.8, 2.5]},
        )

        result = BehavioralDistance().compute(engine_a, engine_b, probes)
        assert abs(result.value - 0.925) < 1e-6
        assert len(result.details["per_prompt"]) == 2


class TestBDSymmetry:
    def test_bd_ab_equals_bd_ba(self):
        # BD is symmetric by construction: BD(A,B) = BD(B,A)
        probes = ["p"]

        engine_a = _make_mock_engine(
            gen_outputs=["oa"],
            score_map={"aa": [2.0], "ba": [3.0]},
        )
        engine_b = _make_mock_engine(
            gen_outputs=["ob"],
            score_map={"ab": [2.5], "bb": [1.5]},
        )
        bd_ab = BehavioralDistance().compute(engine_a, engine_b, probes).value

        # Now swap: engine_b is "A", engine_a is "B"
        engine_b2 = _make_mock_engine(
            gen_outputs=["ob"],
            score_map={"aa_new": [1.5], "ba_new": [2.5]},
        )
        engine_a2 = _make_mock_engine(
            gen_outputs=["oa"],
            score_map={"ab_new": [3.0], "bb_new": [2.0]},
        )
        bd_ba = BehavioralDistance().compute(engine_b2, engine_a2, probes).value

        assert abs(bd_ab - bd_ba) < 1e-6


class TestBDSelfDistance:
    def test_self_distance_zero(self):
        # Same engine passed as both A and B.
        # score is called 4 times on the same mock: aa, ab(=aa), ba(=bb), bb
        # All CEs equal → BD = 0
        probes = ["p"]

        engine = _make_mock_engine(
            gen_outputs=["out"],
            score_map={"aa": [2.0], "ab": [2.0], "ba": [2.0], "bb": [2.0]},
        )

        result = BehavioralDistance().compute(engine, engine, probes)
        assert abs(result.value) < 1e-6


class TestBDNonNegative:
    def test_non_negative_jensen(self):
        # By Jensen's inequality, cross-entropy >= self-entropy
        # So CE(A,B) >= CE(A,A) and CE(B,A) >= CE(B,B), thus BD >= 0
        probes = ["p"]

        engine_a = _make_mock_engine(
            gen_outputs=["oa"],
            score_map={"aa": [2.0], "ba": [2.5]},
        )
        engine_b = _make_mock_engine(
            gen_outputs=["ob"],
            score_map={"ab": [3.0], "bb": [1.8]},
        )

        result = BehavioralDistance().compute(engine_a, engine_b, probes)
        assert result.value >= -1e-3


class TestBDBpbNormalization:
    def test_bpb_path_when_tokenizers_differ(self):
        probes = ["hi"]

        config_a = Config(model="model-a")
        config_b = Config(model="model-b")

        engine_a = _make_mock_engine(
            gen_outputs=[" world"],
            score_map={"aa": [1.0], "ba": [1.5]},
            token_ids_map={"aa": [[1, 2]], "ba": [[10, 20, 30]]},
            config=config_a,
        )
        engine_b = _make_mock_engine(
            gen_outputs=[" earth"],
            score_map={"ab": [2.0], "bb": [0.8]},
            token_ids_map={"ab": [[5, 6, 7]], "bb": [[40, 50]]},
            config=config_b,
        )

        with patch(
            "modeldiff.metrics.output.behavioral_distance.tokenizers_equivalent",
            return_value=False,
        ):
            result = BehavioralDistance().compute(engine_a, engine_b, probes)

        assert result.details["bpb_normalized"] is True
        pp = result.details["per_prompt"][0]
        # BPB uses continuation text only: " world" (6 bytes), " earth" (6 bytes)
        # ce_aa: bpb = (1.0 * 2 / log2) / 6
        # ce_ab: bpb = (2.0 * 3 / log2) / 6
        # ce_ba: bpb = (1.5 * 3 / log2) / 6
        # ce_bb: bpb = (0.8 * 2 / log2) / 6
        log2 = math.log(2)
        exp_aa = (1.0 * 2 / log2) / 6
        exp_ab = (2.0 * 3 / log2) / 6
        exp_ba = (1.5 * 3 / log2) / 6
        exp_bb = (0.8 * 2 / log2) / 6
        assert abs(pp["ce_aa"] - exp_aa) < 1e-6
        assert abs(pp["ce_ab"] - exp_ab) < 1e-6
        assert abs(pp["ce_ba"] - exp_ba) < 1e-6
        assert abs(pp["ce_bb"] - exp_bb) < 1e-6

        exp_bd = 0.5 * (exp_ab - exp_bb) + 0.5 * (exp_ba - exp_aa)
        assert abs(result.value - exp_bd) < 1e-6

    def test_no_bpb_when_same_tokenizer(self):
        probes = ["hi"]

        config = Config(model="gpt2")
        engine_a = _make_mock_engine(
            gen_outputs=[" out"],
            score_map={"aa": [1.0], "ba": [1.5]},
            config=config,
        )
        engine_b = _make_mock_engine(
            gen_outputs=[" out"],
            score_map={"ab": [2.0], "bb": [0.8]},
            config=config,
        )

        result = BehavioralDistance().compute(engine_a, engine_b, probes)
        assert result.details["bpb_normalized"] is False


class TestBDRequirements:
    def test_requirements(self):
        reqs = BehavioralDistance.requirements()
        assert reqs["generations"] is True
        assert reqs["logits"] is False
        assert reqs["hidden_states"] is False

    def test_is_applicable(self):
        assert BehavioralDistance.is_applicable(None, None)


class TestBDArchitecture:
    def test_no_transformers_import(self):
        import modeldiff.metrics.output.behavioral_distance as mod
        import inspect
        source = inspect.getsource(mod)
        assert "import transformers" not in source

    def test_no_other_metric_import(self):
        import modeldiff.metrics.output.behavioral_distance as mod
        import inspect
        source = inspect.getsource(mod)
        assert "from modeldiff.metrics.output.token_entropy" not in source
        assert "from modeldiff.metrics.output.token_kl" not in source


@pytest.mark.slow
class TestBDSmokeReal:
    def test_gpt2_self_bd_near_zero(self, tiny_model):
        probes = ["The capital of France is", "2 + 2 =", "Once upon a time"]
        result = BehavioralDistance().compute(
            tiny_model, tiny_model, probes, max_new_tokens=16,
        )
        assert abs(result.value) < 0.01

    def test_gpt2_vs_distilgpt2_positive(self, tiny_model, distil_engine):
        probes = ["The capital of France is", "2 + 2 =", "Once upon a time"]
        result = BehavioralDistance().compute(
            tiny_model, distil_engine, probes, max_new_tokens=16,
        )
        print(f"\n=== BD(gpt2, distilgpt2) = {result.value:.6f} ===")
        print(f"    ce_aa={result.details['ce_aa']:.4f}  ce_ab={result.details['ce_ab']:.4f}")
        print(f"    ce_ba={result.details['ce_ba']:.4f}  ce_bb={result.details['ce_bb']:.4f}")
        print(f"    asymmetry={result.details['asymmetry']:.6f}")
        for pp in result.details["per_prompt"]:
            print(f"    [{pp['probe'][:30]}] bd={pp['bd']:.4f} asym={pp['asymmetry']:.4f}")
        assert result.value > 0
        assert result.details["bpb_normalized"] is False
