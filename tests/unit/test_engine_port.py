"""The task layer against the Engine Protocol. v0.4.5.

PHASE_PLAN Z.4 item 6. Scope: ``docs/internal/v045_engine_port_notes.md``.

**The two tests this file exists for are in `TestConfigReachesTheEngine`.**
The port's failure mode is not a crash. ``InferenceEngine`` resolved
``system_prompt`` / ``context`` / ``decode`` from its *stored config*;
the Protocol's engines are stateless. So the obvious translation --

    engine.generate(texts, n_samples=1, max_new_tokens=N)
      ->  [engine.generate(t, max_new_tokens=N).text for t in texts]

-- silently drops both. A ``system_prompt`` variant loses its scaffold,
and a ``temperature=1.5`` variant becomes **greedy**, because
``HFEngine.generate`` computes ``do_sample`` from ``temperature != 1.0``
and the default is ``1.0``. Nothing raises. The report names a
configuration that was never measured, which is the v0.4.0 cutover's
failure mode exactly.

So those two assert on *what reached the engine*, not on any computed
value -- there is no computed value that differs.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from lmdiff._config import Config, DecodeSpec  # noqa: E402
from lmdiff._prompting import (  # noqa: E402
    assemble_prompt,
    generate_kwargs,
    prefix_text,
)
from lmdiff.probes.loader import Probe, ProbeSet  # noqa: E402
from lmdiff.tasks.base import Task  # noqa: E402
from lmdiff.tasks.evaluators import ExactMatch  # noqa: E402


class _Recorder:
    """Engine Protocol surface that records every call verbatim."""

    name = "recorder"

    def __init__(self, outputs: list[str] | None = None) -> None:
        self.calls: list[dict] = []
        self._outputs = list(outputs or [])

    def generate(self, prompt, *, prefix_text="", **kw):
        self.calls.append({"prompt": prompt, "prefix_text": prefix_text, **kw})
        text = self._outputs.pop(0) if self._outputs else ""
        return type("R", (), {"text": text, "tokens": []})()

    def score(self, prompt, continuation, *, prefix_text=""):
        self.calls.append({
            "prompt": prompt, "continuation": continuation,
            "prefix_text": prefix_text,
        })
        return type("S", (), {"avg_logprob": -1.0, "tokens": [0]})()


PROBES = ProbeSet([
    Probe(id="a", text="17 + 25 = ", domain="math", expected="42"),
    Probe(id="b", text="7 * 8 = ", domain="math", expected="56"),
])


# ── the two that matter ──────────────────────────────────────────────


class TestConfigReachesTheEngine:
    def test_sampling_kwargs_reach_generate(self):
        """A ``temperature=1.5`` variant must be generated at 1.5.

        Under a naive port the engine is called with no decode kwargs
        at all, HFEngine's defaults make ``do_sample=False``, and the
        variant is measured **greedy**. The accuracy that comes back is
        a real number for the wrong configuration, so no assertion on
        an accuracy could catch this -- only one on the call.
        """
        cfg = Config(
            model="gpt2", name="temp_1.5",
            decode=DecodeSpec(strategy="sample", temperature=1.5, top_p=0.95),
        )
        eng = _Recorder(["42", "56"])
        Task("t", PROBES, ExactMatch()).run(
            eng,
            prefix_text=prefix_text(cfg),
            generate_kwargs=generate_kwargs(cfg, max_new_tokens=16),
        )

        assert len(eng.calls) == 2
        for call in eng.calls:
            assert call["temperature"] == 1.5, (
                "sampling temperature did not reach the engine; the "
                "variant would be measured greedy"
            )
            assert call["top_p"] == 0.95
            # top_k=0 is not a no-op: HF defaults it to 50 when omitted,
            # which truncates the sampling distribution (L-030 / v0.4.0).
            assert call["top_k"] == 0
            assert call["max_new_tokens"] == 16

    def test_scaffold_is_prepended(self):
        """A ``system_prompt`` variant must be generated with it.

        Under a naive port ``prefix_text`` stays ``""`` and the model
        sees a bare probe. Again: plausible numbers, wrong
        configuration.
        """
        cfg = Config(
            model="gpt2", name="chat",
            system_prompt="You are a helpful assistant.",
        )
        eng = _Recorder(["42", "56"])
        Task("t", PROBES, ExactMatch()).run(
            eng,
            prefix_text=prefix_text(cfg),
            generate_kwargs=generate_kwargs(cfg, max_new_tokens=16),
        )

        assert len(eng.calls) == 2
        for call, probe in zip(eng.calls, PROBES):
            assert call["prefix_text"] == "You are a helpful assistant.\n", (
                "scaffold did not reach the engine; the variant would be "
                "measured without its system prompt"
            )
            # The probe itself is unchanged -- the engine concatenates,
            # split-tokenizing so the probe span stays byte-aligned
            # (L-030). Prepending here instead would defeat that.
            assert call["prompt"] == probe.text

    def test_greedy_config_sends_no_decode_kwargs(self):
        """The negative case, so the two above cannot pass by a stub
        that echoes whatever it is handed."""
        eng = _Recorder(["42", "56"])
        cfg = Config(model="gpt2")
        Task("t", PROBES, ExactMatch()).run(
            eng,
            prefix_text=prefix_text(cfg),
            generate_kwargs=generate_kwargs(cfg, max_new_tokens=16),
        )
        for call in eng.calls:
            assert call["prefix_text"] == ""
            assert "temperature" not in call
            assert call["max_new_tokens"] == 16


# ── the shared extraction ────────────────────────────────────────────


class TestPromptingIsOneDefinition:
    def test_pipeline_uses_the_shared_module(self):
        """``_pipeline`` imports rather than defines. A second copy is
        two things that agree today (L-035), and the detail most likely
        to be lost is the trailing newline below."""
        from lmdiff import _pipeline, _prompting

        assert _pipeline._prefix_text is _prompting.prefix_text
        assert _pipeline._generate_kwargs is _prompting.generate_kwargs
        assert _pipeline._assemble_prompt is _prompting.assemble_prompt

    def test_prefix_keeps_its_trailing_newline(self):
        """Load-bearing for byte-equivalence with the v0.2.x calibration
        baseline. The docstring says don't strip; this makes it fail."""
        cfg = Config(model="gpt2", system_prompt="Be concise.")
        assert prefix_text(cfg) == "Be concise.\n"

    def test_prompting_imports_no_engine(self):
        """Pure functions of a Config — no torch, no engine, so both the
        pipeline and the task layer can depend on it."""
        import inspect

        from lmdiff import _prompting

        src = inspect.getsource(_prompting)
        for banned in ("import torch", "from lmdiff._engine", "from lmdiff.engine"):
            assert banned not in src


# ── the port proper ──────────────────────────────────────────────────


class TestTaskSpeaksTheProtocol:
    def test_generate_is_called_once_per_probe(self):
        """The loop moved out of the engine. Both implementations were
        always per-prompt at batch size 1, so this costs nothing."""
        eng = _Recorder(["42", "56"])
        tr = Task("t", PROBES, ExactMatch()).run(eng)
        assert [c["prompt"] for c in eng.calls] == ["17 + 25 = ", "7 * 8 = "]
        assert tr.accuracy == 1.0

    def test_result_uses_protocol_name_not_model_name(self):
        eng = _Recorder(["42", "56"])
        assert Task("t", PROBES, ExactMatch()).run(eng).engine_name == "recorder"

    def test_outputs_skips_generation_entirely(self):
        """The pairing hook (L-010). The family pipeline scores accuracy
        on the very completions δ was computed from, so no second
        generation happens — which is also why δ cannot move."""
        eng = _Recorder()
        tr = Task("t", PROBES, ExactMatch()).run(eng, outputs=["42", "56"])
        assert eng.calls == []
        assert tr.accuracy == 1.0

    def test_misaligned_outputs_raise(self):
        with pytest.raises(ValueError, match="align by index"):
            Task("t", PROBES, ExactMatch()).run(_Recorder(), outputs=["42"])

    def test_loglikelihood_scores_one_choice_per_call(self):
        from lmdiff.tasks.loglikelihood import loglikelihood_accuracy

        probes = ProbeSet([
            Probe(id="p", text="Q?", metadata={
                "choices": ["A", "B", "C"], "correct_index": 0,
            }),
        ])
        eng = _Recorder()
        loglikelihood_accuracy(probes, eng, task_name="t")
        assert [c["continuation"] for c in eng.calls] == ["A", "B", "C"]


class TestRadarSplit:
    def test_run_single_is_exported(self):
        """It was unexported, which is why nobody reported that it had
        stopped working — an unexported thing has no users to report it."""
        import lmdiff

        assert hasattr(lmdiff, "CapabilityRadar")
        assert hasattr(lmdiff, "RadarResult")

    def test_run_single_threads_config_through(self):
        from lmdiff.tasks.capability_radar import CapabilityRadar
        from lmdiff.tasks.evaluators import ContainsAnswer

        probes = ProbeSet([
            Probe(id="m", text="1+1=", domain="math", expected="2"),
            Probe(id="k", text="FR=", domain="knowledge", expected="Paris"),
        ])
        eng = _Recorder(["2", "Paris"])
        cfg = Config(model="gpt2", system_prompt="Be brief.")
        CapabilityRadar(probes, evaluator=ContainsAnswer()).run_single(
            eng,
            prefix_text=prefix_text(cfg),
            generate_kwargs=generate_kwargs(cfg, max_new_tokens=8),
        )
        assert eng.calls, "run_single generated nothing"
        for call in eng.calls:
            assert call["prefix_text"] == "Be brief.\n"

    def test_run_pair_warns_that_it_is_going(self):
        """Deprecated in v0.4.5, removed in v0.5.0. Nothing in
        lmdiff/tasks/ carried a DeprecationWarning before this."""
        import inspect

        from lmdiff.tasks.capability_radar import CapabilityRadar

        src = inspect.getsource(CapabilityRadar.run_pair)
        assert "DeprecationWarning" in src
        assert "v0.5.0" in src


class TestGenerativeCellsDerived:
    def test_derived_from_output_type_not_a_hardcoded_list(self):
        from lmdiff._findings import _generative_cells
        from lmdiff.geometry import GeoResult

        r = GeoResult(
            base_name="b", variant_names=["v"], n_probes=3,
            magnitudes={"v": 1.0}, cosine_matrix={"v": {"v": 1.0}},
            change_vectors={"v": [0.1, 0.2, 0.3]}, per_probe={"v": {}},
            probe_domains=("math", "math", "commonsense"),
            probe_output_types=("generate_until", "generate_until",
                                "multiple_choice"),
        )
        assert _generative_cells(r) == {"math"}

    def test_pre_v8_result_yields_nothing_rather_than_guessing(self):
        from lmdiff._findings import _generative_cells
        from lmdiff.geometry import GeoResult

        r = GeoResult(
            base_name="b", variant_names=["v"], n_probes=1,
            magnitudes={"v": 1.0}, cosine_matrix={"v": {"v": 1.0}},
            change_vectors={"v": [0.1]}, per_probe={"v": {}},
            probe_domains=("math",),
        )
        assert _generative_cells(r) == set()


# ── accuracy restored to the live path ───────────────────────────────


class TestLiveAccuracy:
    """`accuracy_by_variant` has been `{}` since v0.4.1. v0.4.5 fills it
    in — from the completions δ was already computed from, so nothing
    that already existed can move."""

    def _run(self, **kw):
        from tests.fixtures.mock_engine import MockEngine

        from lmdiff._pipeline import run_family_pipeline

        base_cfg = Config(model="mock_base")
        v_cfg = Config(model="mock_variant")
        return run_family_pipeline(
            base_engine=MockEngine(config=base_cfg, seed=1),
            base_config=base_cfg,
            variant_engines={"v": MockEngine(config=v_cfg, seed=2)},
            variant_configs={"v": v_cfg},
            probe_set=ProbeSet([
                Probe(id=f"p{i}", text=f"probe {i} ", domain="math",
                      expected="x", output_type="generate_until")
                for i in range(4)
            ]),
            max_new_tokens=4,
            **kw,
        )

    def test_accuracy_is_populated(self):
        meta = self._run().metadata
        assert "accuracy_by_variant" in meta, (
            "accuracy has been missing from the live path since v0.4.1"
        )
        assert set(meta["accuracy_by_variant"]) == {"v"}

    def test_keyed_by_domain_not_by_task(self):
        acc = self._run().metadata["accuracy_by_variant"]["v"]
        assert set(acc) == {"math"}

    def test_reused_outputs_produce_a_real_number(self):
        """Gate the claim on the quantity, not on the key (L-039).

        A mutation that loses the delta completions leaves the key in
        place with a `None` value -- every probe drops out of the
        generative branch, `n_scorable` is 0, and the domain reports
        "nothing could be scored". `set(acc) == {"math"}` still holds.
        So assert the number.
        """
        from lmdiff._validity import cell_accuracy

        acc = self._run().metadata["accuracy_by_variant"]["v"]
        assert isinstance(cell_accuracy(acc["math"]), float), (
            "accuracy is None -- the completions delta was computed from "
            "did not reach the evaluator"
        )
        assert 0.0 <= cell_accuracy(acc["math"]) <= 1.0
        # The support travels with the value (v0.4.5).
        assert acc["math"]["n_scorable"] > 0
        assert acc["math"]["n_probes"] == 4

    def test_base_accuracy_is_a_real_number_too(self):
        from lmdiff._validity import cell_accuracy

        base = self._run().metadata["base_accuracy"]
        assert isinstance(cell_accuracy(base["math"]), float)
        assert 0.0 <= cell_accuracy(base["math"]) <= 1.0

    def test_base_is_scored_too(self):
        """Without this, `BaseAccuracyMissingFinding` fires on every run
        forever: nothing else in lmdiff has ever written `base_accuracy`,
        and the finding's condition is "variants have it, base does
        not"."""
        meta = self._run().metadata
        assert "base_accuracy" in meta
        assert set(meta["base_accuracy"]) == {"math"}

    def test_base_accuracy_silences_the_permanent_caveat(self):
        from lmdiff._findings import BaseAccuracyMissingFinding, extract_findings

        findings = extract_findings(self._run())
        assert not [
            f for f in findings if isinstance(f, BaseAccuracyMissingFinding)
        ], "the caveat must fire only when base scoring actually failed"

    def test_geometry_is_untouched_by_accuracy(self):
        """The guard on this whole change: accuracy fills in a missing
        number, it does not move an existing one. Variants are scored on
        the δ generations rather than new ones, so no extra RNG is
        consumed; the base pass runs after every variant, so it cannot
        perturb what they saw."""
        a, b = self._run(), self._run()
        assert a.change_vectors == b.change_vectors
        assert a.magnitudes == b.magnitudes
        assert a.share_per_domain == b.share_per_domain
        assert a.magnitudes_per_domain_normalized == \
            b.magnitudes_per_domain_normalized

    def test_multiple_choice_is_scored_by_loglikelihood_not_string_match(self):
        """Three of the five calibration tasks are multiple-choice, and
        `from_lm_eval` gives those probes `scoring=None`. Evaluating
        their *generated text* against the gold choice is not a low
        accuracy — it is a different measurement. So the pipeline
        dispatches on `output_type`: MC probes are scored over their
        stored choices, everything else on the completion.
        """
        from tests.fixtures.mock_engine import MockEngine

        from lmdiff._pipeline import run_family_pipeline

        base_cfg = Config(model="mock_base")
        v_cfg = Config(model="mock_variant")
        probes = ProbeSet([
            Probe(
                id=f"mc{i}", text=f"q{i} ", domain="commonsense",
                output_type="multiple_choice",
                metadata={"choices": ["A", "B"], "correct_index": 0},
            )
            for i in range(4)
        ])
        result = run_family_pipeline(
            base_engine=MockEngine(config=base_cfg, seed=1),
            base_config=base_cfg,
            variant_engines={"v": MockEngine(config=v_cfg, seed=2)},
            variant_configs={"v": v_cfg},
            probe_set=probes,
            max_new_tokens=4,
        )
        from lmdiff._validity import cell_accuracy

        acc = cell_accuracy(
            result.metadata["accuracy_by_variant"]["v"]["commonsense"]
        )
        # Not None: string-matching a completion against "A"/"B" would
        # score 0.0 here, and scoring the choices gives a real number.
        assert acc is not None
        assert 0.0 <= acc <= 1.0


# ── the validity floor on accuracy (v0.4.5) ──────────────────────────


class TestAccuracyFloor:
    """`min_valid_fraction` applied to accuracy — the same threshold, not
    a second one. See docs/methodology/normalization.md.

    Accuracy is a single-engine quantity, so the floor tests *that
    engine's* valid fraction, where share tests valid-for-both. On the
    Llama-2 calibration that reproduces `domain_status`'s `out_of_range`
    determination exactly and extends it to base.
    """

    def _cell(self, n_valid, n_probes, n_correct=None, floor=0.5):
        from lmdiff._validity import accuracy_cell

        n_correct = n_valid // 2 if n_correct is None else n_correct
        return accuracy_cell(
            n_correct=n_correct, n_scorable=n_valid,
            n_valid=n_valid, n_probes=n_probes, min_valid_fraction=floor,
        )

    def test_the_nine_of_a_hundred_survivors_are_suppressed(self):
        """The calibration's real case: 9 of 100 long-context probes fit
        the 4096-token window, and those 9 are the short left tail."""
        from lmdiff._validity import ACCURACY_BELOW_FLOOR, cell_accuracy

        cell = self._cell(9, 100)
        assert cell_accuracy(cell) is None
        assert cell["reason"] == ACCURACY_BELOW_FLOOR

    def test_eighty_nine_of_a_hundred_survives(self):
        """CodeLlama's 16k window covers 89. That is a measurement, and
        suppressing it would discard real signal."""
        from lmdiff._validity import cell_accuracy

        cell = self._cell(89, 100, n_correct=13)
        assert cell_accuracy(cell) is not None
        assert cell["reason"] is None

    @pytest.mark.parametrize("n_valid,kept", [
        (49, False), (50, True), (51, True),
    ])
    def test_boundary_is_at_the_floor_exactly(self, n_valid, kept):
        from lmdiff._validity import cell_accuracy

        assert (cell_accuracy(self._cell(n_valid, 100)) is not None) is kept

    def test_floor_comes_from_the_run_not_a_constant(self):
        """`min_valid_fraction=0.0` disables it, exactly as it does for
        share — the escape hatch means the same thing in both places."""
        from lmdiff._validity import cell_accuracy

        assert cell_accuracy(self._cell(9, 100, floor=0.0)) is not None
        assert cell_accuracy(self._cell(9, 100, floor=0.95)) is None

    def test_support_shown_is_scorable_over_domain_size(self):
        """Not scorable-over-attempted. `(89/89)` is true of the probes
        CodeLlama attempted and hides the 11 it could not — a
        denominator that is not the one the claim rests on (L-039)."""
        from lmdiff._validity import cell_support

        assert cell_support(self._cell(89, 100, n_correct=13)) == (89, 100)

    def test_n_valid_kept_as_the_floors_audit_trail(self):
        cell = self._cell(9, 100)
        assert cell["n_valid"] == 9 and cell["n_probes"] == 100

    def test_suppressed_is_distinguishable_from_unjudgeable(self):
        """Two different reasons for `None`, and a reader must be able to
        tell them apart without re-deriving either."""
        from lmdiff._validity import ACCURACY_BELOW_FLOOR, accuracy_cell

        floored = self._cell(9, 100)
        nothing = accuracy_cell(
            n_correct=0, n_scorable=0, n_valid=100, n_probes=100,
        )
        assert floored["reason"] == ACCURACY_BELOW_FLOOR
        assert nothing["reason"] is None
        assert nothing["accuracy"] is None

    def test_legacy_bare_float_still_reads(self):
        """`examples/*.json` came from the deprecated path and carry a
        bare float. A renderer must not raise on one (the v0.4.2
        `to_html` lesson)."""
        from lmdiff._validity import cell_accuracy, cell_support

        assert cell_accuracy(0.53) == 0.53
        assert cell_support(0.53) is None
        assert cell_accuracy(float("nan")) is None
        assert cell_accuracy(None) is None

    def test_renderers_show_the_denominator(self):
        """Rendered and read, not asserted on a computed value (L-038)."""
        from lmdiff.report.markdown import _build_accuracy_table

        acc = {"v": {
            "long-context": self._cell(89, 100, n_correct=13),
            "math": self._cell(9, 100),
        }}
        text = "\n".join(_build_accuracy_table(["v"], acc, set()))
        assert "(89/100)" in text, "accuracy rendered without its denominator"
        assert "n/a" in text, "suppressed cell rendered as a number"

    def test_base_block_of_all_suppressed_cells_is_not_base_accuracy(self):
        """A base block whose every cell is floored is the absence of
        base accuracy wearing the right shape — the caveat must fire."""
        from lmdiff._findings import BaseAccuracyMissingFinding, extract_findings
        from lmdiff.geometry import GeoResult

        r = GeoResult(
            base_name="b", variant_names=["v"], n_probes=2,
            magnitudes={"v": 1.0}, cosine_matrix={"v": {"v": 1.0}},
            change_vectors={"v": [0.1, 0.2]}, per_probe={"v": {}},
            metadata={
                "accuracy_by_variant": {"v": {"math": self._cell(100, 100)}},
                "base_accuracy": {"math": self._cell(9, 100)},
            },
        )
        assert [
            f for f in extract_findings(r)
            if isinstance(f, BaseAccuracyMissingFinding)
        ], "base scoring produced nothing usable and nothing said so"

    def _floor_run(self, min_valid_fraction):
        """A domain the variant can only partly attempt: 2 of 10 probes
        fit its window. `MockEngine.token_count` is word-split, so probe
        length is controllable exactly."""
        from tests.fixtures.mock_engine import MockEngine

        from lmdiff._pipeline import run_family_pipeline

        base_cfg = Config(model="mock_base")
        v_cfg = Config(model="mock_variant")
        probes = ProbeSet(
            [
                Probe(id=f"s{i}", text="short probe here ", domain="thin",
                      expected="x", output_type="generate_until")
                for i in range(2)
            ] + [
                Probe(id=f"l{i}", text="word " * 200, domain="thin",
                      expected="x", output_type="generate_until")
                for i in range(8)
            ] + [
                Probe(id=f"w{i}", text="short probe here ", domain="wide",
                      expected="x", output_type="generate_until")
                for i in range(10)
            ]
        )
        kw = {} if min_valid_fraction is None else {
            "min_valid_fraction": min_valid_fraction,
        }
        return run_family_pipeline(
            base_engine=MockEngine(config=base_cfg, seed=1, max_context=64),
            base_config=base_cfg,
            variant_engines={"v": MockEngine(config=v_cfg, seed=2,
                                             max_context=64)},
            variant_configs={"v": v_cfg},
            probe_set=probes,
            max_new_tokens=4,
            **kw,
        )

    def test_pipeline_threads_the_runs_floor_to_accuracy(self):
        """End-to-end wiring, not the predicate.

        `accuracy_cell` is unit-tested above with synthetic counts, which
        proves the rule and says nothing about whether the pipeline hands
        it the run's `min_valid_fraction`. A pipeline that always passed
        0.0 would pass every one of those tests. So: the same run at two
        floors must disagree.
        """
        from lmdiff._validity import ACCURACY_BELOW_FLOOR, cell_accuracy

        floored = self._floor_run(0.5).metadata["accuracy_by_variant"]["v"]
        disabled = self._floor_run(0.0).metadata["accuracy_by_variant"]["v"]

        assert cell_accuracy(floored["thin"]) is None, (
            "2 of 10 probes attempted and the cell still reports — the "
            "run's min_valid_fraction did not reach the accuracy path"
        )
        assert floored["thin"]["reason"] == ACCURACY_BELOW_FLOOR
        assert cell_accuracy(disabled["thin"]) is not None, (
            "min_valid_fraction=0.0 must restore the pre-floor behaviour, "
            "exactly as it does for share"
        )
        # The floor is not a blanket suppression: the fully-attempted
        # domain reports either way.
        assert cell_accuracy(floored["wide"]) is not None
        assert cell_accuracy(disabled["wide"]) is not None

    def test_floor_applies_to_base_too(self):
        """Base has no `domain_status` row, so nothing else would have
        gated it."""
        from lmdiff._validity import cell_accuracy

        base = self._floor_run(0.5).metadata["base_accuracy"]
        assert cell_accuracy(base["thin"]) is None
        assert cell_accuracy(base["wide"]) is not None
