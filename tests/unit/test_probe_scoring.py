"""Per-probe scoring labels and per-probe evaluator selection. v0.4.4.

Commit 4.3. PHASE_PLAN §5.1 originally specified a capability-valued
``task_type``; the v0.4.4 investigation replaced it with the two fields
here. Background: ``docs/internal/v044_taxonomy_notes.md``.

The defect these protect is not "accuracy is wrong". It is **accuracy
displayed without the condition that decides whether it means
anything** — ``CapabilityRadar`` applied one caller-chosen evaluator to
every probe, and ``MultipleChoice`` on ``v01`` rendered a clean ``0.0``
across all 90 while each carried ``reason: "missing_mc_metadata"``.
That is the L-039 shape, so the assertions below aim at the denominator
and the label, not only at the quotient.
"""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from lmdiff.probes.loader import (  # noqa: E402
    KNOWN_OUTPUT_TYPES,
    Probe,
    ProbeSet,
)
from lmdiff.tasks.base import (  # noqa: E402
    UNSCORABLE_REASONS,
    Task,
)
from lmdiff.tasks.evaluators import ContainsAnswer, ExactMatch, MultipleChoice  # noqa: E402
from lmdiff.tasks.registry import (  # noqa: E402
    EVALUATOR_REGISTRY,
    KNOWN_SCORINGS,
    get_evaluator,
)

V01 = pathlib.Path(__file__).resolve().parents[2] / "lmdiff" / "probes" / "v01.json"


class _Gen:
    """v0.2.x GenerationResult shape: completions[probe][sample]."""

    def __init__(self, texts: list[str]) -> None:
        self.completions = [[t] for t in texts]


class _Engine:
    """Minimal stand-in for the engine surface ``Task.run`` uses.

    Deliberately the *legacy* surface. Porting ``Task`` to the ``Engine``
    Protocol is tracked as a v0.5.0 blocker (PHASE_PLAN Z.4 item 6) and
    is explicitly not commit 4.3's job — these tests pin scoring
    behaviour, not engine compatibility.
    """

    model_name = "stub"

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = outputs

    def generate(self, prompts, n_samples=1, max_new_tokens=64, **kw):
        return _Gen(self._outputs)


# ── the registry ─────────────────────────────────────────────────────


class TestEvaluatorRegistry:
    def test_registry_covers_every_evaluator(self):
        """Was two partial dicts in two modules: cli.py listed three of
        five, experiments/family.py mapped task names onto two."""
        assert set(EVALUATOR_REGISTRY) == {
            "exact_match", "contains_answer", "multiple_choice",
            "f1", "gsm8k_number_match",
        }

    def test_keys_are_the_classes_own_names(self):
        """Derived, not written out — so the registry cannot drift from
        the classes it indexes."""
        for key, cls in EVALUATOR_REGISTRY.items():
            assert cls.name == key

    def test_get_evaluator_returns_an_instance(self):
        ev = get_evaluator("f1")
        assert ev is not None and ev.name == "f1"

    @pytest.mark.parametrize("bad", [None, "", "no_such_evaluator"])
    def test_get_evaluator_is_total(self, bad):
        """Returns None rather than raising: callers fall back, which is
        what keeps an unrecognised `scoring` degrading instead of
        failing a run."""
        assert get_evaluator(bad) is None

    def test_cli_and_family_no_longer_carry_their_own_maps(self):
        """The consolidation, pinned. Either module growing a private
        evaluator table again is the regression."""
        import lmdiff.cli as cli
        from lmdiff.experiments import family

        assert not hasattr(cli, "_EVALUATOR_MAP")
        assert family.GENERATE_EVALUATORS["gsm8k"] is EVALUATOR_REGISTRY[
            "gsm8k_number_match"
        ]
        assert family.GENERATE_EVALUATORS["triviaqa"] is EVALUATOR_REGISTRY["f1"]


# ── the fields ───────────────────────────────────────────────────────


class TestProbeFields:
    def test_defaults_are_none_not_a_value(self):
        """`None` means unlabelled. Defaulting to a value would assert
        something nobody stated — the L-039 shape at field level."""
        p = Probe(id="p", text="t")
        assert p.output_type is None
        assert p.scoring is None

    def test_probeset_accessors_drop_none(self):
        ps = ProbeSet([
            Probe(id="a", text="x", output_type="generate_until", scoring="f1"),
            Probe(id="b", text="y"),
        ])
        assert ps.output_types == ["generate_until"]
        assert ps.scorings == ["f1"]

    def test_filter_by_new_fields(self):
        ps = ProbeSet([
            Probe(id="a", text="x", output_type="multiple_choice"),
            Probe(id="b", text="y", output_type="generate_until", scoring="f1"),
        ])
        assert ps.filter(output_type="generate_until").ids == ["b"]
        assert ps.filter(scoring="f1").ids == ["b"]

    def test_by_output_type_buckets_none_as_unknown(self):
        """Mirrors by_domain's convention rather than inventing a second."""
        ps = ProbeSet([
            Probe(id="a", text="x", output_type="generate_until"),
            Probe(id="b", text="y"),
        ])
        assert {k: len(v) for k, v in ps.by_output_type().items()} == {
            "generate_until": 1, "unknown": 1,
        }

    def test_round_trip_preserves_both_fields(self, tmp_path):
        ps = ProbeSet([
            Probe(id="a", text="x", domain="math",
                  output_type="generate_until", scoring="exact_match",
                  expected="1"),
        ], name="t", version="1")
        out = tmp_path / "p.json"
        ps.to_json(out)
        back = ProbeSet.from_json(out)
        assert back[0].output_type == "generate_until"
        assert back[0].scoring == "exact_match"

    def test_unset_fields_are_not_written(self, tmp_path):
        """Keeps the labels out of every probe file that does not use
        them, matching how `domain` has always been written."""
        out = tmp_path / "p.json"
        ProbeSet([Probe(id="a", text="x")]).to_json(out)
        raw = json.loads(out.read_text(encoding="utf-8"))
        assert "output_type" not in raw["probes"][0]
        assert "scoring" not in raw["probes"][0]


class TestVocabularyWarnings:
    def _write(self, tmp_path, **extra):
        out = tmp_path / "p.json"
        out.write_text(json.dumps({
            "name": "t", "version": "1",
            "probes": [{"id": "a", "text": "x", **extra}],
        }), encoding="utf-8")
        return out

    def test_unknown_output_type_warns_and_keeps_the_value(self, tmp_path):
        path = self._write(tmp_path, output_type="telepathy")
        with pytest.warns(UserWarning, match="output_type='telepathy'"):
            ps = ProbeSet.from_json(path)
        assert ps[0].output_type == "telepathy", "warn, do not silently drop"

    def test_unknown_scoring_warns_and_keeps_the_value(self, tmp_path):
        path = self._write(tmp_path, scoring="vibes")
        with pytest.warns(UserWarning, match="scoring='vibes'"):
            ps = ProbeSet.from_json(path)
        assert ps[0].scoring == "vibes"

    def test_known_values_are_silent(self, tmp_path):
        path = self._write(tmp_path, output_type="generate_until", scoring="f1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ProbeSet.from_json(path)

    def test_output_type_vocabulary_is_defined_once(self):
        """`adapters` used to carry its own copy of the same four
        values (L-035)."""
        from lmdiff.probes import adapters

        assert adapters._SUPPORTED_OUTPUT_TYPES is KNOWN_OUTPUT_TYPES

    def test_scoring_vocabulary_is_the_registry(self):
        assert KNOWN_SCORINGS == frozenset(EVALUATOR_REGISTRY)


# ── per-probe evaluator selection ────────────────────────────────────


class TestPerProbeEvaluator:
    def test_each_probe_gets_the_evaluator_it_names(self):
        """The commit's reason for existing. One ProbeSet, two formats,
        both scored correctly — impossible under a single evaluator."""
        probes = ProbeSet([
            # Exact-match probe: the output is precisely the answer.
            Probe(id="e", text="q", expected="42", scoring="exact_match"),
            # Containment probe: the answer is embedded in prose.
            Probe(id="c", text="q", expected="Paris", scoring="contains_answer"),
        ])
        tr = Task("t", probes, ExactMatch()).run(
            _Engine(["42", "The capital is Paris, in France."]),
        )
        assert [r.evaluator for r in tr.per_probe] == [
            "exact_match", "contains_answer",
        ]
        assert tr.n_correct == 2, (
            "ExactMatch alone marks the prose answer wrong; "
            "ContainsAnswer alone would accept a wrong exact answer"
        )

    def test_constructor_evaluator_is_the_fallback(self):
        probes = ProbeSet([Probe(id="a", text="q", expected="Paris")])
        tr = Task("t", probes, ContainsAnswer()).run(
            _Engine(["The capital is Paris."]),
        )
        assert tr.per_probe[0].evaluator == "contains_answer"
        assert tr.accuracy == 1.0

    def test_unknown_scoring_falls_back_rather_than_raising(self):
        probes = ProbeSet([
            Probe(id="a", text="q", expected="Paris", scoring="from_the_future"),
        ])
        tr = Task("t", probes, ContainsAnswer()).run(_Engine(["... Paris ..."]))
        assert tr.per_probe[0].evaluator == "contains_answer"
        assert tr.accuracy == 1.0


# ── the unscorable count ─────────────────────────────────────────────


class TestUnscorable:
    def _mc_on_unlabelled(self):
        """The v01 case: MultipleChoice against probes with no metadata."""
        probes = ProbeSet([
            Probe(id="m1", text="q", domain="math", expected="42"),
            Probe(id="m2", text="q", domain="math", expected="7"),
        ])
        return Task("t", probes, MultipleChoice()).run(_Engine(["42", "7"]))

    def test_accuracy_is_none_not_zero_when_nothing_is_scorable(self):
        """The headline. 0.0 says "the model got everything wrong";
        None says "this evaluator cannot judge these probes". They are
        different claims and only one is true here."""
        tr = self._mc_on_unlabelled()
        assert tr.accuracy is None
        assert tr.n_unscorable == 2
        assert tr.n_scorable == 0

    def test_per_domain_accuracy_is_none_too(self):
        tr = self._mc_on_unlabelled()
        assert tr.per_domain["math"]["accuracy"] is None
        assert tr.per_domain["math"]["n_unscorable"] == 2
        assert tr.per_domain["math"]["n_scorable"] == 0

    def test_unscorable_probes_leave_the_denominator(self):
        """Not "counted wrong". A probe the rule cannot judge carries no
        information about the model, so including it in the denominator
        makes the model look worse for a defect in the probe set —
        "excluded cells are removed, not nulled", one layer down."""
        probes = ProbeSet([
            Probe(id="ok", text="q", expected="42"),
            Probe(id="bad", text="q", expected=None),  # no gold to compare
        ])
        tr = Task("t", probes, ExactMatch()).run(_Engine(["42", "anything"]))
        assert tr.n_probes == 2
        assert tr.n_unscorable == 1
        assert tr.accuracy == 1.0, "1 correct of 1 scorable, not 1 of 2"

    def test_a_bad_output_is_scorable_and_wrong(self):
        """The distinction the frozenset encodes. `extraction_failed`
        and `no_choice_parsed` are facts about the model, so they count
        against it; only probe-side defects are unscorable."""
        from lmdiff.tasks.evaluators import Gsm8kNumberMatch

        probes = ProbeSet([Probe(id="g", text="q", expected="#### 42")])
        tr = Task("t", probes, Gsm8kNumberMatch()).run(
            _Engine(["no numbers here at all"]),
        )
        assert tr.per_probe[0].metadata["reason"] == "extraction_failed"
        assert tr.per_probe[0].scorable is True
        assert tr.n_unscorable == 0
        assert tr.accuracy == 0.0

    def test_empty_output_is_scorable_and_wrong(self):
        probes = ProbeSet([Probe(id="a", text="q", expected="42")])
        tr = Task("t", probes, ExactMatch()).run(_Engine(["   "]))
        assert tr.per_probe[0].scorable is True
        assert tr.accuracy == 0.0

    def test_unscorable_vocabulary_is_shared_not_restated(self):
        """Evaluators import the constants, so a typo is a NameError
        rather than a silently unmatched string."""
        import inspect

        from lmdiff.tasks import evaluators

        src = inspect.getsource(evaluators)
        for literal in ('"no_expected"', '"empty_expected"',
                        '"missing_mc_metadata"'):
            assert literal not in src, (
                f"{literal} is written out again in evaluators.py; use the "
                f"constant from tasks.base"
            )
        assert UNSCORABLE_REASONS == frozenset({
            "no_expected", "empty_expected", "missing_mc_metadata",
        })


class TestRadarSurfacesTheDenominator:
    def _radar_row(self, n_unscorable: int):
        from lmdiff.tasks.capability_radar import DomainRadarResult, RadarResult

        dr = DomainRadarResult(
            domain="math", n_probes=30,
            accuracy=None if n_unscorable == 30 else 0.5,
            n_unscorable=n_unscorable,
        )
        return RadarResult(
            engine_a_name="a", engine_b_name=None, domains=["math"],
            a_by_domain={"math": dr}, b_by_domain=None, bd_by_domain=None,
            bd_healthy_by_domain=None, degeneracy_rates=None,
        )

    def test_summary_table_carries_the_denominator(self):
        """No consumer can read accuracy_a without n_scorable_a being
        in the same row."""
        row = self._radar_row(30).summary_table()[0]
        assert row["accuracy_a"] is None
        assert row["n_scorable_a"] == 0
        assert row["n_unscorable_a"] == 30
        assert row["n_probes"] == 30

    def test_json_carries_the_denominator(self):
        from lmdiff.report.json_report import to_json_dict

        d = to_json_dict(self._radar_row(30).a_by_domain["math"])
        assert d["accuracy"] is None
        assert d["n_scorable"] == 0
        assert d["n_unscorable"] == 30

    def test_terminal_prints_the_denominator_beside_the_number(self):
        """Rendered and read, not asserted on a computed value — three
        of the four v0.4.2 defects produced output that was wrong rather
        than absent (L-038)."""
        import io

        from rich.console import Console

        from lmdiff.report.terminal import print_radar

        buf = io.StringIO()
        print_radar(self._radar_row(12), console=Console(file=buf, width=120))
        out = buf.getvalue()
        assert "18/30" in out, "accuracy must carry its denominator"
        assert "could not be judged" in out

    def test_terminal_does_not_crash_on_none_accuracy(self):
        """The v0.4.2 to_html lesson: a None that renders is worth more
        than a None that raises."""
        import io

        from rich.console import Console

        from lmdiff.report.terminal import print_radar

        buf = io.StringIO()
        print_radar(self._radar_row(30), console=Console(file=buf, width=120))
        assert "n/a" in buf.getvalue()


# ── the bundled set ──────────────────────────────────────────────────


class TestV01Labelled:
    def test_all_ninety_probes_declare_an_output_type(self):
        ps = ProbeSet.from_json(V01)
        assert len(ps) == 90
        assert ps.output_types == ["generate_until"]
        assert all(p.output_type == "generate_until" for p in ps)

    def test_three_domains_thirty_each(self):
        ps = ProbeSet.from_json(V01)
        assert {k: len(v) for k, v in ps.by_domain().items()} == {
            "code": 30, "knowledge": 30, "math": 30,
        }

    def test_scoring_is_deliberately_unset(self):
        """Not an oversight. All 90 are prefix completions -- the answer
        is the immediate continuation -- and none of the five evaluators
        implements that rule. `contains_answer` measurably misfires on
        the short ones (`"i"`, `"n"` match almost any output) and
        `exact_match` rejects every continuation that runs on. Labelling
        them with a rule known to be wrong would be worse than leaving
        the caller's fallback in place. See the [QUESTION] in
        docs/internal/v044_taxonomy_notes.md."""
        ps = ProbeSet.from_json(V01)
        assert ps.scorings == []

    def test_loads_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ProbeSet.from_json(V01)
