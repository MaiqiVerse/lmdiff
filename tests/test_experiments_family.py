"""Tests for lmdiff.experiments.family."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lmdiff.experiments.family import (
    DEFAULT_TASKS,
    FamilyExperimentResult,
    GENERATE_EVALUATORS,
    _accuracy_for_task,
    _compute_normalized_delta_by_task,
    _l2_norm,
    _partition_delta_by_task,
    plot_family_geometry,
    run_family_experiment,
)
from lmdiff.geometry import GeoResult
from lmdiff.probes.loader import Probe, ProbeSet


# ── Helpers ────────────────────────────────────────────────────────────


_probe_counter = [0]


def _probe(text: str, expected: str = "") -> Probe:
    _probe_counter[0] += 1
    return Probe(id=f"p{_probe_counter[0]}", text=text, expected=expected or None)


def _make_geo(
    *,
    variants: tuple[str, ...] = ("v1", "v2"),
    probe_texts: tuple[str, ...] = ("p0", "p1", "p2"),
    domains: tuple[str | None, ...] | None = None,
    avg_tokens: tuple[float, ...] = (),
) -> GeoResult:
    """Build a minimal GeoResult that's plausible for partition + normalize tests."""
    n = len(probe_texts)
    per_probe = {
        v: {probe_texts[i]: float((i + 1) * (idx + 1))
            for i in range(n)}
        for idx, v in enumerate(variants)
    }
    change_vectors = {v: [per_probe[v][t] for t in probe_texts] for v in variants}
    magnitudes = {
        v: float(sum(x * x for x in change_vectors[v]) ** 0.5)
        for v in variants
    }
    return GeoResult(
        base_name="base-mock",
        variant_names=list(variants),
        n_probes=n,
        magnitudes=magnitudes,
        cosine_matrix={
            v1: {v2: 1.0 if v1 == v2 else 0.5 for v2 in variants}
            for v1 in variants
        },
        change_vectors=change_vectors,
        per_probe=per_probe,
        probe_domains=domains or (),
        avg_tokens_per_probe=avg_tokens,
    )


# ── Pure-helper tests ─────────────────────────────────────────────────


class TestPartitionDeltaByTask:
    def test_groups_per_task(self):
        per_probe = {
            "v1": {"p_a": 1.0, "p_b": 2.0, "p_c": 3.0},
            "v2": {"p_a": -1.0, "p_b": -2.0, "p_c": -3.0},
        }
        per_task_probes = {
            "task_x": [_probe("p_a"), _probe("p_b")],
            "task_y": [_probe("p_c")],
        }
        out = _partition_delta_by_task(per_probe, per_task_probes)
        assert set(out.keys()) == {"v1", "v2"}
        assert out["v1"]["task_x"] == [1.0, 2.0]
        assert out["v1"]["task_y"] == [3.0]
        assert out["v2"]["task_x"] == [-1.0, -2.0]
        assert out["v2"]["task_y"] == [-3.0]

    def test_unknown_text_is_dropped(self):
        # Probe text not present in any task list should silently drop.
        per_probe = {"v1": {"p_a": 1.0, "stranger": 9.9}}
        per_task_probes = {"t": [_probe("p_a")]}
        out = _partition_delta_by_task(per_probe, per_task_probes)
        assert out["v1"]["t"] == [1.0]


class TestL2Norm:
    def test_pythagorean(self):
        assert _l2_norm([3.0, 4.0]) == pytest.approx(5.0)

    def test_empty_returns_zero(self):
        assert _l2_norm([]) == 0.0


class TestComputeNormalizedDeltaByTask:
    def test_returns_empty_without_token_data(self):
        geo = _make_geo(domains=("a", "b", "a"), avg_tokens=())
        assert _compute_normalized_delta_by_task(geo, ["a", "b"]) == {}

    def test_returns_empty_without_probe_domains(self):
        geo = _make_geo(domains=(), avg_tokens=(4.0, 4.0, 4.0))
        assert _compute_normalized_delta_by_task(geo, ["a", "b"]) == {}

    def test_populates_per_variant_per_task(self):
        geo = _make_geo(
            domains=("a", "b", "a"),
            avg_tokens=(4.0, 4.0, 4.0),
        )
        out = _compute_normalized_delta_by_task(geo, ["a", "b"])
        assert set(out.keys()) == {"v1", "v2"}
        assert set(out["v1"].keys()) == {"a", "b"}
        # All values are floats and finite (no NaN injection without bad input).
        for v, per_task in out.items():
            for task, val in per_task.items():
                assert isinstance(val, float)


# ── DEFAULT_TASKS shape ───────────────────────────────────────────────


def test_default_tasks_is_immutable_tuple():
    assert isinstance(DEFAULT_TASKS, tuple)
    assert "hellaswag" in DEFAULT_TASKS
    assert "longbench_2wikimqa" in DEFAULT_TASKS


def test_generate_evaluators_covers_known_generate_tasks():
    # Sanity: gsm8k uses Gsm8kNumberMatch, longbench tasks use F1.
    assert GENERATE_EVALUATORS["gsm8k"].__name__ == "Gsm8kNumberMatch"
    assert GENERATE_EVALUATORS["longbench_2wikimqa"].__name__ == "F1"


# ── _accuracy_for_task dispatch ───────────────────────────────────────


class TestAccuracyForTask:
    def test_unknown_task_uses_contains_answer(self):
        engine = MagicMock()
        ps = ProbeSet([_probe("hi", "hello")], name="custom", version="x")

        with patch("lmdiff.experiments.family.Task") as MockTask:
            mock_task = MockTask.return_value
            mock_task.run.return_value = MagicMock(accuracy=0.42)
            acc = _accuracy_for_task("not_a_real_task", ps, engine)
        assert acc == 0.42
        # First positional arg is task name; evaluator class name should be ContainsAnswer.
        eval_arg = MockTask.call_args.args[2]
        assert type(eval_arg).__name__ == "ContainsAnswer"

    def test_multiple_choice_uses_loglikelihood(self):
        engine = MagicMock()
        ps = ProbeSet(
            [_probe("p0", "A")], name="hellaswag", version="lm-eval-harness",
        )
        with patch(
            "lmdiff.experiments.family.loglikelihood_accuracy"
        ) as mock_ll:
            mock_ll.return_value = MagicMock(accuracy=0.7)
            acc = _accuracy_for_task("hellaswag", ps, engine)
        assert acc == 0.7
        mock_ll.assert_called_once()

    def test_requires_execution_returns_nan(self):
        engine = MagicMock()
        ps = ProbeSet(
            [_probe("p", "")], name="humaneval", version="lm-eval-harness",
        )
        acc = _accuracy_for_task("humaneval", ps, engine)
        assert acc != acc  # NaN


# ── Library entry: run_family_experiment ──────────────────────────────


def _build_fake_geo(probe_texts: list[str]) -> GeoResult:
    return _make_geo(
        variants=("v1",),
        probe_texts=tuple(probe_texts),
        domains=tuple(["d"] * len(probe_texts)),
        avg_tokens=tuple([4.0] * len(probe_texts)),
    )


class TestRunFamilyExperiment:
    def test_rejects_empty_variants(self, tmp_path):
        with pytest.raises(ValueError, match="variants"):
            run_family_experiment(
                base="b",
                variants={},
                tasks=["hellaswag"],
                output_dir=tmp_path,
                progress=False,
            )

    def test_rejects_empty_tasks(self, tmp_path):
        with pytest.raises(ValueError, match="tasks"):
            run_family_experiment(
                base="b",
                variants={"v1": "m"},
                tasks=[],
                output_dir=tmp_path,
                progress=False,
            )

    def test_requires_output_dir_when_writing(self):
        with pytest.raises(ValueError, match="output_dir"):
            run_family_experiment(
                base="b",
                variants={"v1": "m"},
                tasks=["hellaswag"],
                output_dir=None,
                write_outputs=True,
                progress=False,
            )

    def test_happy_path_no_writes(self, tmp_path):
        # Mock the heavy machinery: load probes, ChangeGeometry, accuracy.
        probe_texts = ["q0", "q1"]
        per_task_probes = {"hellaswag": [_probe(t) for t in probe_texts]}
        mega = ProbeSet(
            [_probe(t) for t in probe_texts],
            name="lm_eval:hellaswag",
            version="lm-eval-harness",
        )
        fake_geo = _build_fake_geo(probe_texts)

        with patch(
            "lmdiff.experiments.family._load_concatenated_probes",
            return_value=(mega, per_task_probes),
        ), patch(
            "lmdiff.experiments.family.ChangeGeometry"
        ) as MockCG, patch(
            "lmdiff.experiments.family._accuracy_for_task",
            return_value=0.5,
        ), patch(
            "lmdiff.experiments.family.print_geometry"
        ), patch(
            "lmdiff.engine.InferenceEngine"
        ) as _MockEng:
            MockCG.return_value.analyze.return_value = fake_geo
            result = run_family_experiment(
                base="base-id",
                variants={"v1": "model-1"},
                tasks=["hellaswag"],
                limit_per_task=2,
                max_new_tokens=8,
                seed=0,
                output_dir=None,
                write_outputs=False,
                render_radars=False,
                progress=False,
            )

        assert isinstance(result, FamilyExperimentResult)
        assert result.base == "base-id"
        assert result.variants == {"v1": "model-1"}
        assert result.tasks == ["hellaswag"]
        assert result.geo is fake_geo
        assert "v1" in result.delta_magnitude_by_variant
        assert "hellaswag" in result.delta_magnitude_by_variant["v1"]
        assert result.accuracy_by_variant["v1"]["hellaswag"] == 0.5
        # write_outputs=False → no paths recorded.
        assert result.output_paths == {}

    def test_skip_accuracy_omits_phase_b(self, tmp_path):
        probe_texts = ["q0", "q1"]
        per_task_probes = {"hellaswag": [_probe(t) for t in probe_texts]}
        mega = ProbeSet(
            [_probe(t) for t in probe_texts],
            name="lm_eval:hellaswag", version="lm-eval-harness",
        )
        fake_geo = _build_fake_geo(probe_texts)

        with patch(
            "lmdiff.experiments.family._load_concatenated_probes",
            return_value=(mega, per_task_probes),
        ), patch(
            "lmdiff.experiments.family.ChangeGeometry"
        ) as MockCG, patch(
            "lmdiff.experiments.family._accuracy_for_task"
        ) as mock_acc, patch(
            "lmdiff.experiments.family.print_geometry"
        ), patch(
            "lmdiff.engine.InferenceEngine"
        ):
            MockCG.return_value.analyze.return_value = fake_geo
            result = run_family_experiment(
                base="b",
                variants={"v1": "m"},
                tasks=["hellaswag"],
                output_dir=None,
                write_outputs=False,
                render_radars=False,
                skip_accuracy=True,
                progress=False,
            )
        assert mock_acc.call_count == 0
        assert result.accuracy_by_variant == {}

    def test_writes_summary_and_georesult_json(self, tmp_path):
        probe_texts = ["q0", "q1"]
        per_task_probes = {"hellaswag": [_probe(t) for t in probe_texts]}
        mega = ProbeSet(
            [_probe(t) for t in probe_texts],
            name="lm_eval:hellaswag", version="lm-eval-harness",
        )
        fake_geo = _build_fake_geo(probe_texts)

        with patch(
            "lmdiff.experiments.family._load_concatenated_probes",
            return_value=(mega, per_task_probes),
        ), patch(
            "lmdiff.experiments.family.ChangeGeometry"
        ) as MockCG, patch(
            "lmdiff.experiments.family._accuracy_for_task",
            return_value=0.3,
        ), patch(
            "lmdiff.experiments.family.print_geometry"
        ), patch(
            "lmdiff.engine.InferenceEngine"
        ):
            MockCG.return_value.analyze.return_value = fake_geo
            result = run_family_experiment(
                base="b",
                variants={"v1": "m"},
                tasks=["hellaswag"],
                output_dir=tmp_path,
                output_prefix="exp",
                write_outputs=True,
                render_radars=False,
                progress=False,
            )

        summary_path = tmp_path / "exp.json"
        geo_path = tmp_path / "exp_georesult.json"
        assert summary_path.exists()
        assert geo_path.exists()

        summary = json.loads(summary_path.read_text())
        assert summary["base"] == "b"
        assert summary["variants"] == {"v1": "m"}
        assert summary["tasks"] == ["hellaswag"]
        assert "delta_magnitude_by_variant" in summary
        assert "delta_magnitude_by_variant_normalized" in summary
        # token data populated → normalized field non-empty.
        assert summary["delta_magnitude_by_variant_normalized"] != {}
        assert "magnitudes_total_normalized" in summary

        geo_dump = json.loads(geo_path.read_text())
        assert geo_dump["schema_version"] in ("3", "4")

        assert result.output_paths["summary_json"] == summary_path
        assert result.output_paths["georesult_json"] == geo_path


# ── plot_family_geometry ──────────────────────────────────────────────


_HAS_MATPLOTLIB = False
try:
    import matplotlib  # noqa: F401
    _HAS_MATPLOTLIB = True
except ImportError:
    pass


@pytest.mark.skipif(not _HAS_MATPLOTLIB, reason="matplotlib not available")
class TestPlotFamilyGeometry:
    @staticmethod
    def _force_agg():
        import matplotlib
        matplotlib.use("Agg")

    def test_renders_direction_heatmap_from_minimal_geo(self, tmp_path):
        self._force_agg()
        geo = _make_geo(
            variants=("v1", "v2"),
            probe_texts=("p0", "p1"),
            domains=("d", "d"),
            avg_tokens=(4.0, 4.0),
        )
        rendered = plot_family_geometry(geo, tmp_path)
        assert "direction_heatmap" in rendered
        assert (tmp_path / "direction_heatmap.png").exists()

    def test_loads_from_json_path(self, tmp_path):
        self._force_agg()
        from lmdiff.report.json_report import write_json
        geo = _make_geo(
            variants=("v1", "v2"),
            probe_texts=("p0", "p1"),
            domains=("d", "d"),
            avg_tokens=(4.0, 4.0),
        )
        json_path = tmp_path / "geo.json"
        write_json(geo, json_path)
        out_dir = tmp_path / "figs"
        rendered = plot_family_geometry(json_path, out_dir)
        assert (out_dir / "direction_heatmap.png").exists()
        assert "direction_heatmap" in rendered

    def test_missing_json_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            plot_family_geometry(tmp_path / "does_not_exist.json", tmp_path)
