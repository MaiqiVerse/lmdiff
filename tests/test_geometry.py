"""Tests for ChangeGeometry (geometry.py)."""
from __future__ import annotations

import inspect
import io
import json
import math
import re
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from typer.testing import CliRunner

from lmdiff.cli import _parse_variant_spec, app
from lmdiff.config import Config
from lmdiff.geometry import ChangeGeometry, GeoResult
from lmdiff.probes.loader import Probe, ProbeSet

V01_PATH = Path(__file__).parent.parent / "lmdiff" / "probes" / "v01.json"


# ── Mock engine factory ─────────────────────────────────────────────────

def _make_mock_engine(
    name: str,
    completions: list[str],
    self_ce: list[float],
    base_of_v_ce: list[float] | None = None,
    tokens_per_probe: int = 4,
):
    """Return a mock that quacks like an InferenceEngine.

    self_ce is used when score() is called with continuation_ids (self-score path).
    base_of_v_ce is used when score() is called with continuations (base-of-variant path).
    Mock tokenizer is shared across all engines by default (same vocab_size) so
    ChangeGeometry.shares_tokenizer_with returns True (same model string) or the
    fallback tokenizers_equivalent check trivially passes.
    """
    engine = MagicMock(name=f"engine-{name}")
    engine.model_name = name

    # generate → GenerationResult-like: one sample per prompt
    def generate_side(prompts, n_samples=1, max_new_tokens=16):
        n = len(prompts)
        # Each prompt gets one completion and one token id list of fixed length
        gen = MagicMock()
        gen.completions = [[completions[i % len(completions)]] for i in range(n)]
        gen.token_ids = [
            [list(range(tokens_per_probe))] for _ in range(n)
        ]
        return gen

    def score_side(prompts, continuations=None, continuation_ids=None):
        n = len(prompts)
        sr = MagicMock()
        if continuation_ids is not None:
            # self-score path
            ces = list(self_ce[:n])
        else:
            # base-of-variant score path
            if base_of_v_ce is None:
                raise AssertionError(
                    f"engine {name}: score(continuations=...) called but no base_of_v_ce configured"
                )
            ces = list(base_of_v_ce[:n])
        sr.cross_entropies = ces
        sr.token_ids = [list(range(tokens_per_probe)) for _ in range(n)]
        return sr

    engine.generate.side_effect = generate_side
    engine.score.side_effect = score_side
    return engine


class _EngineFactory:
    """Callable that returns a pre-built mock engine when InferenceEngine(config) is called.

    Keys on config.display_name so tests can share a single model string across
    configs (keeps shares_tokenizer_with → True so BPB does not kick in) and
    distinguish engines via `name=`.
    """

    def __init__(self, engines_by_name: dict[str, MagicMock]) -> None:
        self.engines_by_name = engines_by_name
        self.calls: list[Config] = []

    def __call__(self, config: Config, device: str | None = None) -> MagicMock:
        self.calls.append(config)
        return self.engines_by_name[config.display_name]


def _install_mock_engines(
    monkeypatch, engines_by_name: dict[str, MagicMock]
) -> _EngineFactory:
    factory = _EngineFactory(engines_by_name)
    monkeypatch.setattr("lmdiff.geometry.InferenceEngine", factory)
    return factory


def _cfg(name: str) -> Config:
    """All test configs share the same model string so shares_tokenizer_with is True."""
    return Config(model="mock-model", name=name)


# ── TestChangeVectorComputation ─────────────────────────────────────────

class TestChangeVectorComputation:
    def test_delta_matches_manual(self, monkeypatch):
        probes = ["p0", "p1", "p2"]
        # variant A: base_of_v CE - v_self CE → [3-1, 4-2, 5-3] = [2, 2, 2]
        engine_a = _make_mock_engine(
            "A",
            completions=["a0", "a1", "a2"],
            self_ce=[1.0, 2.0, 3.0],
        )
        # variant B: [2-1, 5-3, 1-4] = [1, 2, -3]
        engine_b = _make_mock_engine(
            "B",
            completions=["b0", "b1", "b2"],
            self_ce=[1.0, 3.0, 4.0],
        )
        engine_base = _make_mock_engine(
            "base",
            completions=["ignored"],
            self_ce=[0.0],
            base_of_v_ce=[3.0, 4.0, 5.0],  # will be overridden per call below
        )

        # base needs different CE sequences per variant call. Configure a callable
        # that dispatches based on which continuations text list it receives.
        def base_score_side(prompts, continuations=None, continuation_ids=None):
            sr = MagicMock()
            if continuations is None:
                raise AssertionError("base should only be called with continuations")
            if continuations == ["a0", "a1", "a2"]:
                sr.cross_entropies = [3.0, 4.0, 5.0]
            elif continuations == ["b0", "b1", "b2"]:
                sr.cross_entropies = [2.0, 5.0, 1.0]
            else:
                raise AssertionError(f"unexpected continuations: {continuations}")
            sr.token_ids = [[0, 1, 2, 3]] * len(prompts)
            return sr

        engine_base.score.side_effect = base_score_side

        _install_mock_engines(
            monkeypatch,
            {"base": engine_base, "A": engine_a, "B": engine_b},
        )

        base_cfg = _cfg("base")
        variants = {
            "A": _cfg("A"),
            "B": _cfg("B"),
        }
        cg = ChangeGeometry(base=base_cfg, variants=variants, prompts=probes)
        result = cg.analyze(max_new_tokens=8)

        np.testing.assert_allclose(result.change_vectors["A"], [2.0, 2.0, 2.0])
        np.testing.assert_allclose(result.change_vectors["B"], [1.0, 2.0, -3.0])

        # Magnitudes = L2 norm
        assert abs(result.magnitudes["A"] - math.sqrt(4 + 4 + 4)) < 1e-9
        assert abs(result.magnitudes["B"] - math.sqrt(1 + 4 + 9)) < 1e-9

        # Diagonal = 1.0 when magnitude > 0
        assert result.cosine_matrix["A"]["A"] == 1.0
        assert result.cosine_matrix["B"]["B"] == 1.0

        # Symmetry
        assert result.cosine_matrix["A"]["B"] == result.cosine_matrix["B"]["A"]

        # n_probes == 3 (no NaNs)
        assert result.n_probes == 3
        assert result.metadata["n_skipped"] == 0

    def test_per_probe_keyed_by_text(self, monkeypatch):
        probes = ["The capital of France is ", "2 + 2 = "]
        engine_v = _make_mock_engine("V", ["out0", "out1"], self_ce=[1.0, 1.0])
        engine_base = _make_mock_engine(
            "base", ["ignored"], self_ce=[0.0],
            base_of_v_ce=[2.0, 2.0],
        )
        _install_mock_engines(monkeypatch, {"base": engine_base, "V": engine_v})

        cg = ChangeGeometry(
            base=_cfg("base"),
            variants={"V": _cfg("V")},
            prompts=probes,
        )
        result = cg.analyze()

        assert set(result.per_probe["V"].keys()) == {
            "The capital of France is ",
            "2 + 2 = ",
        }
        assert result.per_probe["V"]["The capital of France is "] == pytest.approx(1.0)


class TestChangeVectorProperties:
    def test_identical_variants_cos_is_one(self, monkeypatch):
        probes = ["p0", "p1", "p2"]

        def make_engine(tag):
            # Both variants produce same outputs and CEs → same δ vector
            eng = _make_mock_engine(tag, ["o0", "o1", "o2"], self_ce=[1.0, 1.0, 1.0])
            return eng

        engine_v1 = make_engine("V1")
        engine_v2 = make_engine("V2")

        def base_score_side(prompts, continuations=None, continuation_ids=None):
            sr = MagicMock()
            sr.cross_entropies = [2.0, 3.0, 4.0]
            sr.token_ids = [[0, 1, 2, 3]] * len(prompts)
            return sr

        engine_base = MagicMock(name="engine-base")
        engine_base.model_name = "base"
        engine_base.score.side_effect = base_score_side

        _install_mock_engines(
            monkeypatch,
            {"base": engine_base, "V1": engine_v1, "V2": engine_v2},
        )

        cg = ChangeGeometry(
            base=_cfg("base"),
            variants={"V1": _cfg("V1"), "V2": _cfg("V2")},
            prompts=probes,
        )
        result = cg.analyze()

        assert result.cosine_matrix["V1"]["V2"] == pytest.approx(1.0)
        assert result.cosine_matrix["V2"]["V1"] == pytest.approx(1.0)
        assert result.magnitudes["V1"] == pytest.approx(result.magnitudes["V2"])

    def test_zero_vector_diagonal_is_nan(self, monkeypatch):
        probes = ["p0", "p1", "p2"]
        # Variant's self-CE equals base-of-variant CE everywhere → δ = 0 vector
        engine_v = _make_mock_engine("V", ["o0", "o1", "o2"], self_ce=[2.0, 2.0, 2.0])

        def base_score_side(prompts, continuations=None, continuation_ids=None):
            sr = MagicMock()
            sr.cross_entropies = [2.0, 2.0, 2.0]
            sr.token_ids = [[0, 1, 2, 3]] * len(prompts)
            return sr

        engine_base = MagicMock(name="engine-base")
        engine_base.model_name = "base"
        engine_base.score.side_effect = base_score_side

        _install_mock_engines(
            monkeypatch, {"base": engine_base, "V": engine_v},
        )
        cg = ChangeGeometry(
            base=_cfg("base"),
            variants={"V": _cfg("V")},
            prompts=probes,
        )
        result = cg.analyze()

        assert result.magnitudes["V"] == 0.0
        assert math.isnan(result.cosine_matrix["V"]["V"])


# ── TestNaNHandling ─────────────────────────────────────────────────────

class TestNaNHandling:
    def test_nan_in_one_variant_filters_globally(self, monkeypatch):
        probes = ["p0", "p1", "p2"]
        # A is fine on all probes; B has NaN on probe 1
        engine_a = _make_mock_engine("A", ["a0", "a1", "a2"], self_ce=[1.0, 1.0, 1.0])
        engine_b = _make_mock_engine(
            "B", ["b0", "b1", "b2"],
            self_ce=[1.0, float("nan"), 1.0],
        )

        def base_score_side(prompts, continuations=None, continuation_ids=None):
            sr = MagicMock()
            if continuations == ["a0", "a1", "a2"]:
                sr.cross_entropies = [2.0, 2.0, 2.0]
            elif continuations == ["b0", "b1", "b2"]:
                sr.cross_entropies = [2.0, 2.0, 2.0]
            else:
                raise AssertionError(f"unexpected: {continuations}")
            sr.token_ids = [[0, 1, 2, 3]] * len(prompts)
            return sr

        engine_base = MagicMock(name="engine-base")
        engine_base.model_name = "base"
        engine_base.score.side_effect = base_score_side

        _install_mock_engines(
            monkeypatch,
            {"base": engine_base, "A": engine_a, "B": engine_b},
        )

        cg = ChangeGeometry(
            base=_cfg("base"),
            variants={"A": _cfg("A"), "B": _cfg("B")},
            prompts=probes,
        )
        result = cg.analyze()

        assert result.n_probes == 2
        assert result.metadata["n_skipped"] == 1
        assert len(result.change_vectors["A"]) == 2
        assert len(result.change_vectors["B"]) == 2
        assert set(result.per_probe["A"].keys()) == {"p0", "p2"}
        assert set(result.per_probe["B"].keys()) == {"p0", "p2"}

    def test_zero_valid_probes_returns_empty_result(self, monkeypatch):
        probes = ["p0", "p1"]
        engine_v = _make_mock_engine(
            "V", ["o0", "o1"], self_ce=[float("nan"), float("nan")],
        )

        def base_score_side(prompts, continuations=None, continuation_ids=None):
            sr = MagicMock()
            sr.cross_entropies = [1.0, 1.0]
            sr.token_ids = [[0, 1, 2, 3]] * len(prompts)
            return sr

        engine_base = MagicMock(name="engine-base")
        engine_base.model_name = "base"
        engine_base.score.side_effect = base_score_side

        _install_mock_engines(
            monkeypatch, {"base": engine_base, "V": engine_v},
        )
        cg = ChangeGeometry(
            base=_cfg("base"),
            variants={"V": _cfg("V")},
            prompts=probes,
        )
        result = cg.analyze()

        assert result.n_probes == 0
        assert result.change_vectors["V"] == []
        assert result.magnitudes["V"] == 0.0
        assert math.isnan(result.cosine_matrix["V"]["V"])
        assert result.per_probe["V"] == {}
        assert result.metadata["n_skipped"] == 2

    def test_empty_probe_set_raises(self, monkeypatch):
        engine_v = _make_mock_engine("V", ["o"], self_ce=[1.0])
        engine_base = _make_mock_engine(
            "base", ["ignored"], self_ce=[0.0], base_of_v_ce=[1.0],
        )
        _install_mock_engines(
            monkeypatch, {"base": engine_base, "V": engine_v},
        )
        cg = ChangeGeometry(
            base=Config(model="base"),
            variants={"V": Config(model="V")},
            prompts=[],
        )
        with pytest.raises(ValueError, match="empty probe set"):
            cg.analyze()


# ── TestGeoResultSummaryTable ───────────────────────────────────────────

class TestGeoResultSummaryTable:
    def test_row_count_matches_variants(self):
        result = GeoResult(
            base_name="base",
            variant_names=["A", "B", "C"],
            n_probes=3,
            magnitudes={"A": 1.5, "B": 2.0, "C": 0.5},
            cosine_matrix={
                "A": {"A": 1.0, "B": 0.5, "C": 0.1},
                "B": {"A": 0.5, "B": 1.0, "C": 0.2},
                "C": {"A": 0.1, "B": 0.2, "C": 1.0},
            },
            change_vectors={"A": [1, 1, 1], "B": [2, 0, 0], "C": [0.5, 0, 0]},
            per_probe={"A": {}, "B": {}, "C": {}},
        )
        rows = result.summary_table()
        assert len(rows) == 3
        for row in rows:
            assert set(row.keys()) == {"variant", "magnitude", "cosines"}
            assert isinstance(row["cosines"], dict)
            assert set(row["cosines"].keys()) == {"A", "B", "C"}


# ── TestArchitecture ────────────────────────────────────────────────────

class TestArchitecture:
    def test_geometry_does_not_import_transformers(self):
        import lmdiff.geometry as mod
        src = inspect.getsource(mod)
        assert "import transformers" not in src
        assert "from transformers" not in src

    def test_geometry_does_not_import_metrics(self):
        import lmdiff.geometry as mod
        src = inspect.getsource(mod)
        assert "from lmdiff.metrics" not in src
        assert "import lmdiff.metrics" not in src

    def test_geometry_does_not_import_torch(self):
        import lmdiff.geometry as mod
        src = inspect.getsource(mod)
        # torch is reached only via engine.release_cuda_cache()
        assert not re.search(r"^\s*import torch\b", src, re.MULTILINE)
        assert not re.search(r"^\s*from torch\b", src, re.MULTILINE)

    def test_geometry_imports_required_helpers(self):
        import lmdiff.geometry as mod
        src = inspect.getsource(mod)
        assert "from lmdiff.engine" in src
        assert "from lmdiff.config" in src
        assert "release_cuda_cache" in src


# ── TestJsonGeometry ────────────────────────────────────────────────────

def _make_fake_geo_result() -> GeoResult:
    return GeoResult(
        base_name="llama2-7b",
        variant_names=["13b", "70b", "chat"],
        n_probes=4,
        magnitudes={"13b": 1.52, "70b": 2.41, "chat": 1.89},
        cosine_matrix={
            "13b": {"13b": 1.0, "70b": 0.87, "chat": 0.31},
            "70b": {"13b": 0.87, "70b": 1.0, "chat": 0.35},
            "chat": {"13b": 0.31, "70b": 0.35, "chat": 1.0},
        },
        change_vectors={
            "13b": [1.0, 0.5, 0.8, 0.4],
            "70b": [1.5, 0.8, 1.3, 0.6],
            "chat": [0.9, 1.2, 0.4, 1.1],
        },
        per_probe={
            "13b": {"p0": 1.0, "p1": 0.5, "p2": 0.8, "p3": 0.4},
            "70b": {"p0": 1.5, "p1": 0.8, "p2": 1.3, "p3": 0.6},
            "chat": {"p0": 0.9, "p1": 1.2, "p2": 0.4, "p3": 1.1},
        },
        metadata={
            "n_total_probes": 4,
            "n_skipped": 0,
            "bpb_normalized": {"13b": False, "70b": False, "chat": False},
            "max_new_tokens": 16,
        },
    )


class TestJsonGeometry:
    def test_round_trip(self):
        from lmdiff.report.json_report import to_json, to_json_dict

        r = _make_fake_geo_result()
        d = to_json_dict(r)
        s = to_json(r)
        reloaded = json.loads(s)
        assert reloaded["schema_version"] == "1"
        assert reloaded["base_name"] == "llama2-7b"
        assert reloaded["variant_names"] == ["13b", "70b", "chat"]
        assert reloaded["magnitudes"]["13b"] == pytest.approx(1.52)
        assert "generated_at" in reloaded

    def test_nan_cos_becomes_null(self):
        from lmdiff.report.json_report import to_json

        r = GeoResult(
            base_name="base",
            variant_names=["V"],
            n_probes=0,
            magnitudes={"V": 0.0},
            cosine_matrix={"V": {"V": float("nan")}},
            change_vectors={"V": []},
            per_probe={"V": {}},
            metadata={"n_total_probes": 2, "n_skipped": 2, "bpb_normalized": {"V": False}},
        )
        reloaded = json.loads(to_json(r))
        assert reloaded["cosine_matrix"]["V"]["V"] is None

    def test_deterministic_modulo_timestamp(self):
        from lmdiff.report.json_report import to_json

        r = _make_fake_geo_result()
        s1 = to_json(r)
        s2 = to_json(r)

        def strip_ts(s: str) -> str:
            return "\n".join(
                line for line in s.splitlines() if '"generated_at"' not in line
            )

        assert strip_ts(s1) == strip_ts(s2)


# ── TestTerminalGeometry ────────────────────────────────────────────────

class TestTerminalGeometry:
    def test_print_no_crash(self):
        from rich.console import Console
        from lmdiff.report.terminal import print_geometry

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120)
        r = _make_fake_geo_result()
        print_geometry(r, console=console)
        out = buf.getvalue()
        assert "Change Geometry" in out
        assert "13b" in out
        assert "70b" in out
        # cosine values rendered with sign
        assert "+0.870" in out or "+0.87" in out

    def test_bpb_footer(self):
        from rich.console import Console
        from lmdiff.report.terminal import print_geometry

        r = _make_fake_geo_result()
        r.metadata["bpb_normalized"] = {"13b": False, "70b": True, "chat": True}
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120)
        print_geometry(r, console=console)
        out = buf.getvalue()
        assert "BPB-normalized" in out
        assert "70b" in out and "chat" in out

    def test_skipped_footer(self):
        from rich.console import Console
        from lmdiff.report.terminal import print_geometry

        r = _make_fake_geo_result()
        r.metadata["n_skipped"] = 5
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120)
        print_geometry(r, console=console)
        out = buf.getvalue()
        assert "skipped 5" in out


# ── TestCLIGeometry ─────────────────────────────────────────────────────

class TestCLIGeometry:
    def setup_method(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(app, ["geometry", "--help"])
        assert result.exit_code == 0

    def test_parse_variant_spec_ok(self):
        name, model_id = _parse_variant_spec("13b=meta-llama/Llama-2-13b-hf")
        assert name == "13b"
        assert model_id == "meta-llama/Llama-2-13b-hf"

    def test_parse_variant_spec_strip(self):
        name, model_id = _parse_variant_spec("  13b  =  meta-llama/foo  ")
        assert name == "13b"
        assert model_id == "meta-llama/foo"

    def test_parse_variant_spec_no_equals(self):
        import typer
        with pytest.raises(typer.BadParameter):
            _parse_variant_spec("no-equals-here")

    def test_parse_variant_spec_empty_name(self):
        import typer
        with pytest.raises(typer.BadParameter):
            _parse_variant_spec("=model")

    def test_parse_variant_spec_empty_model(self):
        import typer
        with pytest.raises(typer.BadParameter):
            _parse_variant_spec("name=")

    def test_duplicate_variant_name_errors(self):
        result = self.runner.invoke(app, [
            "geometry", "gpt2", "a=gpt2", "a=distilgpt2",
        ])
        assert result.exit_code != 0

    def test_probe_not_found_errors(self):
        result = self.runner.invoke(app, [
            "geometry", "gpt2", "v1=distilgpt2",
            "--probes", "nonexistent_probe_xyz",
        ])
        assert result.exit_code != 0


# ── Slow E2E ────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestGeometryE2E:
    def _make_small_probes(self) -> ProbeSet:
        ps_full = ProbeSet.from_json(V01_PATH)
        by_d = ps_full.by_domain()
        selected: list[Probe] = []
        for d in sorted(by_d):
            selected.extend(list(by_d[d])[:3])
        probes = ProbeSet(selected, name=ps_full.name, version=ps_full.version)
        assert len(set(p.domain for p in probes)) == 3
        return probes

    def test_gpt2_vs_distilgpt2(self, capsys):
        probes = self._make_small_probes()
        cg = ChangeGeometry(
            base=Config(model="gpt2", name="gpt2"),
            variants={"distil": Config(model="distilgpt2", name="distil")},
            prompts=probes,
        )
        result = cg.analyze(max_new_tokens=16)

        assert result.n_probes > 0
        assert len(result.change_vectors["distil"]) == result.n_probes
        assert result.cosine_matrix["distil"]["distil"] == pytest.approx(1.0)
        assert result.metadata["bpb_normalized"]["distil"] is False

        # Print for the record
        print(f"\n--- GeoResult for gpt2 vs distilgpt2 ---")
        print(f"  n_probes={result.n_probes}")
        print(f"  magnitude(distil) = {result.magnitudes['distil']:.4f}")
        print(f"  cos[distil][distil] = {result.cosine_matrix['distil']['distil']:.4f}")
        print(f"  bpb_normalized={result.metadata['bpb_normalized']}")

    def test_two_identical_variants_cos_one(self):
        probes = self._make_small_probes()
        cg = ChangeGeometry(
            base=Config(model="gpt2", name="gpt2"),
            variants={
                "d1": Config(model="distilgpt2", name="d1"),
                "d2": Config(model="distilgpt2", name="d2"),
            },
            prompts=probes,
        )
        result = cg.analyze(max_new_tokens=16)

        # Same engine twice under greedy decode → identical outputs → identical δ
        assert result.cosine_matrix["d1"]["d2"] == pytest.approx(1.0, abs=1e-6)
        assert result.magnitudes["d1"] == pytest.approx(result.magnitudes["d2"], abs=1e-6)
