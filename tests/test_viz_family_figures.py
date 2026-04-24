"""Smoke tests for the v0.2.3 paper-grade family-figure suite.

Each plotter is asserted to produce a non-trivial PNG; image content is
not validated. matplotlib is forced to the Agg backend so CI runs
headless. Skip the entire module when matplotlib is unavailable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

try:
    import matplotlib
    matplotlib.use("Agg")
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

if not _HAS_MPL:
    pytest.skip("matplotlib not installed", allow_module_level=True)

from lmdiff.experiments.family import TASK_TO_DOMAIN
from lmdiff.geometry import GeoResult
from lmdiff.probes.loader import Probe
from lmdiff.viz.cosine import (
    plot_cosine_heatmap_raw,
    plot_cosine_heatmap_selective,
)
from lmdiff.viz.family_figures import FIGURE_REGISTRY, plot_family_figures
from lmdiff.viz.normalization_effect import plot_normalization_effect
from lmdiff.viz.normalized_magnitude import plot_normalized_magnitude_heatmap
from lmdiff.viz.pca import plot_pca_scatter_normalized, plot_pca_scatter_raw
from lmdiff.viz.specialization import plot_specialization_heatmap


def _synthetic_georesult_v4(
    *,
    variants: tuple[str, ...] = ("yarn", "long", "code", "math"),
) -> GeoResult:
    """4-variant × 5-domain × 4-probes-per-domain GeoResult that exercises
    every path in the figure suite (selective cosine, probe domains, token
    counts, distinct per-variant per-domain magnitudes)."""
    domain_for_task = TASK_TO_DOMAIN
    tasks = list(domain_for_task.keys())
    probes_per_task = 4
    probe_texts = [f"{t}::p{j}" for t in tasks for j in range(probes_per_task)]
    domains = [domain_for_task[t] for t in tasks for _ in range(probes_per_task)]
    n = len(probe_texts)

    per_probe = {}
    for vi, v in enumerate(variants):
        d = {}
        for ti, t in enumerate(tasks):
            base = float((vi + 1) * (ti + 1))
            for j in range(probes_per_task):
                d[f"{t}::p{j}"] = base + 0.1 * j
        per_probe[v] = d
    cv = {v: [per_probe[v][t] for t in probe_texts] for v in variants}
    mags = {v: float(sum(x * x for x in cv[v]) ** 0.5) for v in variants}
    cos = {a: {b: 1.0 if a == b else 0.5 for b in variants} for a in variants}
    sel = {a: {b: 1.0 if a == b else 0.4 for b in variants} for a in variants}
    return GeoResult(
        base_name="base-mock",
        variant_names=list(variants),
        n_probes=n,
        magnitudes=mags,
        cosine_matrix=cos,
        selective_cosine_matrix=sel,
        change_vectors=cv,
        per_probe=per_probe,
        probe_domains=tuple(domains),
        avg_tokens_per_probe=tuple([8.0] * n),
        magnitudes_normalized={v: mags[v] / 12.0 for v in variants},
    )


@pytest.fixture
def synthetic_georesult_v4() -> GeoResult:
    return _synthetic_georesult_v4()


@pytest.fixture
def synthetic_georesult_v4_jsonpath(tmp_path) -> Path:
    """Round-trip through the writer so the CLI can load it from disk."""
    from lmdiff.report.json_report import write_json
    geo = _synthetic_georesult_v4()
    path = tmp_path / "geo_v4.json"
    write_json(geo, path)
    return path


def _assert_png(path: Path) -> None:
    assert path.exists(), path
    assert path.stat().st_size > 1000, (
        f"PNG suspiciously small: {path.stat().st_size} bytes"
    )


# ── per-plotter smoke tests ────────────────────────────────────────────


def test_plot_cosine_raw_emits_png(tmp_path, synthetic_georesult_v4):
    out = plot_cosine_heatmap_raw(synthetic_georesult_v4, tmp_path / "cos_raw.png")
    _assert_png(out)


def test_plot_cosine_selective_emits_png(tmp_path, synthetic_georesult_v4):
    out = plot_cosine_heatmap_selective(synthetic_georesult_v4, tmp_path / "cos_sel.png")
    _assert_png(out)


def test_plot_normalized_magnitude_emits_png(tmp_path, synthetic_georesult_v4):
    out = plot_normalized_magnitude_heatmap(
        synthetic_georesult_v4, tmp_path / "norm_mag.png",
    )
    _assert_png(out)


def test_plot_specialization_emits_png(tmp_path, synthetic_georesult_v4):
    out = plot_specialization_heatmap(synthetic_georesult_v4, tmp_path / "spec.png")
    _assert_png(out)


def test_plot_pca_raw_emits_png(tmp_path, synthetic_georesult_v4):
    out = plot_pca_scatter_raw(synthetic_georesult_v4, tmp_path / "pca_raw.png")
    _assert_png(out)


def test_plot_pca_normalized_emits_png(tmp_path, synthetic_georesult_v4):
    out = plot_pca_scatter_normalized(synthetic_georesult_v4, tmp_path / "pca_norm.png")
    _assert_png(out)


def test_plot_normalization_effect_emits_png(tmp_path, synthetic_georesult_v4):
    out = plot_normalization_effect(synthetic_georesult_v4, tmp_path / "norm_eff.png")
    _assert_png(out)


# ── orchestrator ───────────────────────────────────────────────────────


def test_plot_family_figures_all_emits_seven(tmp_path, synthetic_georesult_v4):
    rendered = plot_family_figures(synthetic_georesult_v4, tmp_path)
    assert set(rendered.keys()) == set(FIGURE_REGISTRY.keys())
    assert len(rendered) == 7
    for path in rendered.values():
        _assert_png(path)


def test_plot_family_figures_subset(tmp_path, synthetic_georesult_v4):
    rendered = plot_family_figures(
        synthetic_georesult_v4, tmp_path,
        which=["specialization", "cosine_raw"],
    )
    assert set(rendered.keys()) == {"specialization", "cosine_raw"}
    assert len(rendered) == 2


def test_plot_family_figures_unknown_key_raises(tmp_path, synthetic_georesult_v4):
    with pytest.raises(ValueError, match="Unknown figure key"):
        plot_family_figures(
            synthetic_georesult_v4, tmp_path, which=["does_not_exist"],
        )


# ── CLI e2e ────────────────────────────────────────────────────────────


def test_plot_geometry_full_emits_seven_figures(tmp_path, synthetic_georesult_v4_jsonpath):
    from typer.testing import CliRunner
    from lmdiff.cli import app
    out_dir = tmp_path / "figs"
    result = CliRunner().invoke(
        app,
        [
            "plot-geometry",
            str(synthetic_georesult_v4_jsonpath),
            "--output-dir", str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    pngs = sorted(out_dir.glob("*.png"))
    assert len(pngs) == 7, [p.name for p in pngs]


def test_plot_geometry_figures_subset_flag(tmp_path, synthetic_georesult_v4_jsonpath):
    from typer.testing import CliRunner
    from lmdiff.cli import app
    out_dir = tmp_path / "figs"
    result = CliRunner().invoke(
        app,
        [
            "plot-geometry",
            str(synthetic_georesult_v4_jsonpath),
            "--output-dir", str(out_dir),
            "--figures", "specialization,cosine_selective",
        ],
    )
    assert result.exit_code == 0, result.output
    pngs = sorted(out_dir.glob("*.png"))
    assert len(pngs) == 2, [p.name for p in pngs]
    names = {p.name for p in pngs}
    assert "04_specialization_zscore.png" in names
    assert "02_cosine_selective.png" in names
