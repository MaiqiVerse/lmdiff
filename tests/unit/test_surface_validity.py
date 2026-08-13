"""Every user-facing output path, driven on data with excluded cells.

This suite exists because v0.4.1 shipped with ``to_html()`` raising
``TypeError`` for every run containing a below-floor domain — the
release's headline scenario — while CI was green, 140 calibration
assertions passed, and 1012 tests passed. Nothing in the release process
rendered an HTML report, so nothing caught it. Three of ten surfaces
were additionally wrong without raising at all.

Four independent assertions per surface. Not-crashing is the weakest of
them and was the only one anybody was checking:

  R  the surface renders without raising
  U  any formula description it emits matches the shared constant,
     rather than a local literal naming a superseded formula
  V  it displays no value for a (variant, domain) cell that the same
     result reports as unmeasured
  A  derived statistics aggregate over measured cells only — excluded
     cells absent from mean / std / set membership, not merely hidden
     at draw time

``A`` is the one the sweep taught us, and the only one that would have
caught the specialization z-score: it displayed the excluded domain
*and* let it contaminate every other cell in the row through the mean
and std.

``test_every_renderer_is_registered`` fails when a renderer is added
without being registered here, so a coverage gap is visible rather than
silent.
"""
from __future__ import annotations

import json
import math
import re
import warnings

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from lmdiff._validity import (  # noqa: E402
    PDN_AXIS_LABEL,
    PDN_DESCRIPTION,
    PDN_FORMULA,
)
from lmdiff.geometry import GeoResult  # noqa: E402

# Fingerprint of the superseded Formula B vocabulary. Any renderer
# emitting this is describing ``sqrt(sum-d2 / sum-T)`` (nats/token^1.5)
# for a column that now contains ``sqrt(mean(d2))`` (nats/token).
FORMULA_B = re.compile(r"√tok|per-√token|/√token|√\(n · ⟨tok⟩\)|√\(N · ⟨tok⟩\)|per √token")

EXCLUDED_DOMAIN = "long-context"
MEASURED_DOMAINS = ("commonsense", "math")


@pytest.fixture(scope="module")
def result() -> GeoResult:
    """Two variants, three domains, one of them excluded for both.

    Values are chosen so every excluded cell has a distinctive raw
    magnitude that ``V`` can search rendered output for.
    """
    n_per = 4
    domains = (
        ("commonsense",) * n_per + ("math",) * n_per + (EXCLUDED_DOMAIN,) * n_per
    )
    cv = {
        # commonsense small, math large, long-context distinctive (7.0)
        "A": [0.5] * n_per + [2.0] * n_per + [7.0] * n_per,
        "B": [1.5] * n_per + [0.5] * n_per + [7.0] * n_per,
    }
    g = GeoResult(
        base_name="base-model",
        variant_names=["A", "B"],
        n_probes=len(domains),
        magnitudes={v: float(sum(x * x for x in c) ** 0.5) for v, c in cv.items()},
        cosine_matrix={"A": {"A": 1.0, "B": 0.4}, "B": {"A": 0.4, "B": 1.0}},
        selective_cosine_matrix={"A": {"A": 1.0, "B": 0.2}, "B": {"A": 0.2, "B": 1.0}},
        change_vectors=cv,
        per_probe={v: {f"p{i}": c[i] for i in range(len(c))} for v, c in cv.items()},
        metadata={"tasks": {}},
        probe_domains=domains,
        avg_tokens_per_probe=tuple([30.0] * n_per + [40.0] * n_per + [9000.0] * n_per),
    )
    g.domain_status = {
        v: {
            "commonsense": "full",
            "math": "full",
            EXCLUDED_DOMAIN: status,
        }
        for v, status in (("A", "out_of_range"), ("B", "variant_only"))
    }
    # measured cells only; excluded carry None, as the pipeline emits
    g.magnitudes_per_domain_normalized = {
        "A": {"commonsense": 0.5, "math": 2.0, EXCLUDED_DOMAIN: None},
        "B": {"commonsense": 1.5, "math": 0.5, EXCLUDED_DOMAIN: None},
    }
    g.share_per_domain = {
        "A": {"commonsense": 0.0588, "math": 0.9412, EXCLUDED_DOMAIN: None},
        "B": {"commonsense": 0.9, "math": 0.1, EXCLUDED_DOMAIN: None},
    }
    g.magnitudes_normalized = {"A": 1.46, "B": 1.12}
    return g


# ── surface registry ─────────────────────────────────────────────────
#
# kind="text"   -> R, U, V checked on the rendered string
# kind="figure" -> R checked by rendering; U and V are checked on the
#                  accessor the figure consumes, since a PNG cannot be
#                  grepped. A figure reading a validity-aware accessor
#                  cannot draw an excluded cell as a number.


def _md(result, tmp_path):
    p = tmp_path / "r.md"
    result.to_markdown(str(p))
    return p.read_text(encoding="utf-8")


def _html(result, tmp_path):
    p = tmp_path / "r.html"
    result.to_html(str(p))
    return p.read_text(encoding="utf-8")


def _terminal(result, tmp_path):
    from lmdiff.report.terminal import render
    return render(result, color=False)


def _json(result, tmp_path):
    from lmdiff.report.json_report import to_json
    return to_json(result)


TEXT_SURFACES = {
    "to_markdown": _md,
    "to_html": _html,
    "terminal": _terminal,
    "to_json": _json,
}


def _fig_drift_share(result, path):
    from lmdiff.viz.drift_share import render_drift_share
    return render_drift_share(result, path)


def _fig_direction(result, path):
    from lmdiff.viz.direction import render_direction
    return render_direction(result, path)


def _fig_change_size(result, path):
    from lmdiff.viz.change_size import render_change_size
    return render_change_size(result, path)


def _fig_domain_bar(result, path):
    """In no figure tier, and it takes the heatmap as a *parameter* — so
    every caller decides independently whether to filter, and every
    caller can get it wrong. Registered here for exactly that reason:
    the assertion is that the filtered heatmap is what a correct caller
    passes, and that the figure renders from it."""
    from lmdiff._validity import filter_measured_cells
    from lmdiff.viz.domain_bar import plot_domain_bar

    heat = filter_measured_cells(result.domain_status, result.domain_heatmap())
    for v, row in heat.items():
        assert EXCLUDED_DOMAIN not in row, (
            "filter_measured_cells must drop the excluded domain before "
            "plot_domain_bar sees it — the figure has no validity input "
            "of its own"
        )
    return plot_domain_bar(heat, out_path=str(path))


def _paper(key):
    def _render(result, path):
        from lmdiff.viz.family_figures import plot_family_figures
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return plot_family_figures(result, path.parent, which=[key])
    return _render


FIGURE_SURFACES = {
    "applied:drift_share": _fig_drift_share,
    "applied:direction": _fig_direction,
    "applied:change_size": _fig_change_size,
    "viz:domain_bar": _fig_domain_bar,
    "paper:cosine_raw": _paper("cosine_raw"),
    "paper:cosine_selective": _paper("cosine_selective"),
    "paper:normalized_magnitude": _paper("normalized_magnitude"),
    "paper:specialization": _paper("specialization"),
    "paper:pca_raw": _paper("pca_raw"),
    "paper:pca_normalized": _paper("pca_normalized"),
    "paper:normalization_effect": _paper("normalization_effect"),
}


# ── coverage gate ────────────────────────────────────────────────────


def test_every_renderer_is_registered():
    """Adding a renderer without registering it here is a coverage gap;
    make it visible instead of silent."""
    from lmdiff.viz.family_figures import FIGURE_REGISTRY

    registered = {k.split(":", 1)[1] for k in FIGURE_SURFACES if k.startswith("paper:")}
    assert set(FIGURE_REGISTRY) == registered, (
        "paper-tier figures changed; register the new one in FIGURE_SURFACES"
    )

    # applied tier, as enumerated by report.figures
    import inspect
    from lmdiff.report import figures as figures_mod
    src = inspect.getsource(figures_mod)
    for name in ("render_drift_share", "render_direction", "render_change_size"):
        assert name in src
    applied = {k for k in FIGURE_SURFACES if k.startswith("applied:")}
    assert len(applied) == 3, "applied tier changed; update FIGURE_SURFACES"


# ── R: renders without raising ───────────────────────────────────────


@pytest.mark.parametrize("name", sorted(TEXT_SURFACES))
def test_R_text_surface_does_not_raise(result, tmp_path, name):
    """v0.4.1's to_html() raised TypeError here for every run with an
    excluded domain."""
    out = TEXT_SURFACES[name](result, tmp_path)
    assert out and isinstance(out, str)


@pytest.mark.parametrize("name", sorted(FIGURE_SURFACES))
def test_R_figure_surface_does_not_raise(result, tmp_path, name):
    FIGURE_SURFACES[name](result, tmp_path / f"{name.replace(':', '_')}.png")


# ── U: formula description matches the shared constant ───────────────


@pytest.mark.parametrize("name", sorted(TEXT_SURFACES))
def test_U_no_superseded_formula_vocabulary(result, tmp_path, name):
    text = TEXT_SURFACES[name](result, tmp_path)
    hits = sorted(set(FORMULA_B.findall(text)))
    assert not hits, (
        f"{name} describes the superseded Formula B: {hits}. "
        f"Use lmdiff._validity.PDN_* rather than a local literal."
    )


@pytest.mark.parametrize("name", ["to_markdown", "to_html", "terminal"])
def test_U_uses_the_shared_label(result, tmp_path, name):
    """Positive half of U: the renderer must actually reference the
    shared constant, not merely avoid the old literal."""
    text = TEXT_SURFACES[name](result, tmp_path)
    assert PDN_AXIS_LABEL in text or PDN_DESCRIPTION in text


def test_U_viz_modules_reference_the_constant_not_a_literal():
    import inspect
    from lmdiff.viz import change_size, normalization_effect, normalized_magnitude
    for mod in (change_size, normalization_effect, normalized_magnitude):
        src = inspect.getsource(mod)
        assert not FORMULA_B.search(src.split('"""', 2)[-1]), (
            f"{mod.__name__} still carries a Formula-B literal"
        )
        assert "PDN_FORMULA" in src or "PDN_UNITS" in src


# ── V: no value shown for an unmeasured cell ─────────────────────────


def _excluded_raw_values(result):
    heat = result.domain_heatmap()
    return {
        v: heat[v][EXCLUDED_DOMAIN]
        for v in result.variant_names
        if result.share_per_domain[v][EXCLUDED_DOMAIN] is None
    }


@pytest.mark.parametrize("name", sorted(TEXT_SURFACES))
def test_V_no_excluded_cell_value_is_rendered(result, tmp_path, name):
    """Cross-references one surface against itself: any cell the share
    table reports as unmeasured must not appear as a number anywhere
    else in the same output. v0.4.1's markdown and terminal drift tables
    printed all of them."""
    if name == "to_json":
        pytest.skip("JSON carries raw fields by design; see test_V_json_fields")
    text = TEXT_SURFACES[name](result, tmp_path)
    for variant, raw in _excluded_raw_values(result).items():
        assert f"{raw:.4f}" not in text, (
            f"{name} renders {raw:.4f} for {variant}/{EXCLUDED_DOMAIN}, "
            f"which share_per_domain reports as unmeasured"
        )


def test_V_json_fields_are_none_for_excluded_cells(result, tmp_path):
    """JSON is a data format, not a display: it must carry None on the
    per-domain fields and must not serialize domain_heatmap at all."""
    d = json.loads(TEXT_SURFACES["to_json"](result, tmp_path))
    assert "domain_heatmap" not in d
    for v in result.variant_names:
        assert d["share_per_domain"][v][EXCLUDED_DOMAIN] is None
        assert d["magnitudes_per_domain_normalized"][v][EXCLUDED_DOMAIN] is None


def test_V_figure_accessors_are_validity_aware(result):
    """Figures cannot be grepped, so check the accessors they read.
    A figure consuming a validity-aware accessor cannot draw an excluded
    cell as a number."""
    per_task = result.magnitudes_per_task_normalized()   # normalized_magnitude
    zscore = result.magnitudes_specialization_zscore()   # specialization
    for v in result.variant_names:
        assert math.isnan(per_task[v][EXCLUDED_DOMAIN])
        assert math.isnan(zscore[v][EXCLUDED_DOMAIN])


# ── A: derived statistics aggregate over measured cells only ─────────


def test_A_zscore_excludes_unmeasured_from_mean_and_std(result):
    """The assertion the sweep taught us, and the only one that would
    have caught the z-score: an excluded cell must be absent from the
    statistic, not merely hidden when drawn.

    Two measured domains z-scored against each other give exactly ±1.
    If the excluded domain were still in the mean and std, they would
    not."""
    zscore = result.magnitudes_specialization_zscore()
    for v in result.variant_names:
        vals = sorted(zscore[v][d] for d in MEASURED_DOMAINS)
        assert vals[0] == pytest.approx(-1.0)
        assert vals[1] == pytest.approx(+1.0)


def test_A_zscore_filters_its_own_input(result, monkeypatch):
    """Isolates the z-score's aggregation from its source.

    There are two independent guards: ``magnitudes_per_task_normalized``
    excludes the cell, and the z-score excludes it again. Composed, they
    mask each other — reverting either alone leaves the other covering
    it, so a test that only checks the composed output cannot tell which
    (or whether both) is doing the work. A mutation check on the v0.4.2
    branch confirmed exactly that: reverting the z-score's own filter
    changed nothing observable.

    So hand the z-score an unfiltered source and require it to exclude
    the cell itself. Defense in depth is only defense if each layer is
    pinned separately.
    """
    unfiltered = {
        v: {"commonsense": 1.0, "math": 3.0, EXCLUDED_DOMAIN: 99.0}
        for v in result.variant_names
    }
    monkeypatch.setattr(
        type(result), "magnitudes_per_task_normalized",
        lambda self: {v: dict(row) for v, row in unfiltered.items()},
    )
    zscore = result.magnitudes_specialization_zscore()
    for v in result.variant_names:
        assert math.isnan(zscore[v][EXCLUDED_DOMAIN]), (
            "z-score must exclude the cell even when its source supplies "
            "a value for it"
        )
        # 1.0 and 3.0 z-scored against each other are exactly ±1; the
        # 99.0 would flatten them to roughly -0.7 / -0.7 if included.
        vals = sorted(zscore[v][d] for d in MEASURED_DOMAINS)
        assert vals[0] == pytest.approx(-1.0)
        assert vals[1] == pytest.approx(+1.0)


def test_A_complementarity_excludes_unmeasured_from_affected_sets(result):
    """Set membership is an aggregation too."""
    c = result.complementarity("A", "B", threshold=0.01)
    seen = set(c.overlap_domains) | set(c.unique_v1_domains) | set(c.unique_v2_domains)
    assert EXCLUDED_DOMAIN not in seen


def test_A_share_rows_sum_to_one_over_measured_cells(result):
    for v in result.variant_names:
        total = sum(
            s for s in result.share_per_domain[v].values() if s is not None
        )
        assert total == pytest.approx(1.0, abs=1e-3)


def test_A_report_drift_totals_exclude_unmeasured(result):
    """The per-variant total in the report drift tables aggregates the
    per-domain column, so it must exclude the same cells the column
    does."""
    from lmdiff.report.markdown import _domain_drift, _per_variant_total_drift
    drift = _domain_drift(result)
    for v in result.variant_names:
        assert EXCLUDED_DOMAIN not in drift[v]
    totals = _per_variant_total_drift(drift)
    assert set(totals) == set(result.variant_names)
