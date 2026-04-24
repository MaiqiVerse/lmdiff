"""Family geometry + accuracy radar over an lm-eval task mix.

Library entry point for the workflow previously inlined in
``scripts/run_family_geometry_lm_eval.py``: load N lm-eval tasks into a
single concatenated ProbeSet, run ChangeGeometry once, then per-variant
per-task accuracy, and emit a JSON summary + raw GeoResult JSON +
optional radar PNGs.

``plot_family_geometry`` is the library equivalent of
``scripts/plot_family_geometry.py``: render direction heatmap / selective
heatmap / PCA scatter / domain bar from a previously written GeoResult
JSON.

Both functions are imported by the typer CLI (``lmdiff
family-experiment`` / ``lmdiff plot-geometry``) and by the legacy
scripts, which are now thin argparse wrappers.
"""
from __future__ import annotations

import gc
import json
import math
import sys
import time
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

from lmdiff.config import Config
from lmdiff.engine import release_cuda_cache
from lmdiff.geometry import ChangeGeometry, GeoResult
from lmdiff.probes.adapters import KNOWN_TASK_DOMAINS, from_lm_eval
from lmdiff.probes.loader import Probe, ProbeSet
from lmdiff.report.json_report import (
    geo_result_from_json_dict,
    write_json as _write_geo_json,
)
from lmdiff.report.terminal import print_geometry
from lmdiff.tasks.base import Task
from lmdiff.tasks.evaluators import F1, Gsm8kNumberMatch
from lmdiff.tasks.loglikelihood import loglikelihood_accuracy

DEFAULT_TASKS: tuple[str, ...] = (
    "hellaswag",
    "arc_challenge",
    "gsm8k",
    "mmlu_college_computer_science",
    "longbench_2wikimqa",
)

# Mapping from lm-eval task name to lmdiff domain label. Mirrors
# KNOWN_TASK_DOMAINS in lmdiff/probes/adapters.py for the DEFAULT_TASKS
# subset, but is the canonical source for summary-JSON serialization
# (task-keyed wire format → domain-keyed GeoResult dicts).
TASK_TO_DOMAIN: dict[str, str] = {
    "hellaswag": "commonsense",
    "arc_challenge": "reasoning",
    "gsm8k": "math",
    "mmlu_college_computer_science": "code",
    "longbench_2wikimqa": "long-context",
}

DEFAULT_DOMAIN_ORDER: list[str] = [
    "commonsense",
    "reasoning",
    "math",
    "code",
    "long-context",
]


# Evaluator class for each generate_until task whose accuracy we score.
GENERATE_EVALUATORS: dict[str, type] = {
    "gsm8k": Gsm8kNumberMatch,
    "longbench_2wikimqa": F1,
    "longbench_hotpotqa": F1,
    "longbench_narrativeqa": F1,
    "longbench_qasper": F1,
    "squadv2": F1,
    "triviaqa": F1,
    "nq_open": F1,
}


@dataclass
class FamilyExperimentResult:
    """Bundle of everything a family experiment produces.

    ``geo`` is the raw GeoResult; the per-task aggregates and accuracies
    are derived from it plus the per-task probe partition. Paths point at
    files written to ``output_dir`` (empty dict when ``write_outputs=False``).
    """

    base: str
    variants: dict[str, str]
    tasks: list[str]
    limit_per_task: int
    max_new_tokens: int
    seed: int
    geo: GeoResult
    delta_magnitude_by_variant: dict[str, dict[str, float]]
    delta_magnitude_by_variant_normalized: dict[str, dict[str, float]] = field(
        default_factory=dict,
    )
    delta_specialization_zscore_by_variant: dict[str, dict[str, float]] = field(
        default_factory=dict,
    )
    accuracy_by_variant: dict[str, dict[str, float]] = field(default_factory=dict)
    output_paths: dict[str, Path] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def to_summary_dict(self) -> dict:
        """JSON-serializable summary (drops the heavy GeoResult).

        Inner keys of ``delta_magnitude_by_variant*`` and
        ``delta_specialization_zscore_by_variant`` are lm-eval *task* names
        (e.g. ``hellaswag``) — task→domain reverse-lookup happens in the
        helper functions before serialization.
        """
        return {
            "base": self.base,
            "variants": dict(self.variants),
            "tasks": list(self.tasks),
            "limit_per_task": self.limit_per_task,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "delta_magnitude_by_variant": self.delta_magnitude_by_variant,
            "delta_magnitude_by_variant_normalized": (
                self.delta_magnitude_by_variant_normalized
            ),
            "delta_specialization_zscore_by_variant": (
                self.delta_specialization_zscore_by_variant
            ),
            "accuracy_by_variant": self.accuracy_by_variant,
            "geometry_metadata": self.geo.metadata,
            "magnitudes_total": dict(self.geo.magnitudes),
            "magnitudes_total_normalized": dict(self.geo.magnitudes_normalized),
        }


def _load_concatenated_probes(
    task_names: list[str], limit_per_task: int, seed: int,
) -> tuple[ProbeSet, dict[str, list[Probe]]]:
    per_task: dict[str, list[Probe]] = {}
    all_probes: list[Probe] = []
    for task in task_names:
        ps = from_lm_eval(task, limit=limit_per_task, seed=seed)
        per_task[task] = list(ps)
        all_probes.extend(ps)
    mega = ProbeSet(
        all_probes,
        name=f"lm_eval:{'+'.join(task_names)}",
        version="lm-eval-harness",
    )
    return mega, per_task


def _accuracy_for_task(task_name: str, probes: ProbeSet, engine: Any) -> float:
    """Run the correct evaluator for one task on one engine. Returns acc in [0,1] or NaN."""
    info = KNOWN_TASK_DOMAINS.get(task_name)
    if info is None:
        from lmdiff.tasks.evaluators import ContainsAnswer
        task = Task(task_name, probes, ContainsAnswer(), max_new_tokens=32)
        return task.run(engine).accuracy

    if info.output_type == "multiple_choice":
        return loglikelihood_accuracy(probes, engine, task_name=task_name).accuracy

    if info.output_type == "generate_until":
        if info.requires_execution:
            return float("nan")
        evaluator_cls = GENERATE_EVALUATORS.get(task_name)
        if evaluator_cls is None:
            from lmdiff.tasks.evaluators import ContainsAnswer
            evaluator_cls = ContainsAnswer
        task = Task(task_name, probes, evaluator_cls(), max_new_tokens=64)
        return task.run(engine).accuracy

    return float("nan")


def _task_probeset(
    per_task_probes: dict[str, list[Probe]], task: str, mega: ProbeSet,
) -> ProbeSet:
    return ProbeSet(
        per_task_probes[task], name=f"lm_eval:{task}", version=mega.version,
    )


def _partition_delta_by_task(
    per_probe: dict[str, dict[str, float]],
    per_task_probes: dict[str, list[Probe]],
) -> dict[str, dict[str, list[float]]]:
    """Split per_probe[variant] (keyed by probe text) into per-task lists."""
    task_by_text: dict[str, str] = {}
    for task, probes in per_task_probes.items():
        for p in probes:
            task_by_text[p.text] = task

    out: dict[str, dict[str, list[float]]] = {}
    for variant, probe_deltas in per_probe.items():
        out[variant] = {t: [] for t in per_task_probes}
        for text, delta in probe_deltas.items():
            task = task_by_text.get(text)
            if task is not None:
                out[variant][task].append(delta)
    return out


def _l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def _compute_normalized_delta_by_task(
    geo: GeoResult, task_names: list[str],
) -> dict[str, dict[str, float]]:
    """Per-token-normalized δ magnitude per variant per task.

    GeoResult.magnitudes_per_task_normalized() is keyed by *domain* label.
    The summary-JSON wire format keeps the inner key as the lm-eval *task*
    name for consumer stability, so we reverse-lookup via TASK_TO_DOMAIN.

    Falls back to {} when GeoResult lacks probe_domains or token counts
    (legacy v1/v2/v3 JSON, or list[str] prompts).
    """
    if not geo.probe_domains or not geo.avg_tokens_per_probe:
        return {}
    try:
        per_domain = geo.magnitudes_per_task_normalized()
    except ValueError:
        return {}
    out: dict[str, dict[str, float]] = {}
    for variant, domain_vals in per_domain.items():
        per_task: dict[str, float] = {}
        for task in task_names:
            domain = TASK_TO_DOMAIN.get(task, task)
            per_task[task] = float(domain_vals.get(domain, 0.0))
        out[variant] = per_task
    return out


def _compute_specialization_zscore_by_task(
    geo: GeoResult, task_names: list[str],
) -> dict[str, dict[str, float]]:
    """Per-variant per-task specialization z-score keyed by lm-eval task name.

    Same task→domain mapping as _compute_normalized_delta_by_task, but
    reads from GeoResult.magnitudes_specialization_zscore() (row-wise
    z-score of normalized magnitudes; see L-023).
    """
    if not geo.probe_domains or not geo.avg_tokens_per_probe:
        return {}
    try:
        per_domain = geo.magnitudes_specialization_zscore()
    except ValueError:
        return {}
    out: dict[str, dict[str, float]] = {}
    for variant, domain_zs in per_domain.items():
        per_task: dict[str, float] = {}
        for task in task_names:
            domain = TASK_TO_DOMAIN.get(task, task)
            val = domain_zs.get(domain)
            per_task[task] = float(val) if val is not None else 0.0
        out[variant] = per_task
    return out


def run_family_experiment(
    base: str,
    variants: dict[str, str],
    tasks: list[str] | tuple[str, ...] = DEFAULT_TASKS,
    *,
    limit_per_task: int = 100,
    max_new_tokens: int = 16,
    seed: int = 42,
    dtype: str | None = None,
    skip_accuracy: bool = False,
    output_dir: Path | str | None = None,
    output_prefix: str = "family_geometry_lm_eval",
    write_outputs: bool = True,
    render_radars: bool = True,
    progress: bool = True,
) -> FamilyExperimentResult:
    """Run ChangeGeometry + per-task accuracy on a base × N-variant family.

    Args:
        base: HF id or path for the base model.
        variants: ``{name: model_id}``. Names are used as variant labels in
            radars and JSON output.
        tasks: lm-eval task names to concatenate into one probe set.
        limit_per_task: probes per task (forwarded to ``from_lm_eval``).
        max_new_tokens: generation length for δ computation + generate_until tasks.
        seed: probe-sampling seed for ``from_lm_eval``.
        dtype: forwarded to every ``Config``. ``None`` lets the engine pick.
        skip_accuracy: if True, only Phase A (δ) runs.
        output_dir: where JSON / PNG artifacts land. Required when
            ``write_outputs=True``.
        output_prefix: filename prefix for all outputs.
        write_outputs: if False, skip all disk writes (returned ``output_paths``
            is empty). Useful for in-process callers that want the result object.
        render_radars: if False, skip the matplotlib radar rendering even when
            writing other outputs.
        progress: if True, ``print()`` phase markers and per-task accuracy.

    Returns:
        A ``FamilyExperimentResult`` with the GeoResult, per-task aggregates,
        accuracies, and (when written) output file paths + per-phase timings.
    """
    task_names = [t for t in tasks if t]
    if not task_names:
        raise ValueError("tasks must be a non-empty sequence")
    if not variants:
        raise ValueError("variants must be a non-empty mapping")

    output_dir_path: Path | None = None
    if write_outputs:
        if output_dir is None:
            raise ValueError("output_dir is required when write_outputs=True")
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

    if progress:
        print("=== lm-eval family geometry experiment ===")
        print(f"Base: {base}")
        print(f"Variants ({len(variants)}): {variants}")
        print(f"Tasks ({len(task_names)}): {task_names}")
        print(f"Probes per task: {limit_per_task}")
        print(f"Max new tokens: {max_new_tokens}")

    timings: dict[str, float] = {}

    t0 = time.time()
    mega, per_task_probes = _load_concatenated_probes(
        task_names, limit_per_task, seed,
    )
    timings["load_s"] = time.time() - t0
    if progress:
        print(
            f"\nLoaded {len(mega)} probes across {len(task_names)} tasks "
            f"in {timings['load_s']:.1f}s"
        )
        for t in task_names:
            print(f"  {t}: {len(per_task_probes[t])} probes")

    # Phase A: ChangeGeometry over the full mega probe set.
    if progress:
        print(f"\n=== ChangeGeometry ({len(variants)} variants) ===")
    t0 = time.time()
    base_cfg = Config(model=base, dtype=dtype)
    variant_cfgs = {
        name: Config(model=mid, name=name, dtype=dtype)
        for name, mid in variants.items()
    }
    cg = ChangeGeometry(base=base_cfg, variants=variant_cfgs, prompts=mega)
    geo = cg.analyze(max_new_tokens=max_new_tokens)
    timings["geometry_s"] = time.time() - t0
    if progress:
        print(f"ChangeGeometry done in {timings['geometry_s'] / 60:.1f} min")
        print_geometry(geo)

    per_task_delta = _partition_delta_by_task(geo.per_probe, per_task_probes)
    delta_mag_by_variant: dict[str, dict[str, float]] = {
        variant: {t: _l2_norm(d) for t, d in task_deltas.items()}
        for variant, task_deltas in per_task_delta.items()
    }
    delta_mag_normalized = _compute_normalized_delta_by_task(geo, task_names)
    delta_zscore = _compute_specialization_zscore_by_task(geo, task_names)

    del cg
    gc.collect()
    release_cuda_cache()

    # Phase B: per-variant per-task accuracy.
    accuracy_by_variant: dict[str, dict[str, float]] = {}
    if skip_accuracy:
        if progress:
            print("\nSkipping accuracy phase (skip_accuracy=True).")
    else:
        if progress:
            print(
                f"\n=== Accuracy ({len(variants)} variants x "
                f"{len(task_names)} tasks) ==="
            )
        t0 = time.time()
        from lmdiff.engine import InferenceEngine
        for vname, vcfg in variant_cfgs.items():
            if progress:
                print(f"  loading {vname} ...")
            engine = InferenceEngine(vcfg)
            try:
                accuracy_by_variant[vname] = {}
                for task in task_names:
                    ps_task = _task_probeset(per_task_probes, task, mega)
                    acc = _accuracy_for_task(task, ps_task, engine)
                    accuracy_by_variant[vname][task] = acc
                    if progress:
                        if acc == acc:
                            print(f"    {task}: acc={acc:.3f}")
                        else:
                            print(
                                f"    {task}: n/a (requires_execution or unsupported)"
                            )
            finally:
                del engine
                gc.collect()
                release_cuda_cache()
        timings["accuracy_s"] = time.time() - t0
        if progress:
            print(f"Accuracy done in {timings['accuracy_s'] / 60:.1f} min")

    result = FamilyExperimentResult(
        base=base,
        variants=dict(variants),
        tasks=list(task_names),
        limit_per_task=limit_per_task,
        max_new_tokens=max_new_tokens,
        seed=seed,
        geo=geo,
        delta_magnitude_by_variant=delta_mag_by_variant,
        delta_magnitude_by_variant_normalized=delta_mag_normalized,
        delta_specialization_zscore_by_variant=delta_zscore,
        accuracy_by_variant=accuracy_by_variant,
        timings=timings,
    )

    if not write_outputs:
        return result

    assert output_dir_path is not None  # for type checker

    summary_path = output_dir_path / f"{output_prefix}.json"
    summary_path.write_text(
        json.dumps(result.to_summary_dict(), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    result.output_paths["summary_json"] = summary_path
    if progress:
        print(f"\nJSON report: {summary_path}")

    geo_path = output_dir_path / f"{output_prefix}_georesult.json"
    _write_geo_json(geo, geo_path)
    result.output_paths["georesult_json"] = geo_path
    if progress:
        print(f"GeoResult JSON: {geo_path}")

    if render_radars:
        try:
            from lmdiff.viz.radar import plot_radar
        except ImportError as exc:
            if progress:
                print(f"[WARN] matplotlib not available; skipping radar plots: {exc}")
            return result

        delta_png = output_dir_path / f"{output_prefix}_delta_radar.png"
        plot_radar(
            delta_mag_by_variant,
            axes=task_names,
            title=f"delta-magnitude vs {base}",
            out_path=delta_png,
        )
        result.output_paths["delta_radar_png"] = delta_png
        if progress:
            print(f"delta-magnitude radar: {delta_png}")

        if delta_mag_normalized:
            delta_norm_png = (
                output_dir_path / f"{output_prefix}_delta_radar_normalized.png"
            )
            plot_radar(
                delta_mag_normalized,
                axes=task_names,
                title=f"delta-magnitude (per-token) vs {base}",
                out_path=delta_norm_png,
            )
            result.output_paths["delta_radar_normalized_png"] = delta_norm_png
            if progress:
                print(f"per-token-normalized radar: {delta_norm_png}")

        if accuracy_by_variant:
            acc_for_radar = {
                v: {t: (a if a == a else 0.0) for t, a in d.items()}
                for v, d in accuracy_by_variant.items()
            }
            acc_png = output_dir_path / f"{output_prefix}_accuracy_radar.png"
            plot_radar(
                acc_for_radar,
                axes=task_names,
                title="Accuracy per task",
                out_path=acc_png,
                value_range=(0.0, 1.0),
            )
            result.output_paths["accuracy_radar_png"] = acc_png
            if progress:
                print(f"Accuracy radar: {acc_png}")

    return result


# --------------------------------------------------------------------------- #
# plot_family_geometry: GeoResult JSON → figures                              #
# --------------------------------------------------------------------------- #

def _write_index_html(output_dir: Path, entries: list[tuple[str, str]]) -> Path:
    rows = []
    for title, filename in entries:
        rows.append(
            f'<figure><figcaption>{escape(title)}</figcaption>'
            f'<img src="{escape(filename)}" alt="{escape(title)}" '
            f'style="max-width:100%;"/></figure>'
        )
    html = (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        '<title>lmdiff geometry figures</title>'
        '<style>body{font-family:system-ui,sans-serif;margin:2em;max-width:900px}'
        'figure{margin:2em 0;padding:1em;border:1px solid #ddd;border-radius:8px}'
        'figcaption{font-weight:600;margin-bottom:0.5em;color:#333}'
        '</style></head><body>'
        '<h1>lmdiff — GeoResult figures</h1>'
        + "\n".join(rows)
        + "</body></html>"
    )
    path = output_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def _plot_or_warn(label: str, func, **kwargs) -> Path | None:
    try:
        out = func(**kwargs)
        print(f"  {label}: {out}")
        return out
    except Exception as exc:  # noqa: BLE001 - per-plot isolation
        print(
            f"  [WARN] {label} skipped: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def plot_family_geometry(
    geo_or_path: GeoResult | Path | str,
    output_dir: Path | str,
    *,
    write_index_html: bool = True,
) -> dict[str, Path]:
    """Render the pre-v0.2.3 figure suite from a GeoResult (or its JSON path).

    .. deprecated:: 0.2.3
        Use :func:`lmdiff.viz.family_figures.plot_family_figures` for the
        7-figure paper-grade set (adds specialization z-score, normalized
        magnitude, normalization-effect bars). This function will be removed
        in v0.4.0.

    Args:
        geo_or_path: GeoResult instance or path to a v1/v2/v3/v4 GeoResult JSON.
        output_dir: directory for the PNGs (created if missing).
        write_index_html: if True, also write a static ``index.html`` index.

    Returns:
        ``{label: path}`` for every figure that rendered successfully. Plots
        whose required data is absent (e.g. ``probe_domains`` for the domain
        bar) are skipped with a stderr warning, not raised.
    """
    import warnings as _warnings
    _warnings.warn(
        "lmdiff.experiments.family.plot_family_geometry is deprecated "
        "since v0.2.3; use lmdiff.viz.plot_family_figures for the paper-grade "
        "7-figure set. Will be removed in v0.4.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(geo_or_path, (str, Path)):
        path = Path(geo_or_path)
        if not path.exists():
            raise FileNotFoundError(f"GeoResult JSON not found: {path}")
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        geo: GeoResult = geo_result_from_json_dict(payload)
    else:
        geo = geo_or_path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded GeoResult: {len(geo.variant_names)} variants × {geo.n_probes} probes")
    print(f"Writing to {out_dir}")

    entries: list[tuple[str, str]] = []
    rendered: dict[str, Path] = {}

    # 1. Direction heatmap (always available from v1+)
    from lmdiff.viz.direction_heatmap import plot_direction_heatmap
    out = _plot_or_warn(
        "direction_heatmap",
        plot_direction_heatmap,
        cosine_matrix=geo.cosine_matrix,
        variant_names=geo.variant_names,
        out_path=out_dir / "direction_heatmap.png",
        title=f"Direction similarity: {geo.base_name} vs "
              f"{len(geo.variant_names)} variants",
    )
    if out is not None:
        entries.append(("Direction similarity (cosine)", "direction_heatmap.png"))
        rendered["direction_heatmap"] = Path(out)

    # 2. Selective heatmap (v2+ only)
    if geo.selective_cosine_matrix:
        out = _plot_or_warn(
            "selective_heatmap",
            plot_direction_heatmap,
            cosine_matrix=geo.selective_cosine_matrix,
            variant_names=geo.variant_names,
            out_path=out_dir / "selective_heatmap.png",
            title="Selective cosine (Pearson r, mean-removed)",
        )
        if out is not None:
            entries.append(("Selective cosine (Pearson r)", "selective_heatmap.png"))
            rendered["selective_heatmap"] = Path(out)
    else:
        print(
            "  [WARN] selective_heatmap skipped: selective_cosine_matrix empty "
            "(legacy v1 JSON)",
            file=sys.stderr,
        )

    # 3. PCA scatter (needs n_variants >= 2)
    if len(geo.variant_names) < 2:
        print("  [WARN] pca_scatter skipped: n_variants < 2", file=sys.stderr)
    else:
        try:
            pca_result = geo.pca_map(n_components=2)
        except ValueError as exc:
            print(
                f"  [WARN] pca_scatter skipped: pca_map failed: {exc}",
                file=sys.stderr,
            )
        else:
            from lmdiff.viz.pca_scatter import plot_pca_scatter
            out = _plot_or_warn(
                "pca_scatter",
                plot_pca_scatter,
                pca_result=pca_result,
                out_path=out_dir / "pca_scatter.png",
                title=f"Change geometry (PCA) — base: {geo.base_name}",
            )
            if out is not None:
                entries.append(("PCA scatter", "pca_scatter.png"))
                rendered["pca_scatter"] = Path(out)

    # 4. Domain bar — prefer per-token-normalized (v4) when available,
    # fall back to raw (v3). Skip entirely without probe_domains.
    if not geo.probe_domains:
        print(
            "  [WARN] domain_bar skipped: probe_domains empty "
            "(legacy v1/v2 JSON or list[str] prompts)",
            file=sys.stderr,
        )
    else:
        heatmap = None
        domain_bar_title = f"Per-domain δ magnitude — base: {geo.base_name}"
        normalized = False
        if geo.avg_tokens_per_probe:
            try:
                heatmap = geo.magnitudes_per_task_normalized()
                normalized = True
                domain_bar_title = (
                    f"Per-domain δ magnitude (per-token normalized) "
                    f"— base: {geo.base_name}"
                )
            except ValueError as exc:
                print(
                    f"  [WARN] normalized domain_bar failed, falling back "
                    f"to raw: {exc}",
                    file=sys.stderr,
                )
        if heatmap is None:
            try:
                heatmap = geo.domain_heatmap()
            except ValueError as exc:
                print(
                    f"  [WARN] domain_bar skipped: domain_heatmap failed: {exc}",
                    file=sys.stderr,
                )

        if heatmap is not None:
            from lmdiff.viz.domain_bar import plot_domain_bar
            out = _plot_or_warn(
                "domain_bar",
                plot_domain_bar,
                domain_heatmap=heatmap,
                out_path=out_dir / "domain_bar.png",
                title=domain_bar_title,
            )
            if out is not None:
                label = (
                    "Per-domain δ magnitude (per-token normalized)"
                    if normalized
                    else "Per-domain δ magnitude (raw)"
                )
                entries.append((label, "domain_bar.png"))
                rendered["domain_bar"] = Path(out)

    if write_index_html and entries:
        idx = _write_index_html(out_dir, entries)
        rendered["index_html"] = idx
        print(f"\nIndex: {idx}")
        print(f"Figures rendered: {len(entries)}")
    elif not entries:
        print("\n[WARN] no figures rendered", file=sys.stderr)

    return rendered


__all__ = [
    "DEFAULT_TASKS",
    "TASK_TO_DOMAIN",
    "DEFAULT_DOMAIN_ORDER",
    "GENERATE_EVALUATORS",
    "FamilyExperimentResult",
    "run_family_experiment",
    "plot_family_geometry",
]
