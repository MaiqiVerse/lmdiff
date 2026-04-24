"""lmdiff CLI — compare language model configurations from the terminal.

All heavy imports (transformers, torch, engine) happen inside command
functions so that ``lmdiff --help`` and ``lmdiff list-metrics``
remain fast.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="lmdiff",
    help="Compare language model configurations via behavioral distance and multi-level diagnostics.",
    add_completion=False,
)

_BUILTIN_PROBES_DIR = Path(__file__).parent / "probes"

_EVALUATOR_MAP = {
    "exact_match": "ExactMatch",
    "contains_answer": "ContainsAnswer",
    "multiple_choice": "MultipleChoice",
}


def _resolve_probes_path(probes: str) -> Path:
    """Resolve a probe specifier to a file path.

    Accepts: "v01", "v01.json", or a full/relative file path.
    """
    # Try as builtin name first
    builtin = _BUILTIN_PROBES_DIR / f"{probes}.json"
    if builtin.exists():
        return builtin
    builtin_raw = _BUILTIN_PROBES_DIR / probes
    if builtin_raw.exists():
        return builtin_raw
    # Try as literal path
    p = Path(probes)
    if p.exists():
        return p
    raise typer.BadParameter(
        f"Probe set not found: '{probes}'. "
        f"Tried builtin ({builtin}) and path ({p})."
    )


def _get_evaluator(name: str):
    """Lazily import and return an evaluator instance."""
    from lmdiff.tasks.evaluators import ContainsAnswer, ExactMatch, MultipleChoice

    mapping = {
        "exact_match": ExactMatch,
        "contains_answer": ContainsAnswer,
        "multiple_choice": MultipleChoice,
    }
    cls = mapping.get(name)
    if cls is None:
        raise typer.BadParameter(
            f"Unknown evaluator: '{name}'. Choose from: {', '.join(mapping)}"
        )
    return cls()


@app.command()
def compare(
    model_a: str = typer.Argument(..., help="HuggingFace model ID or path for config A"),
    model_b: str = typer.Argument(..., help="HuggingFace model ID or path for config B"),
    probes: str = typer.Option("v01", help="Probe set name or path (default: v01)"),
    level: str = typer.Option("output", help="Metric level to run"),
    output_json: bool = typer.Option(False, "--json", help="Output JSON instead of rich table"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write output to file"),
    n_samples: int = typer.Option(5, help="Number of generation samples"),
    max_new_tokens: int = typer.Option(64, help="Max new tokens for generation"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model precision: bfloat16, float16, float32 (default: auto)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all per-probe details"),
) -> None:
    """Run metric-level comparison between two model configurations."""
    if level != "output":
        raise typer.BadParameter(f"Level '{level}' not implemented in Phase 1. Use 'output'.")

    probes_path = _resolve_probes_path(probes)

    from lmdiff.config import Config
    from lmdiff.diff import ModelDiff
    from lmdiff.probes.loader import ProbeSet

    ps = ProbeSet.from_json(probes_path)
    md = ModelDiff(
        Config(model=model_a, dtype=dtype),
        Config(model=model_b, dtype=dtype),
        ps,
        n_samples=n_samples,
    )
    report = md.run(level=level, max_new_tokens=max_new_tokens)

    if output_json:
        from lmdiff.report.json_report import to_json, write_json

        if output:
            write_json(report, output)
            Console().print(f"Written to {output}")
        else:
            typer.echo(to_json(report))
    else:
        from lmdiff.report.terminal import print_report

        print_report(report, verbose=verbose)


@app.command()
def radar(
    model_a: str = typer.Argument(..., help="HuggingFace model ID or path for config A"),
    model_b: str = typer.Argument(..., help="HuggingFace model ID or path for config B"),
    probes: str = typer.Option("v01", help="Probe set name or path (default: v01)"),
    evaluator: str = typer.Option("contains_answer", help="Evaluator: exact_match, contains_answer, multiple_choice"),
    max_new_tokens: int = typer.Option(16, help="Max new tokens for generation"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model precision: bfloat16, float16, float32 (default: auto)"),
    output_json: bool = typer.Option(False, "--json", help="Output JSON instead of rich table"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write output to file"),
) -> None:
    """Run per-domain capability radar (accuracy + BD) on two models."""
    probes_path = _resolve_probes_path(probes)
    eval_instance = _get_evaluator(evaluator)

    from lmdiff.config import Config
    from lmdiff.diff import ModelDiff
    from lmdiff.probes.loader import ProbeSet

    ps = ProbeSet.from_json(probes_path)
    md = ModelDiff(
        Config(model=model_a, dtype=dtype),
        Config(model=model_b, dtype=dtype),
        ps,
    )
    result = md.run_radar(probes=ps, evaluator=eval_instance, max_new_tokens=max_new_tokens)

    if output_json:
        from lmdiff.report.json_report import to_json, write_json

        if output:
            write_json(result, output)
            Console().print(f"Written to {output}")
        else:
            typer.echo(to_json(result))
    else:
        from lmdiff.report.terminal import print_radar

        print_radar(result)


def _parse_variant_spec(spec: str) -> tuple[str, str]:
    """Split a 'name=model_id' variant spec into (name, model_id).

    Rejects specs missing '=', or with blank left/right, via typer.BadParameter.
    """
    if "=" not in spec:
        raise typer.BadParameter(
            f"Variant '{spec}' must be in 'name=model_id' format"
        )
    name, model_id = spec.split("=", 1)
    name = name.strip()
    model_id = model_id.strip()
    if not name:
        raise typer.BadParameter(f"Variant spec '{spec}' has an empty name")
    if not model_id:
        raise typer.BadParameter(f"Variant spec '{spec}' has an empty model id")
    return name, model_id


@app.command()
def geometry(
    base: str = typer.Argument(..., help="Base model HF id or path"),
    variants: list[str] = typer.Argument(
        ...,
        help=(
            "Variant specs in 'name=model_id' format, "
            "e.g. '13b=meta-llama/Llama-2-13b-hf'"
        ),
    ),
    probes: str = typer.Option("v01", help="Probe set name or path (default: v01)"),
    max_new_tokens: int = typer.Option(16, help="Max new tokens for generation"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model precision: bfloat16, float16, float32 (default: auto)"),
    output_json: bool = typer.Option(False, "--json", help="Output JSON instead of rich table"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write output to file"),
) -> None:
    """Analyze change geometry: one base vs multiple named variants."""
    parsed: dict[str, str] = {}
    for spec in variants:
        name, model_id = _parse_variant_spec(spec)
        if name in parsed:
            raise typer.BadParameter(f"Duplicate variant name: '{name}'")
        parsed[name] = model_id

    probes_path = _resolve_probes_path(probes)

    from lmdiff.config import Config
    from lmdiff.geometry import ChangeGeometry
    from lmdiff.probes.loader import ProbeSet

    ps = ProbeSet.from_json(probes_path)
    variant_configs = {
        name: Config(model=model_id, dtype=dtype, name=name)
        for name, model_id in parsed.items()
    }
    cg = ChangeGeometry(
        base=Config(model=base, dtype=dtype),
        variants=variant_configs,
        prompts=ps,
    )
    result = cg.analyze(max_new_tokens=max_new_tokens)

    if output_json:
        from lmdiff.report.json_report import to_json, write_json

        if output:
            write_json(result, output)
            Console().print(f"Written to {output}")
        else:
            typer.echo(to_json(result))
    else:
        from lmdiff.report.terminal import print_geometry

        print_geometry(result)


@app.command(name="run-task")
def run_task(
    model: str = typer.Argument(..., help="HuggingFace model ID or path"),
    probes: str = typer.Option("v01", help="Probe set name or path (default: v01)"),
    evaluator: str = typer.Option("contains_answer", help="Evaluator: exact_match, contains_answer, multiple_choice"),
    max_new_tokens: int = typer.Option(16, help="Max new tokens for generation"),
    output_json: bool = typer.Option(False, "--json", help="Output JSON instead of rich table"),
) -> None:
    """Run task evaluation on a single model (no diff)."""
    probes_path = _resolve_probes_path(probes)
    eval_instance = _get_evaluator(evaluator)

    from lmdiff.config import Config
    from lmdiff.engine import InferenceEngine
    from lmdiff.probes.loader import ProbeSet
    from lmdiff.tasks.base import Task

    ps = ProbeSet.from_json(probes_path)
    engine = InferenceEngine(Config(model=model))
    task = Task(name="eval", probes=ps, evaluator=eval_instance, max_new_tokens=max_new_tokens)
    result = task.run(engine)

    if output_json:
        from lmdiff.report.json_report import to_json

        typer.echo(to_json(result))
    else:
        console = Console()
        console.rule(f"[bold]Task: {result.task_name} on {result.engine_name}[/bold]")
        console.print(f"Probes: {result.n_probes}  Correct: {result.n_correct}  Accuracy: {result.accuracy:.2%}")
        console.print()
        if result.per_domain:
            tbl = Table(title="Per-Domain", show_lines=True)
            tbl.add_column("Domain", style="cyan")
            tbl.add_column("N", justify="right")
            tbl.add_column("Correct", justify="right")
            tbl.add_column("Accuracy", justify="right")
            for d, info in sorted(result.per_domain.items()):
                tbl.add_row(d, str(info["n"]), str(info["correct"]), f"{info['accuracy']:.2%}")
            console.print(tbl)


@app.command(name="family-experiment")
def family_experiment(
    base: str = typer.Option(..., "--base", help="HuggingFace id or path for the base model"),
    variant: list[str] = typer.Option(
        ...,
        "--variant",
        help=(
            "Repeatable. 'name=model_id' for one variant, "
            "e.g. --variant yarn=NousResearch/Yarn-Llama-2-7b-128k. "
            "Pass --variant once per variant."
        ),
    ),
    tasks: Optional[str] = typer.Option(
        None,
        "--tasks",
        help="Comma-separated lm-eval task names (default: lmdiff.experiments DEFAULT_TASKS).",
    ),
    limit_per_task: int = typer.Option(100, "--limit-per-task", help="Probes per task."),
    max_new_tokens: int = typer.Option(16, "--max-new-tokens"),
    seed: int = typer.Option(42, "--seed"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model precision: bfloat16, float16, float32"),
    skip_accuracy: bool = typer.Option(
        False, "--skip-accuracy", help="Only compute delta-magnitude radar (skip accuracy runs).",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        help="Directory for JSON / radar PNG artifacts (created if missing).",
    ),
    output_prefix: str = typer.Option(
        "family_geometry_lm_eval",
        "--output-prefix",
        help="Filename prefix for all outputs.",
    ),
    no_radars: bool = typer.Option(
        False, "--no-radars", help="Skip matplotlib radar rendering.",
    ),
) -> None:
    """Run a base x N-variant family geometry + accuracy experiment over lm-eval tasks."""
    parsed: dict[str, str] = {}
    for spec in variant:
        name, model_id = _parse_variant_spec(spec)
        if name in parsed:
            raise typer.BadParameter(f"Duplicate variant name: '{name}'")
        parsed[name] = model_id

    from lmdiff.experiments.family import DEFAULT_TASKS, run_family_experiment

    if tasks is None:
        task_list = list(DEFAULT_TASKS)
    else:
        task_list = [t.strip() for t in tasks.split(",") if t.strip()]
        if not task_list:
            raise typer.BadParameter("--tasks must contain at least one task name")

    run_family_experiment(
        base=base,
        variants=parsed,
        tasks=task_list,
        limit_per_task=limit_per_task,
        max_new_tokens=max_new_tokens,
        seed=seed,
        dtype=dtype,
        skip_accuracy=skip_accuracy,
        output_dir=output_dir,
        output_prefix=output_prefix,
        write_outputs=True,
        render_radars=not no_radars,
        progress=True,
    )


@app.command(name="plot-geometry")
def plot_geometry(
    georesult: Path = typer.Argument(..., help="GeoResult JSON path (v1/v2/v3/v4)."),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Directory to write PNGs (created if missing).",
    ),
    figures: str = typer.Option(
        "all",
        "--figures",
        help=(
            "CSV of figure keys to render. Default 'all' = full 7-figure "
            "paper-grade set. Keys: cosine_raw, cosine_selective, "
            "normalized_magnitude, specialization, pca_raw, pca_normalized, "
            "normalization_effect."
        ),
    ),
    variant_order: Optional[str] = typer.Option(
        None,
        "--variant-order",
        help="CSV row/legend order for variants (default: as found in the GeoResult).",
    ),
    domain_order: Optional[str] = typer.Option(
        None,
        "--domain-order",
        help="CSV column order for domain heatmaps (default: DEFAULT_DOMAIN_ORDER).",
    ),
    dpi: int = typer.Option(
        200, "--dpi", help="DPI for all rendered PNGs (default: 200, paper-grade).",
    ),
) -> None:
    """Render the v0.2.3 paper-grade family-figure set from a GeoResult JSON."""
    if not georesult.exists():
        raise typer.BadParameter(f"GeoResult JSON not found: {georesult}")

    from lmdiff.viz.family_figures import FIGURE_REGISTRY, plot_family_figures

    if figures.strip().lower() == "all":
        which: Optional[list[str]] = None
    else:
        which = [k.strip() for k in figures.split(",") if k.strip()]
        unknown = [k for k in which if k not in FIGURE_REGISTRY]
        if unknown:
            raise typer.BadParameter(
                f"Unknown figure key(s): {unknown}. "
                f"Valid keys: {sorted(FIGURE_REGISTRY)}"
            )

    v_order = (
        [s.strip() for s in variant_order.split(",") if s.strip()]
        if variant_order else None
    )
    d_order = (
        [s.strip() for s in domain_order.split(",") if s.strip()]
        if domain_order else None
    )

    rendered = plot_family_figures(
        georesult,
        output_dir,
        which=which,
        variant_order=v_order,
        domain_order=d_order,
        dpi=dpi,
    )
    if not rendered:
        raise typer.Exit(code=1)
    Console().print(
        f"Rendered {len(rendered)} figure(s) to {output_dir}"
    )


@app.command(name="list-metrics")
def list_metrics(
    level: Optional[str] = typer.Option(None, help="Filter by metric level"),
) -> None:
    """List available metrics and their properties."""
    from lmdiff.metrics.base import MetricLevel
    from lmdiff.metrics.output.behavioral_distance import BehavioralDistance
    from lmdiff.metrics.output.token_entropy import TokenEntropy
    from lmdiff.metrics.output.token_kl import TokenKL

    all_metrics = [BehavioralDistance, TokenEntropy, TokenKL]

    if level:
        try:
            target_level = MetricLevel(level)
        except ValueError:
            valid = ", ".join(lv.value for lv in MetricLevel)
            raise typer.BadParameter(f"Unknown level '{level}'. Choose from: {valid}")
        all_metrics = [m for m in all_metrics if m.level == target_level]

    console = Console()
    tbl = Table(title="Available Metrics", show_lines=True)
    tbl.add_column("Name", style="cyan", min_width=25)
    tbl.add_column("Level", min_width=15)
    tbl.add_column("Requirements", min_width=30)

    for cls in all_metrics:
        reqs = cls.requirements()
        req_str = ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in sorted(reqs.items()))
        tbl.add_row(cls.name, cls.level.value, req_str)

    console.print(tbl)


if __name__ == "__main__":
    app()
