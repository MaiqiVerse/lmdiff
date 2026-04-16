"""ModelDiff CLI — compare language model configurations from the terminal.

All heavy imports (transformers, torch, engine) happen inside command
functions so that ``modeldiff --help`` and ``modeldiff list-metrics``
remain fast.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="modeldiff",
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
    from modeldiff.tasks.evaluators import ContainsAnswer, ExactMatch, MultipleChoice

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
) -> None:
    """Run metric-level comparison between two model configurations."""
    if level != "output":
        raise typer.BadParameter(f"Level '{level}' not implemented in Phase 1. Use 'output'.")

    probes_path = _resolve_probes_path(probes)

    from modeldiff.config import Config
    from modeldiff.diff import ModelDiff
    from modeldiff.probes.loader import ProbeSet

    ps = ProbeSet.from_json(probes_path)
    md = ModelDiff(Config(model=model_a), Config(model=model_b), ps, n_samples=n_samples)
    report = md.run(level=level, max_new_tokens=max_new_tokens)

    if output_json:
        from modeldiff.report.json_report import to_json, write_json

        if output:
            write_json(report, output)
            Console().print(f"Written to {output}")
        else:
            typer.echo(to_json(report))
    else:
        from modeldiff.report.terminal import print_report

        print_report(report)


@app.command()
def radar(
    model_a: str = typer.Argument(..., help="HuggingFace model ID or path for config A"),
    model_b: str = typer.Argument(..., help="HuggingFace model ID or path for config B"),
    probes: str = typer.Option("v01", help="Probe set name or path (default: v01)"),
    evaluator: str = typer.Option("contains_answer", help="Evaluator: exact_match, contains_answer, multiple_choice"),
    max_new_tokens: int = typer.Option(16, help="Max new tokens for generation"),
    output_json: bool = typer.Option(False, "--json", help="Output JSON instead of rich table"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write output to file"),
) -> None:
    """Run per-domain capability radar (accuracy + BD) on two models."""
    probes_path = _resolve_probes_path(probes)
    eval_instance = _get_evaluator(evaluator)

    from modeldiff.config import Config
    from modeldiff.diff import ModelDiff
    from modeldiff.probes.loader import ProbeSet

    ps = ProbeSet.from_json(probes_path)
    md = ModelDiff(Config(model=model_a), Config(model=model_b), ps)
    result = md.run_radar(probes=ps, evaluator=eval_instance, max_new_tokens=max_new_tokens)

    if output_json:
        from modeldiff.report.json_report import to_json, write_json

        if output:
            write_json(result, output)
            Console().print(f"Written to {output}")
        else:
            typer.echo(to_json(result))
    else:
        from modeldiff.report.terminal import print_radar

        print_radar(result)


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

    from modeldiff.config import Config
    from modeldiff.engine import InferenceEngine
    from modeldiff.probes.loader import ProbeSet
    from modeldiff.tasks.base import Task

    ps = ProbeSet.from_json(probes_path)
    engine = InferenceEngine(Config(model=model))
    task = Task(name="eval", probes=ps, evaluator=eval_instance, max_new_tokens=max_new_tokens)
    result = task.run(engine)

    if output_json:
        from modeldiff.report.json_report import to_json

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


@app.command(name="list-metrics")
def list_metrics(
    level: Optional[str] = typer.Option(None, help="Filter by metric level"),
) -> None:
    """List available metrics and their properties."""
    from modeldiff.metrics.base import MetricLevel
    from modeldiff.metrics.output.behavioral_distance import BehavioralDistance
    from modeldiff.metrics.output.token_entropy import TokenEntropy
    from modeldiff.metrics.output.token_kl import TokenKL

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
