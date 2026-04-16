from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from modeldiff.diff import DiffReport


def print_report(report: DiffReport, console: Console | None = None) -> None:
    """Print a DiffReport as a rich table to the terminal."""
    console = console or Console()

    name_a = report.metadata.get("name_a", "A")
    name_b = report.metadata.get("name_b", "B")

    console.print()
    console.rule(f"[bold]ModelDiff: {name_a} vs {name_b}[/bold]")
    console.print()

    summary = Table(title="Metric Summary", show_lines=True)
    summary.add_column("Metric", style="cyan", min_width=25)
    summary.add_column("Value", justify="right", min_width=12)
    summary.add_column("Details", min_width=40)

    for r in report.results:
        value_str = _format_value(r.value)
        detail_str = _format_details(r)
        summary.add_row(r.name, value_str, detail_str)

    console.print(summary)

    bd = report.get("behavioral_distance")
    if bd and bd.details and bd.details.get("per_prompt"):
        console.print()
        _print_bd_breakdown(bd, name_a, name_b, console)

    console.print()


def _format_value(value: float | dict | object) -> str:
    if isinstance(value, float):
        return f"{value:+.4f}" if value != 0 else "0.0000"
    return str(value)


def _format_details(r) -> str:
    if r.details is None:
        return ""
    parts: list[str] = []
    skip = {"per_prompt"}
    for k, v in r.details.items():
        if k in skip:
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        elif isinstance(v, bool):
            parts.append(f"{k}={'yes' if v else 'no'}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _print_bd_breakdown(bd, name_a: str, name_b: str, console: Console) -> None:
    tbl = Table(title="Behavioral Distance per Probe", show_lines=True)
    tbl.add_column("Probe", style="white", max_width=40)
    tbl.add_column(f"CE({name_a},{name_a})", justify="right")
    tbl.add_column(f"CE({name_b},{name_a})", justify="right")
    tbl.add_column(f"CE({name_a},{name_b})", justify="right")
    tbl.add_column(f"CE({name_b},{name_b})", justify="right")
    tbl.add_column("BD", justify="right", style="bold")
    tbl.add_column("Asym", justify="right")

    for pp in bd.details["per_prompt"]:
        tbl.add_row(
            pp["probe"][:40],
            f"{pp['ce_aa']:.3f}",
            f"{pp['ce_ab']:.3f}",
            f"{pp['ce_ba']:.3f}",
            f"{pp['ce_bb']:.3f}",
            f"{pp['bd']:.4f}",
            f"{pp['asymmetry']:+.4f}",
        )

    console.print(tbl)
