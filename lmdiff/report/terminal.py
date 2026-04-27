from __future__ import annotations

import math
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from lmdiff.diff import DiffReport
    from lmdiff.geometry import GeoResult
    from lmdiff.tasks.capability_radar import RadarResult


def print_report(
    report: DiffReport,
    console: Console | None = None,
    verbose: bool = False,
) -> None:
    """Print a DiffReport as a rich table to the terminal."""
    console = console or Console()

    name_a = report.metadata.get("name_a", "A")
    name_b = report.metadata.get("name_b", "B")

    console.print()
    console.rule(f"[bold]lmdiff: {name_a} vs {name_b}[/bold]")
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
    if bd and bd.details:
        _print_degeneracy_warning(bd, name_a, name_b, console)
        if bd.details.get("per_prompt"):
            console.print()
            max_rows = None if verbose else 10
            _print_bd_breakdown(bd, name_a, name_b, console, max_rows=max_rows)

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


def _print_degeneracy_warning(bd, name_a: str, name_b: str, console: Console) -> None:
    d = bd.details
    rate_a = d.get("degeneracy_rate_a", 0.0)
    rate_b = d.get("degeneracy_rate_b", 0.0)
    if rate_a <= 0.10 and rate_b <= 0.10:
        return
    console.print()
    bd_healthy = d.get("bd_healthy")
    n_healthy = d.get("n_healthy", 0)
    if bd_healthy is not None and n_healthy >= 3:
        console.print(
            f"[bold yellow]Warning:[/bold yellow] Degeneracy: "
            f"{name_a}={rate_a:.0%}, {name_b}={rate_b:.0%}. "
            f"Healthy BD = {bd_healthy:+.4f} (n={n_healthy})"
        )
    else:
        console.print(
            f"[bold yellow]Warning:[/bold yellow] Degeneracy: "
            f"{name_a}={rate_a:.0%}, {name_b}={rate_b:.0%}. "
            f"Too few healthy probes (n={n_healthy}) for bd_healthy."
        )


def _print_bd_breakdown(
    bd, name_a: str, name_b: str, console: Console, max_rows: int | None = 10,
) -> None:
    import math as _math

    console.print(
        "[dim]CE(X,Y) = engine X scores engine Y's output; "
        "per-token cross-entropy (lower = better fit).[/dim]"
    )
    tbl = Table(title="Behavioral Distance per Probe", show_lines=True)
    tbl.add_column("Probe", style="white", max_width=40)
    tbl.add_column(f"CE({name_a},{name_a})", justify="right")
    tbl.add_column(f"CE({name_b},{name_a})", justify="right")
    tbl.add_column(f"CE({name_a},{name_b})", justify="right")
    tbl.add_column(f"CE({name_b},{name_b})", justify="right")
    tbl.add_column("BD", justify="right", style="bold")
    tbl.add_column("Asym", justify="right")

    all_rows = bd.details["per_prompt"]

    if max_rows is not None and len(all_rows) > max_rows:
        half = max_rows // 2

        def _sort_key(pp: dict) -> float:
            v = pp.get("bd", 0.0)
            return v if not _math.isnan(v) else float("-inf")

        sorted_rows = sorted(all_rows, key=_sort_key, reverse=True)
        top = sorted_rows[:half]
        bottom = sorted_rows[-half:]
        n_hidden = len(all_rows) - max_rows
        display = top + [None] + bottom  # None = separator
    else:
        display = all_rows

    for pp in display:
        if pp is None:
            tbl.add_row(
                f"... {n_hidden} more (--verbose)",
                "", "", "", "", "", "",
            )
            continue
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


def print_radar(result: RadarResult, console: Console | None = None) -> None:
    """Print a RadarResult as a rich table to the terminal."""
    console = console or Console()

    pair_mode = result.b_by_domain is not None
    name_a = result.engine_a_name
    name_b = result.engine_b_name or ""

    if pair_mode:
        console.rule(f"[bold]Capability Radar: {name_a} vs {name_b}[/bold]")
    else:
        console.rule(f"[bold]Capability Radar: {name_a}[/bold]")
    console.print()

    tbl = Table(title="Per-Domain Results", show_lines=True)
    tbl.add_column("Domain", style="cyan", min_width=12)
    tbl.add_column("N", justify="right", min_width=4)
    tbl.add_column(f"Acc({name_a})", justify="right", min_width=10)

    if pair_mode:
        tbl.add_column(f"Acc({name_b})", justify="right", min_width=10)
        tbl.add_column("\u0394Acc", justify="right", min_width=8)
        tbl.add_column("BD", justify="right", min_width=8, style="bold")
        tbl.add_column("BD(healthy)", justify="right", min_width=10)
        tbl.add_column(f"Degen({name_a})", justify="right", min_width=10)
        tbl.add_column(f"Degen({name_b})", justify="right", min_width=10)

    rows = result.summary_table()
    total_n = 0
    total_correct_a = 0
    total_correct_b = 0

    for row in rows:
        n = row["n_probes"]
        total_n += n
        total_correct_a += round(row["accuracy_a"] * n)
        cells = [
            row["domain"],
            str(n),
            f"{row['accuracy_a']:.2%}",
        ]
        if pair_mode:
            total_correct_b += round(row["accuracy_b"] * n)
            delta = row.get("delta_acc", 0.0)
            bd = row.get("bd", 0.0)
            bd_h = row.get("bd_healthy")
            cells += [
                f"{row['accuracy_b']:.2%}",
                f"{delta:+.2%}",
                f"{bd:.4f}",
                f"{bd_h:.4f}" if bd_h is not None else "n/a",
                f"{row.get('degen_a', 0.0):.0%}",
                f"{row.get('degen_b', 0.0):.0%}",
            ]
        tbl.add_row(*cells)

    console.print(tbl)

    # Overall summary line
    overall_acc_a = total_correct_a / total_n if total_n else 0.0
    parts = [f"Overall: {name_a} acc={overall_acc_a:.2%}"]
    if pair_mode and total_n:
        overall_acc_b = total_correct_b / total_n
        parts.append(f"{name_b} acc={overall_acc_b:.2%}")
        if result.bd_by_domain:
            all_bd = list(result.bd_by_domain.values())
            mean_bd = sum(all_bd) / len(all_bd)
            parts.append(f"mean BD={mean_bd:.4f}")
    console.print("  ".join(parts))
    console.print()


def print_geometry(result: "GeoResult", console: Console | None = None) -> None:
    """Print ChangeGeometry analysis: magnitude table + original cosine matrix,
    plus selective cosine matrix and const_frac column when the decomposition
    fields are populated (schema v2 or a fresh analyze() result).
    """
    console = console or Console()

    has_decomp = bool(result.selective_cosine_matrix) and bool(result.selective_magnitudes)
    const_fractions = result.constant_fractions if has_decomp else {}

    console.print()
    console.rule(f"[bold]Change Geometry: {result.base_name} vs {len(result.variant_names)} variants[/bold]")
    console.print()

    # Magnitude ranking (optionally with const_frac column when decomp available)
    mag_table = Table(title="Magnitude ranking", show_lines=False)
    mag_table.add_column("Rank", justify="right", style="dim", min_width=4)
    mag_table.add_column("Variant", style="cyan", min_width=10)
    mag_table.add_column("‖δ‖", justify="right", min_width=10)
    if has_decomp:
        mag_table.add_column("const_frac", justify="right", min_width=10)

    ranked = sorted(result.variant_names, key=lambda v: -result.magnitudes[v])
    for i, name in enumerate(ranked, start=1):
        row = [str(i), name, f"{result.magnitudes[name]:.4f}"]
        if has_decomp:
            cf = const_fractions.get(name, float("nan"))
            row.append("n/a" if math.isnan(cf) else f"{cf:.3f}")
        mag_table.add_row(*row)
    console.print(mag_table)

    console.print()

    # Cosine similarity matrix
    cos_table = Table(title="Cosine similarity matrix", show_lines=False)
    cos_table.add_column("", style="cyan")
    for name in result.variant_names:
        cos_table.add_column(name, justify="right", min_width=8)

    for a in result.variant_names:
        row: list[str] = [a]
        for b in result.variant_names:
            cos = result.cosine_matrix[a][b]
            row.append(_format_cosine(cos, diagonal=(a == b)))
        cos_table.add_row(*row)
    console.print(cos_table)

    # Selective cosine matrix (only when decomposition fields present)
    if has_decomp:
        console.print()
        sel_table = Table(title="Selective cosine matrix (Pearson r)", show_lines=False)
        sel_table.add_column("", style="cyan")
        for name in result.variant_names:
            sel_table.add_column(name, justify="right", min_width=8)

        for a in result.variant_names:
            row = [a]
            for b in result.variant_names:
                cos = result.selective_cosine_matrix[a][b]
                row.append(_format_cosine(cos, diagonal=(a == b)))
            sel_table.add_row(*row)
        console.print(sel_table)

    # Footer
    footer_parts: list[str] = [f"n_probes={result.n_probes}"]
    n_skipped = result.metadata.get("n_skipped", 0)
    if n_skipped:
        footer_parts.append(f"(skipped {n_skipped} with NaN CE)")
    bpb_map = result.metadata.get("bpb_normalized", {}) or {}
    bpb_variants = sorted(name for name, flag in bpb_map.items() if flag)
    if bpb_variants:
        footer_parts.append(f"BPB-normalized: {', '.join(bpb_variants)}")
    if has_decomp:
        sel_summary = " ".join(
            f"{n}={result.selective_magnitudes[n]:.2f}" for n in result.variant_names
        )
        footer_parts.append(f"selective: {sel_summary}")
    console.print("  ".join(footer_parts))
    console.print()


def _format_cosine(value: float, diagonal: bool) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if diagonal:
        return f"[dim]{value:.2f}[/dim]"
    return f"{value:+.3f}"


# ── v0.3.0 Renderer Protocol adapter (commit 1.5) ─────────────────────


def render(
    result: "GeoResult",
    *,
    findings: tuple = (),  # noqa: ARG001  reserved for commit 1.7+
    tables: dict | None = None,  # noqa: ARG001
    file: object = None,
    **_unused,
) -> str:
    """Render a GeoResult to the terminal and also return its plain-text form.

    v0.3.0 skeleton: delegates to the v0.2.x ``print_geometry`` Rich
    renderer for now and prepends a one-line banner pointing at commit
    1.7. The 5-layer application-tier terminal output (Layer 1 one-liner,
    Layer 2 headlines, ...) lands in commit 1.7 without restructuring.

    When ``file`` is given (anything with a ``write`` method), the Rich
    output goes there; otherwise it goes to ``sys.stdout``. The returned
    string is a minimal text dump of the result (used by callers that
    want the content as a string).
    """
    console = Console(file=file) if file is not None else Console()
    console.print(
        "[dim]([italic]v0.3.0 application-tier terminal renderer arrives "
        "in commit 1.7[/italic])[/dim]"
    )
    print_geometry(result, console=console)
    # Plain-text return for in-process callers that want a string handle.
    return (
        f"GeoResult({result.base_name} vs "
        f"{', '.join(result.variant_names)}; "
        f"n_probes={result.n_probes})"
    )
