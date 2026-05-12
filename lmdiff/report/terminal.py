"""Terminal renderer (commit 1.7).

Two pieces live in this module:

  1. The pre-v0.3.0 Rich helpers (``print_report``, ``print_radar``,
     ``print_geometry``) — kept verbatim for v0.2.x backward compat.
  2. The v0.3.0 ``render(result, **kwargs)`` entry point — the real
     5-layer renderer per v6 plan §12.2 + §13. Replaces the v0.3.0-rc
     stub that delegated to ``print_geometry``.

The 5-layer renderer is plain stdout + ANSI; it does NOT use Rich. Rich
formatting is great for v0.2.x interactive sessions but its escape
handling and width detection don't compose cleanly with the strict
Layer-1..Layer-5 spec. The v0.3.0 entry point writes characters
directly so it can be byte-asserted in tests.
"""
from __future__ import annotations

import io
import math
import os
import shutil
import sys
from typing import TYPE_CHECKING, Any, Iterable

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from lmdiff.diff import DiffReport
    from lmdiff.geometry import GeoResult
    from lmdiff.tasks.capability_radar import RadarResult


# ── v0.2.x Rich helpers (unchanged) ───────────────────────────────────


def print_report(
    report: "DiffReport",
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
    summary.add_column("Metric", style="cyan")
    summary.add_column("Level", style="magenta")
    summary.add_column("Value", justify="right")
    summary.add_column("Details", justify="left")
    for r in report.results:
        summary.add_row(r.name, r.level.value, _format_value(r.value), _format_details(r))
    console.print(summary)

    if "behavioral_distance" in [r.name for r in report.results]:
        bd = next(r for r in report.results if r.name == "behavioral_distance")
        _print_degeneracy_warning(bd, name_a, name_b, console)
        _print_bd_breakdown(bd, name_a, name_b, console, verbose=verbose)

    console.print(f"[dim]Probes: {report.metadata.get('n_probes', '?')}[/dim]")
    console.print()


def _format_value(value: float | dict | object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict) and "mean" in value:
        return f"{value['mean']:+.4f}"
    return str(value)


def _format_details(r) -> str:
    if r.details is None:
        return ""
    if r.name == "token_kl":
        forward = r.details.get("forward_kl", float("nan"))
        backward = r.details.get("backward_kl", float("nan"))
        return f"forward={forward:.4f}  backward={backward:.4f}"
    if r.name == "token_entropy":
        h_a = r.details.get("entropy_a", float("nan"))
        h_b = r.details.get("entropy_b", float("nan"))
        return f"H(A)={h_a:.4f}  H(B)={h_b:.4f}"
    return ""


def _print_degeneracy_warning(bd, name_a: str, name_b: str, console: Console) -> None:
    if bd.details is None:
        return
    deg_a = bd.details.get("degeneracy_rate_a", 0.0)
    deg_b = bd.details.get("degeneracy_rate_b", 0.0)
    if deg_a > 0.05 or deg_b > 0.05:
        console.print(
            f"[yellow]⚠ degeneracy: {deg_a:.0%} of {name_a}'s outputs and "
            f"{deg_b:.0%} of {name_b}'s outputs were repetitive[/yellow]"
        )


def _print_bd_breakdown(
    bd, name_a: str, name_b: str, console: Console, *, verbose: bool = False,
) -> None:
    if bd.details is None or "per_prompt" not in bd.details:
        return

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
    max_rows = None if verbose else 10
    n_hidden = 0
    if max_rows is not None and len(all_rows) > max_rows:
        half = max_rows // 2

        def _sort_key(pp: dict) -> float:
            v = pp.get("bd", 0.0)
            return v if not math.isnan(v) else float("-inf")

        sorted_rows = sorted(all_rows, key=_sort_key, reverse=True)
        top = sorted_rows[:half]
        bottom = sorted_rows[-half:]
        n_hidden = len(all_rows) - max_rows
        display = top + [None] + bottom
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


def print_radar(result: "RadarResult", console: Console | None = None) -> None:
    """Print a CapabilityRadar / RadarResult as a per-domain table."""
    console = console or Console()
    name_a = result.engine_a_name
    name_b = result.engine_b_name or "—"
    console.print()
    console.rule(f"[bold]Capability Radar: {name_a} vs {name_b}[/bold]")
    console.print()
    domains = list(result.domains)
    if result.b_by_domain is None:
        tbl = Table(title="Per-Domain Accuracy", show_lines=False)
        tbl.add_column("Domain", style="cyan")
        tbl.add_column("N", justify="right")
        tbl.add_column("Acc(A)", justify="right")
        for d in domains:
            a = result.a_by_domain.get(d)
            if a is None:
                continue
            tbl.add_row(d, str(a.n_probes), f"{a.accuracy:.2%}")
        console.print(tbl)
        return

    tbl = Table(title="Per-Domain Accuracy + BD", show_lines=False)
    tbl.add_column("Domain", style="cyan")
    tbl.add_column("N", justify="right")
    tbl.add_column("Acc(A)", justify="right")
    tbl.add_column("Acc(B)", justify="right")
    tbl.add_column("ΔAcc", justify="right")
    tbl.add_column("BD", justify="right")
    tbl.add_column("BD(healthy)", justify="right")
    for d in domains:
        a = result.a_by_domain.get(d)
        b = result.b_by_domain.get(d)
        if a is None or b is None:
            continue
        bd = result.bd_by_domain.get(d, float("nan"))
        bdh = result.bd_healthy_by_domain.get(d, float("nan"))
        delta = b.accuracy - a.accuracy
        tbl.add_row(
            d,
            str(a.n_probes),
            f"{a.accuracy:.2%}",
            f"{b.accuracy:.2%}",
            f"{delta:+.2%}",
            "n/a" if math.isnan(bd) else f"{bd:.4f}",
            "n/a" if math.isnan(bdh) else f"{bdh:.4f}",
        )
    console.print(tbl)


def print_geometry(result: "GeoResult", console: Console | None = None) -> None:
    """Print ChangeGeometry analysis: magnitude table + original cosine matrix,
    plus selective cosine matrix and const_frac column when the decomposition
    fields are populated (schema v2 or a fresh analyze() result).
    """
    console = console or Console()

    has_decomp = bool(result.selective_cosine_matrix) and bool(result.selective_magnitudes)
    const_fractions = result.constant_fractions if has_decomp else {}

    console.print()
    console.rule(
        f"[bold]Change Geometry: {result.base_name} vs "
        f"{len(result.variant_names)} variants[/bold]"
    )
    console.print()

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


# ─────────────────────────────────────────────────────────────────────
#  v0.3.0 commit 1.7 — 5-layer renderer
# ─────────────────────────────────────────────────────────────────────


# ── ANSI palette ──────────────────────────────────────────────────────


_ANSI: dict[str, str] = {
    "bold":     "\033[1m",
    "red":      "\033[31m",
    "red_bold": "\033[1;31m",
    "orange":      "\033[38;5;208m",
    "orange_bold": "\033[1;38;5;208m",
    "yellow":   "\033[33m",
    "green":    "\033[32m",
    "purple":   "\033[35m",
    "dim":      "\033[2m",
    "reset":    "\033[0m",
}


class _Styler:
    """ANSI styler with auto-disable + bold-fallback when colour is off."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, style: str, text: str) -> str:
        if not self.enabled:
            # Bold replaces saturation; everything else (dim included) is plain.
            if style in ("bold", "red_bold", "orange_bold"):
                return text
            return text
        code = _ANSI.get(style)
        if code is None:
            return text
        return f"{code}{text}{_ANSI['reset']}"


def _resolve_color(file: Any, color: bool | None) -> bool:
    """Decide whether to emit ANSI sequences.

    Precedence (per v6 §12.2):

      1. ``color=True`` → always color.
      2. ``color=False`` → never color.
      3. ``NO_COLOR`` env var set → no color.
      4. ``not file.isatty()`` → no color.
      5. Otherwise color on.
    """
    if color is True:
        return True
    if color is False:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    isatty = getattr(file, "isatty", None)
    if isatty is None or not isatty():
        return False
    return True


# ── Helpers ───────────────────────────────────────────────────────────


def _fmt_drift(value: float, width: int = 8) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a".rjust(width)
    return f"{value:.4f}".rjust(width)


def _fmt_share_pct(share: float, width: int = 7) -> str:
    if share is None or (isinstance(share, float) and math.isnan(share)):
        return "n/a".rjust(width)
    return f"{share * 100:.0f}%".rjust(width)


def _fmt_cosine(value: float, width: int = 8, diagonal: bool = False) -> str:
    if diagonal:
        return "—".rjust(width)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a".rjust(width)
    return f"{value:+.2f}".rjust(width)


def _fmt_acc(value: float, artifact: bool, width: int = 7) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a".rjust(width)
    base = f"{value:.2f}"
    if artifact:
        base = base + "*"
    return base.rjust(width)


def _drift_color(value: float) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if value >= 0.20:
        return "red_bold"
    if value >= 0.10:
        return "red"
    if value < 0.025:
        return "green"
    if value < 0.05:
        return "dim"
    return None


def _share_color(share: float) -> str | None:
    if share is None or (isinstance(share, float) and math.isnan(share)):
        return None
    if share >= 0.30:
        return "orange_bold"
    if share >= 0.22:
        return "yellow"
    if share < 0.10:
        return "dim"
    return None


def _cosine_color(value: float, diagonal: bool = False) -> str | None:
    if diagonal:
        return "dim"
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if value >= 0.95:
        return "red_bold"
    if value >= 0.85:
        return "red"
    if value < 0.70:
        return "purple"
    return None


def _ordered_domains(result: "GeoResult", tables: dict | None) -> list[str]:
    """Stable domain order: result.metadata['domain_order'] when present,
    else first-occurrence order in result.probe_domains."""
    meta_order = (result.metadata or {}).get("domain_order")
    if meta_order:
        return list(meta_order)
    seen: list[str] = []
    for d in (result.probe_domains or ()):
        if d is None:
            continue
        if d not in seen:
            seen.append(d)
    if seen:
        return seen
    # Fallback: drift table key order.
    if tables and isinstance(tables.get("share"), dict):
        for v in tables["share"].values():
            if isinstance(v, dict):
                return list(v.keys())
    return []


def _domain_drift_table(result: "GeoResult", domains: list[str]) -> dict[str, dict[str, float]]:
    """Per-variant per-domain raw drift L2 magnitude.

    Reuses ``result.domain_heatmap()`` which is the canonical source.
    Returns ``{}`` if probe_domains is empty.
    """
    if not result.probe_domains:
        return {}
    try:
        return result.domain_heatmap()
    except (ValueError, AttributeError):
        return {}


def _per_variant_total_drift(
    drift_per_domain: dict[str, dict[str, float]],
) -> dict[str, float]:
    """RMS drift across domains per variant."""
    out: dict[str, float] = {}
    for v, by_dom in drift_per_domain.items():
        vals = [float(x) for x in by_dom.values() if isinstance(x, (int, float))]
        if not vals:
            out[v] = float("nan")
            continue
        sq = sum(x * x for x in vals)
        out[v] = math.sqrt(sq / len(vals))
    return out


# ── 5-layer assembly ──────────────────────────────────────────────────


def _layer_banner(title: str, width: int, sty: _Styler) -> list[str]:
    rule = "═" * min(width, 90)
    return [
        sty("bold", rule),
        f"  {sty('bold', title)}",
        sty("bold", rule),
    ]


def _layer1_one_liner(
    findings: tuple, n_probes: int, n_domains: int, sty: _Styler,
) -> list[str]:
    from lmdiff.report._pipeline import _compose_one_liner
    text = _compose_one_liner(findings, n_probes=n_probes, n_domains=n_domains)
    return [sty("bold", line) for line in text.split("\n")]


def _layer2_headlines(findings: tuple, sty: _Styler) -> list[str]:
    from lmdiff._findings import (
        BiggestMoveFinding,
        DirectionClusterFinding,
        DirectionOutlierFinding,
        MostLikeBaseFinding,
        SpecializationPeakFinding,
    )

    out = [sty("bold", "Headlines")]
    label_w = 22

    for f in findings:
        if isinstance(f, MostLikeBaseFinding):
            label = "Most like base".ljust(label_w)
            out.append(
                f"  {label}: {sty('green', f.summary)}"
            )
        elif isinstance(f, BiggestMoveFinding):
            label = "Biggest single move".ljust(label_w)
            out.append(
                f"  {label}: {sty('red', f.summary)}"
            )
        elif isinstance(f, DirectionClusterFinding):
            label = "Direction cluster".ljust(label_w)
            out.append(f"  {label}: {f.summary}")
        elif isinstance(f, DirectionOutlierFinding):
            label = "Direction outlier".ljust(label_w)
            out.append(f"  {label}: {sty('purple', f.summary)}")
        elif isinstance(f, SpecializationPeakFinding):
            # Per v6 §12.6: every Finding.summary must appear verbatim in
            # every renderer. Even when Layer 1 fires on cluster/outlier,
            # peaks still get headline lines here.
            label = "Specialization peak".ljust(label_w)
            out.append(f"  {label}: {sty('orange', f.summary)}")
    return out


def _layer3_share_table(
    variants: list[str],
    domains: list[str],
    share: dict[str, dict[str, float]],
    sty: _Styler,
) -> list[str]:
    if not variants or not domains or not share:
        return []
    title = sty("bold", "Where each variant acts biggest")
    sub = sty("dim", "  share of total drift; rows sum to 100%")
    out = [title, sub, ""]

    name_w = max(8, max(len(v) for v in variants) + 2)
    domain_w = 9
    peak_w = 12
    header_cells = [" " * name_w] + [d[:domain_w].rjust(domain_w) for d in domains] + [
        sty("dim", "peak".rjust(peak_w))
    ]
    out.append("  " + " ".join(header_cells))

    for v in variants:
        row_share = share.get(v, {})
        cells = [v.ljust(name_w)]
        # v0.4.1: skip None entries (out_of_range / variant_only) when
        # finding the peak — `max` over None vs float would raise.
        valid_share = {d: s for d, s in row_share.items() if s is not None}
        peak_dom = (
            max(valid_share, key=lambda d: valid_share[d])
            if valid_share else None
        )
        for d in domains:
            val = row_share.get(d, 0.0)
            text = _fmt_share_pct(val, width=domain_w)
            color = _share_color(val)
            cells.append(sty(color, text) if color else text)
        peak_label = sty("dim", (peak_dom or "—").rjust(peak_w))
        cells.append(peak_label)
        out.append("  " + " ".join(cells))
    return out


def _layer3_drift_table(
    variants: list[str],
    domains: list[str],
    drift: dict[str, dict[str, float]],
    norm_totals: dict[str, float],
    sty: _Styler,
) -> list[str]:
    if not variants or not domains or not drift:
        return []
    title = sty("bold", "How big is each move")
    sub = sty("dim", "  per-domain ‖δ‖ raw; rightmost col is per-√token (comparable)")
    out = [title, sub, ""]

    name_w = max(8, max(len(v) for v in variants) + 2)
    domain_w = 9
    total_w = 10
    header_cells = [" " * name_w] + [d[:domain_w].rjust(domain_w) for d in domains] + [
        sty("dim", "‖δ‖/√tok".rjust(total_w))
    ]
    out.append("  " + " ".join(header_cells))

    for v in variants:
        row = drift.get(v, {})
        cells = [v.ljust(name_w)]
        for d in domains:
            val = row.get(d, float("nan"))
            text = _fmt_drift(val, width=domain_w)
            color = _drift_color(val)
            cells.append(sty(color, text) if color else text)
        nt = norm_totals.get(v, float("nan"))
        # Normalized magnitudes are typically 0.01–0.30 — give them
        # 4 decimal places so they don't all collapse to 0.0.
        if isinstance(nt, (int, float)) and nt == nt:
            nt_text = f"{nt:.4f}".rjust(total_w)
        else:
            nt_text = "n/a".rjust(total_w)
        cells.append(sty("bold", nt_text))
        out.append("  " + " ".join(cells))
    return out


def _layer3_cosine_table(
    variants: list[str],
    cosine: dict[str, dict[str, float]],
    sty: _Styler,
) -> list[str]:
    if not variants or not cosine:
        return []
    title = sty("bold", "Direction agreement")
    sub = sty("dim", "  cosine of δ vectors  (red = same direction; gray-purple = different)")
    out = [title, sub, ""]

    name_w = max(8, max(len(v) for v in variants) + 2)
    cell_w = max(8, max(len(v) for v in variants) + 1)
    header_cells = [" " * name_w] + [v.rjust(cell_w) for v in variants]
    out.append("  " + " ".join(header_cells))

    for v in variants:
        row = cosine.get(v, {})
        cells = [v.ljust(name_w)]
        for w in variants:
            diagonal = (v == w)
            val = row.get(w, float("nan"))
            text = _fmt_cosine(val, width=cell_w, diagonal=diagonal)
            color = _cosine_color(val, diagonal=diagonal)
            cells.append(sty(color, text) if color else text)
        out.append("  " + " ".join(cells))
    return out


def _layer3_accuracy_table(
    variants: list[str],
    accuracy: dict[str, dict[str, float]],
    artifact_tasks: set[str],
    sty: _Styler,
) -> list[str]:
    if not accuracy or not variants:
        return []
    tasks: list[str] = []
    for v in variants:
        for t in (accuracy.get(v) or {}):
            if t not in tasks:
                tasks.append(t)
    if not tasks:
        return []

    title = sty("bold", "Per-task accuracy")
    out = [title, ""]

    name_w = max(8, max(len(v) for v in variants) + 2)
    cell_w = 9
    header_cells = [" " * name_w] + [t[:cell_w].rjust(cell_w) for t in tasks]
    out.append("  " + " ".join(header_cells))

    for v in variants:
        row = accuracy.get(v) or {}
        cells = [v.ljust(name_w)]
        for t in tasks:
            val = row.get(t)
            artifact = t in artifact_tasks
            text = _fmt_acc(val if val is not None else float("nan"),
                            artifact=artifact, width=cell_w)
            if artifact:
                cells.append(sty("yellow", text))
            else:
                cells.append(text)
        out.append("  " + " ".join(cells))
    return out


def _layer4_caveats(findings: tuple, sty: _Styler) -> list[str]:
    """Layer 4: yellow heading + caveat / warning bullets + the fixed
    mental-model reminder ('drift shows where, not whether').
    """
    from lmdiff._findings import (
        AccuracyArtifactFinding,
        BaseAccuracyMissingFinding,
        TokenizerMismatchFinding,
    )

    out = [sty("yellow", "Caveats")]
    for f in findings:
        if isinstance(f, AccuracyArtifactFinding):
            marker = sty("yellow", "*")
            text = f.summary
            out.append(f"  {marker} {sty('yellow', text)}")
        elif isinstance(f, TokenizerMismatchFinding):
            out.append(f"  • {sty('yellow', f.summary)}")
        elif isinstance(f, BaseAccuracyMissingFinding):
            out.append(f"  • {f.summary}")

    # Fixed mental-model reminder — always shown. Per spec invariant #7,
    # NOT a finding type (kept inline; commit 1.11 may revisit).
    out.append(
        "  • Drift magnitude shows where variants change, not whether changes help."
    )
    return out


def _layer5_pointers(result: "GeoResult", sty: _Styler) -> list[str]:
    meta = result.metadata or {}
    pointers: list[tuple[str, str]] = []
    for label, key in (
        ("Full results JSON", "summary_json_path"),
        ("Geometry data    ", "georesult_json_path"),
        ("Detail figures   ", "figures_dir"),
    ):
        val = meta.get(key)
        if val:
            pointers.append((label, str(val)))
    pointers.append(("Metric definitions", "docs/metrics.pdf"))

    if not any(meta.get(k) for k in ("summary_json_path", "georesult_json_path", "figures_dir")):
        return [
            sty("bold", "See also"),
            sty("dim", '  (in-memory result; call result.save("path.json") to persist)'),
        ]
    out = [sty("bold", "See also")]
    for label, val in pointers:
        out.append(f"  {label.ljust(20)} {sty('dim', val)}")
    return out


# ── Compact (< 80 cols) per-variant fallback ──────────────────────────


def _compact_per_variant(
    variants: list[str],
    domains: list[str],
    share: dict[str, dict[str, float]],
    drift: dict[str, dict[str, float]],
    norm_totals: dict[str, float],
    sty: _Styler,
) -> list[str]:
    """Width-constrained alternative to Tables 1-3: per-variant blocks."""
    out = [sty("bold", "Per-variant breakdown"), ""]
    for v in variants:
        nt = norm_totals.get(v, float("nan"))
        nt_text = f"{nt:.4f}" if isinstance(nt, (int, float)) and nt == nt else "n/a"
        out.append(f"  {sty('bold', v)}  ‖δ‖/√tok {sty('bold', nt_text)}")
        sh = share.get(v, {})
        if sh:
            top = sorted(sh.items(), key=lambda kv: -kv[1])[:3]
            line = "    share peaks: " + ",  ".join(
                f"{d} {sty(_share_color(s) or '', f'{s*100:.0f}%')}"
                for d, s in top
            )
            out.append(line)
        dr = drift.get(v, {})
        if dr:
            top = sorted(dr.items(), key=lambda kv: -kv[1] if isinstance(kv[1], (int, float)) else 0)[:3]
            line = "    drift peaks: " + ",  ".join(
                f"{d} {sty(_drift_color(s) or '', _fmt_drift(s).strip())}"
                for d, s in top
            )
            out.append(line)
        out.append("")
    return out


# ── Entry point ───────────────────────────────────────────────────────


def render(
    result: "GeoResult",
    *,
    findings: tuple | None = None,
    tables: dict | None = None,
    file: Any = None,
    width: int | None = None,
    color: bool | None = None,
    **_unused,
) -> str:
    """Render the v0.3.0 5-layer terminal report for ``result``.

    Writes to ``file`` (defaults to ``sys.stdout``) and returns the
    formatted string. ``findings=`` overrides ``result.findings`` (useful
    for torch-free callers — see PR notes about the v0.2.x torch leak).
    ``width`` overrides ``shutil.get_terminal_size().columns``.
    ``color`` overrides the auto-detect (True / False / None).
    """
    out_file = file if file is not None else sys.stdout

    if width is None:
        try:
            width = shutil.get_terminal_size((100, 20)).columns
        except Exception:  # noqa: BLE001
            width = 100

    enabled_color = _resolve_color(out_file, color)
    sty = _Styler(enabled=enabled_color)

    if findings is None:
        findings = result.findings  # type: ignore[assignment]
    findings = tuple(findings or ())

    if tables is None:
        from lmdiff.report._pipeline import build_tables
        tables = build_tables(result)

    domains = _ordered_domains(result, tables)
    n_probes = result.n_probes
    n_domains = len(domains)
    variants = sorted(list(result.variant_names))

    drift = _domain_drift_table(result, domains)
    # Per-variant total: switch from RMS-of-per-domain (raw, not
    # comparable across runs) to per-√token-normalized magnitude. Matches
    # ``change_size_bars.png`` right pane and the markdown report.
    norm_totals = dict(result.magnitudes_normalized or {})
    share = tables.get("share", {}) if tables else {}
    cosine = tables.get("cosine", {}) if tables else {}
    accuracy = tables.get("accuracy", {}) if tables else {}

    # AccuracyArtifactFinding tasks → for Layer 3 Table 4 markers.
    from lmdiff._findings import AccuracyArtifactFinding
    artifact_tasks = {
        f.details.get("task")
        for f in findings
        if isinstance(f, AccuracyArtifactFinding)
    }
    artifact_tasks.discard(None)

    title = (
        f"Family experiment: {result.base_name} vs {len(variants)} variants  "
        f"({n_probes} probes, {n_domains} domains)"
    )

    lines: list[str] = []
    lines.extend(_layer_banner(title, width=width, sty=sty))
    lines.append("")
    lines.extend(_layer1_one_liner(findings, n_probes, n_domains, sty))
    lines.append("")
    lines.extend(_layer2_headlines(findings, sty))
    lines.append("")

    if width < 80 and variants:
        lines.extend(_compact_per_variant(variants, domains, share, drift, norm_totals, sty))
    else:
        lines.extend(_layer3_share_table(variants, domains, share, sty))
        lines.append("")
        lines.extend(_layer3_drift_table(variants, domains, drift, norm_totals, sty))
        lines.append("")
        lines.extend(_layer3_cosine_table(variants, cosine, sty))
        if accuracy:
            lines.append("")
            lines.extend(
                _layer3_accuracy_table(variants, accuracy, artifact_tasks, sty),
            )

    lines.append("")
    lines.extend(_layer4_caveats(findings, sty))
    lines.append("")
    lines.extend(_layer5_pointers(result, sty))
    rule = "═" * min(width, 90)
    lines.append("")
    lines.append(sty("bold", rule))

    text = "\n".join(lines) + "\n"
    out_file.write(text)
    return text
