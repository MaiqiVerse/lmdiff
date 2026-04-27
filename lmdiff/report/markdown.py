"""Markdown renderer (commit 1.11 of v0.3.0 batch 4).

Produces GitHub-flavored markdown that mirrors the terminal renderer's
5-layer structure. Output renders cleanly in GitHub / GitLab / VSCode
preview / pandoc / Slack — **no inline HTML**, no extensions, no styles.

Cross-renderer consistency (v6 §12.6): every ``Finding.summary`` from
``result.findings`` appears verbatim. Only formatting differs (terminal
ANSI → markdown bold; terminal dim → markdown italics).

Figure links
------------
``out_path`` and ``figures_dir`` jointly determine where figure links
point:

  - ``out_path`` given, ``figures_dir`` not given: figures rendered to
    ``out_path.parent / "figs"`` and linked relatively.
  - Both given: figures rendered to ``figures_dir`` and linked relatively
    from ``out_path``.
  - Neither given: figure links omitted entirely. The caller can render
    figures separately and edit links manually.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lmdiff.geometry import GeoResult


# ── Helpers ───────────────────────────────────────────────────────────


def _ordered_domains(result: "GeoResult", explicit: list[str] | None = None) -> list[str]:
    if explicit:
        return list(explicit)
    meta_order = (result.metadata or {}).get("domain_order")
    if meta_order:
        return list(meta_order)
    seen: list[str] = []
    for d in (result.probe_domains or ()):
        if d is None:
            continue
        if d not in seen:
            seen.append(d)
    return seen


def _fmt_float(value: float, places: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{places}f}"


def _fmt_signed(value: float, places: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:+.{places}f}"


def _fmt_pct(share: float) -> str:
    if share is None or (isinstance(share, float) and math.isnan(share)):
        return "n/a"
    return f"{share * 100:.0f}%"


def _domain_drift(result: "GeoResult") -> dict[str, dict[str, float]]:
    if not result.probe_domains:
        return {}
    try:
        return result.domain_heatmap()
    except (ValueError, AttributeError):
        return {}


def _per_variant_total_drift(drift: dict[str, dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for v, by_dom in drift.items():
        vals = [float(x) for x in by_dom.values() if isinstance(x, (int, float))]
        if not vals:
            out[v] = float("nan")
            continue
        sq = sum(x * x for x in vals)
        out[v] = math.sqrt(sq / len(vals))
    return out


def _resolve_figures_dir(
    out_path: Path | None, figures_dir: Path | str | None,
) -> Path | None:
    """Resolve where figures live relative to the markdown file."""
    if figures_dir is not None:
        return Path(figures_dir)
    if out_path is None:
        return None
    return out_path.parent / "figs"


def _figure_link(filename: str, figs_dir: Path, out_dir: Path | None, alt: str) -> str:
    """Render a markdown image link with a relative path from ``out_dir``."""
    target = figs_dir / filename
    if out_dir is not None:
        try:
            rel = os.path.relpath(target, start=out_dir)
            rel = rel.replace(os.sep, "/")
        except ValueError:
            rel = str(target).replace(os.sep, "/")
    else:
        rel = str(target).replace(os.sep, "/")
    return f"[![{alt}]({rel})]({rel})"


# ── Section builders ──────────────────────────────────────────────────


def _build_summary(findings: tuple, n_probes: int, n_domains: int) -> list[str]:
    from lmdiff.report._pipeline import _compose_one_liner
    from lmdiff._findings import (
        BiggestMoveFinding,
        DirectionClusterFinding,
        DirectionOutlierFinding,
        MostLikeBaseFinding,
        SpecializationPeakFinding,
    )

    one_liner = _compose_one_liner(findings, n_probes=n_probes, n_domains=n_domains)
    lines = ["## Summary", ""]
    for chunk in one_liner.split("\n"):
        lines.append(f"> {chunk}" if chunk else ">")
    lines.append("")

    label_map = [
        (MostLikeBaseFinding, "Most like base"),
        (BiggestMoveFinding, "Biggest single move"),
        (DirectionClusterFinding, "Direction cluster"),
        (DirectionOutlierFinding, "Direction outlier"),
        (SpecializationPeakFinding, "Specialization peak"),
    ]
    bullets: list[str] = []
    for f in findings:
        for cls, label in label_map:
            if isinstance(f, cls):
                bullets.append(f"- **{label}**: {f.summary}")
                break
    if bullets:
        lines.extend(bullets)
        lines.append("")
    return lines


def _build_share_table(
    variants: list[str], domains: list[str], share: dict,
) -> list[str]:
    if not variants or not domains or not share:
        return []
    lines = ["## Where each variant acts biggest", ""]
    lines.append("_share of total drift; rows sum to 100%_")
    lines.append("")
    header = "| variant | " + " | ".join(domains) + " | peak |"
    sep = "|---|" + "---|" * (len(domains) + 1)
    lines.append(header)
    lines.append(sep)
    for v in variants:
        row_share = share.get(v, {})
        peak_dom = (
            max(row_share, key=lambda d: row_share.get(d, 0.0)) if row_share else "—"
        )
        cells: list[str] = []
        for d in domains:
            val = row_share.get(d, 0.0)
            text = _fmt_pct(val)
            if d == peak_dom and isinstance(val, (int, float)) and not math.isnan(val):
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {v} | " + " | ".join(cells) + f" | {peak_dom} |")
    lines.append("")
    return lines


def _build_drift_table(
    variants: list[str], domains: list[str], drift: dict, totals: dict,
) -> list[str]:
    if not variants or not domains or not drift:
        return []
    lines = ["## How big is each move", ""]
    lines.append("_per-domain drift magnitude_")
    lines.append("")
    header = "| variant | " + " | ".join(domains) + " | total |"
    sep = "|---|" + "---|" * (len(domains) + 1)
    lines.append(header)
    lines.append(sep)
    for v in variants:
        row = drift.get(v, {})
        peak_dom = max(
            (d for d in domains if isinstance(row.get(d), (int, float))),
            key=lambda d: row.get(d, 0.0),
            default=None,
        )
        cells: list[str] = []
        for d in domains:
            val = row.get(d, float("nan"))
            text = _fmt_float(val, 4)
            if d == peak_dom and isinstance(val, (int, float)) and not math.isnan(val):
                text = f"**{text}**"
            cells.append(text)
        total = totals.get(v, float("nan"))
        lines.append(
            f"| {v} | " + " | ".join(cells) + f" | **{_fmt_float(total, 4)}** |"
        )
    lines.append("")
    return lines


def _build_cosine_table(variants: list[str], cosine: dict) -> list[str]:
    if not variants or not cosine:
        return []
    lines = ["## Direction agreement", ""]
    lines.append(
        "_cosine of δ vectors (red = same direction; gray-purple = different)_"
    )
    lines.append("")
    header = "| | " + " | ".join(variants) + " |"
    sep = "|---|" + "---|" * len(variants)
    lines.append(header)
    lines.append(sep)
    for a in variants:
        row = cosine.get(a, {})
        cells: list[str] = []
        for b in variants:
            if a == b:
                cells.append("—")
                continue
            val = row.get(b, float("nan"))
            cells.append(_fmt_signed(val, 2))
        lines.append(f"| **{a}** | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _build_accuracy_table(
    variants: list[str], accuracy: dict, artifact_tasks: set,
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

    lines = ["## Per-task accuracy", ""]
    header = "| variant | " + " | ".join(tasks) + " |"
    sep = "|---|" + "---|" * len(tasks)
    lines.append(header)
    lines.append(sep)
    for v in variants:
        row = accuracy.get(v) or {}
        cells: list[str] = []
        for t in tasks:
            val = row.get(t)
            if val is None:
                cells.append("n/a")
                continue
            text = _fmt_float(val, 2)
            if t in artifact_tasks:
                text = f"{text}*"
            cells.append(text)
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _build_caveats(findings: tuple) -> list[str]:
    from lmdiff._findings import (
        AccuracyArtifactFinding,
        BaseAccuracyMissingFinding,
        TokenizerMismatchFinding,
    )

    lines = ["## Caveats", ""]
    for f in findings:
        if isinstance(f, AccuracyArtifactFinding):
            lines.append(f"> ⚠ {f.summary}")
            lines.append("")
        elif isinstance(f, TokenizerMismatchFinding):
            lines.append(f"> ⚠ {f.summary}")
            lines.append("")
        elif isinstance(f, BaseAccuracyMissingFinding):
            lines.append(f"> {f.summary}")
            lines.append("")
    lines.append(
        "> Drift magnitude shows where variants change, not whether changes help."
    )
    lines.append("")
    return lines


def _build_methodology(result: "GeoResult", lmdiff_version: str) -> list[str]:
    meta = result.metadata or {}
    lines = ["## Methodology", ""]

    def _row(label: str, value: Any) -> None:
        if value is None:
            return
        lines.append(f"- **{label}**: {value}")

    _row("Probe set", meta.get("probe_set_name"))
    _row("Probe set version", meta.get("probe_set_version"))
    _row("n_probes", result.n_probes)
    _row("n_skipped", meta.get("n_skipped"))
    _row("max_new_tokens", meta.get("max_new_tokens"))
    _row("Base", result.base_name)
    _row("Variants", ", ".join(result.variant_names))
    _row("schema_version", "5")
    _row("lmdiff", lmdiff_version)
    lines.append("")
    return lines


# ── Entry point ───────────────────────────────────────────────────────


def render(
    result: "GeoResult",
    out_path: str | Path | None = None,
    *,
    findings: tuple | None = None,
    tables: dict | None = None,  # noqa: ARG001 (built locally)
    figures_dir: str | Path | None = None,
    **_unused,
) -> Any:
    """Render a GeoResult as GitHub-flavored markdown.

    Returns the markdown string when ``out_path`` is None, else the
    written ``Path``. See module docstring for figure-link semantics.
    """
    if findings is None:
        findings = result.findings
    findings = tuple(findings or ())

    from lmdiff.report._pipeline import build_tables as _build_tables
    tables_local = _build_tables(result)

    variants = sorted(list(result.variant_names))
    domains = _ordered_domains(result)
    drift = _domain_drift(result)
    totals = _per_variant_total_drift(drift)
    share = tables_local.get("share", {})
    cosine = tables_local.get("cosine", {})
    accuracy = tables_local.get("accuracy", {})

    from lmdiff._findings import AccuracyArtifactFinding
    artifact_tasks = {
        f.details.get("task")
        for f in findings
        if isinstance(f, AccuracyArtifactFinding)
    }
    artifact_tasks.discard(None)

    try:
        from lmdiff import __version__ as lmdiff_version
    except ImportError:
        lmdiff_version = "0.3.0-dev"

    out_p = Path(out_path) if out_path is not None else None
    figs_dir = _resolve_figures_dir(out_p, figures_dir)
    out_dir = out_p.parent if out_p is not None else None

    render_figs = (figs_dir is not None and out_dir is not None)

    if render_figs:
        from lmdiff.viz.change_size import render_change_size
        from lmdiff.viz.direction import render_direction
        from lmdiff.viz.drift_share import render_drift_share

        figs_dir.mkdir(parents=True, exist_ok=True)
        render_drift_share(
            result, figs_dir / "drift_share_dual.png", findings=findings,
        )
        render_direction(
            result, figs_dir / "direction_agreement.png", findings=findings,
        )
        render_change_size(
            result, figs_dir / "change_size_bars.png", findings=findings,
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append("# lmdiff Family Report")
    lines.append("")
    lines.append(
        f"**{result.base_name}** vs {len(result.variant_names)} variants · "
        f"{result.n_probes} probes · {timestamp}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.extend(_build_summary(findings, result.n_probes, len(domains)))

    lines.extend(_build_share_table(variants, domains, share))
    if render_figs and figs_dir is not None and out_dir is not None:
        lines.append(
            _figure_link(
                "drift_share_dual.png", figs_dir, out_dir,
                "drift and share heatmaps",
            )
        )
        lines.append("")

    lines.extend(_build_drift_table(variants, domains, drift, totals))

    lines.extend(_build_cosine_table(variants, cosine))
    if render_figs and figs_dir is not None and out_dir is not None:
        lines.append(
            _figure_link(
                "direction_agreement.png", figs_dir, out_dir, "cosine agreement",
            )
        )
        lines.append("")

    lines.extend(_build_accuracy_table(variants, accuracy, artifact_tasks))

    if render_figs and figs_dir is not None and out_dir is not None:
        lines.append("## How big has each variant moved?")
        lines.append("")
        lines.append(
            _figure_link(
                "change_size_bars.png", figs_dir, out_dir,
                "raw vs normalized magnitude bars",
            )
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.extend(_build_caveats(findings))

    lines.append("---")
    lines.append("")
    lines.extend(_build_methodology(result, lmdiff_version))

    lines.append("---")
    lines.append("")
    lines.append(
        f"_Generated by [lmdiff-kit](https://github.com/MaiqiVerse/lmdiff) "
        f"v{lmdiff_version}._"
    )
    lines.append("")

    text = "\n".join(lines)

    if out_p is not None:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(text, encoding="utf-8")
        return out_p
    return text


__all__ = ["render"]
