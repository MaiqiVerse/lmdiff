"""HTML self-contained renderer (commit 1.10 of v0.3.0 batch 4).

Produces a single HTML file:

  - Opens in any browser without network access.
  - Light / dark theme toggle (defaults to OS preference).
  - Same 5-layer content as the terminal renderer (one-liner, headlines,
    tables, caveats, methodology) plus the 3 application-tier figures.
  - Survives email attachment, Slack upload, GitHub Wiki paste.

Modes
-----
``embed_images=True`` (default): images base64-encoded into ``data:`` URIs
inside the HTML. Single self-contained file, ~1-2 MB. Use for sharing.

``embed_images=False``: HTML references PNG files in a ``figs/`` sibling
directory using relative paths. HTML stays under ~50 KB but the figs
directory must travel alongside. Requires ``out_path``.

Cross-renderer consistency: every ``Finding.summary`` from
``result.findings`` appears verbatim in the body, matching the terminal
and markdown renderers (v6 §12.6).
"""
from __future__ import annotations

import base64
import io
import math
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from lmdiff.geometry import GeoResult


# ── CSS ───────────────────────────────────────────────────────────────


_CSS = """
:root {
    --bg: #ffffff;
    --fg: #222222;
    --muted: #666666;
    --link: #0066cc;
    --border: #e5e5e5;
    --accent-red: #c0392b;
    --accent-green: #27ae60;
    --accent-orange: #b35806;
    --accent-yellow: #d4a017;
    --accent-purple: #6f42c1;
    --table-header-bg: #f5f5f5;
    --caveat-bg: #fff7e0;
    --caveat-border: #d4a017;
    --code-bg: #f3f3f3;
}

[data-theme="dark"] {
    --bg: #1a1a1a;
    --fg: #e0e0e0;
    --muted: #aaaaaa;
    --link: #66a3ff;
    --border: #333333;
    --accent-red: #e06c75;
    --accent-green: #98c379;
    --accent-orange: #ff9c4f;
    --accent-yellow: #f0c674;
    --accent-purple: #c678dd;
    --table-header-bg: #262626;
    --caveat-bg: #2a2410;
    --caveat-border: #d4a017;
    --code-bg: #2a2a2a;
}

@media (prefers-color-scheme: dark) {
    [data-theme="auto"] {
        --bg: #1a1a1a;
        --fg: #e0e0e0;
        --muted: #aaaaaa;
        --link: #66a3ff;
        --border: #333333;
        --accent-red: #e06c75;
        --accent-green: #98c379;
        --accent-orange: #ff9c4f;
        --accent-yellow: #f0c674;
        --accent-purple: #c678dd;
        --table-header-bg: #262626;
        --caveat-bg: #2a2410;
        --caveat-border: #d4a017;
        --code-bg: #2a2a2a;
    }
}

* { box-sizing: border-box; }
body {
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui,
        sans-serif;
    line-height: 1.55;
    margin: 0;
    padding: 2em 1em;
    transition: background 0.2s ease, color 0.2s ease;
}
.container { max-width: 1100px; margin: 0 auto; }

header { position: relative; margin-bottom: 1.5em; }
h1 { margin: 0 0 0.2em 0; font-size: 1.6em; }
h2 { margin: 1.6em 0 0.4em; font-size: 1.2em; border-bottom: 1px solid var(--border); padding-bottom: 0.2em; }
.subtitle { color: var(--muted); margin: 0 0 0.6em; font-size: 0.95em; }
.one-liner { font-size: 1.15em; font-weight: 600; margin: 0.6em 0 1em; }

#theme-toggle {
    position: absolute; top: 0; right: 0;
    background: transparent; border: 1px solid var(--border);
    color: var(--fg); font-size: 1.2em; cursor: pointer;
    padding: 0.3em 0.55em; border-radius: 6px;
    line-height: 1;
}
#theme-toggle:hover { border-color: var(--link); }

ul.headlines { padding-left: 1.4em; }
ul.headlines li { margin: 0.25em 0; }

section.figures img {
    display: block; max-width: 100%; height: auto;
    margin: 0.6em auto 1.4em; border: 1px solid var(--border); border-radius: 4px;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.6em 0 1.4em;
    font-size: 0.95em;
}
th, td {
    padding: 0.4em 0.7em;
    border: 1px solid var(--border);
    text-align: right;
    font-variant-numeric: tabular-nums;
}
th { background: var(--table-header-bg); font-weight: 600; }
th:first-child, td:first-child { text-align: left; font-weight: 600; }
caption { caption-side: top; color: var(--muted); padding: 0 0 0.3em; text-align: left; font-size: 0.9em; }

.cell-peak { color: var(--accent-orange); font-weight: 700; }
.cell-large { color: var(--accent-red); font-weight: 700; }
.cell-mid { color: var(--accent-yellow); }
.cell-small { color: var(--accent-green); font-weight: 600; }
.cell-dim { color: var(--muted); }
.cell-purple { color: var(--accent-purple); font-weight: 600; }
.cell-diag { color: var(--muted); text-align: center; }
.cell-artifact-marker { color: var(--accent-yellow); font-weight: 700; }

section.caveats ul { padding-left: 0; list-style: none; }
section.caveats li {
    background: var(--caveat-bg);
    border-left: 4px solid var(--caveat-border);
    padding: 0.55em 0.9em;
    margin: 0.4em 0;
    border-radius: 0 4px 4px 0;
}
section.caveats li.severity-warning { border-left-color: var(--accent-red); }

details.methodology {
    margin-top: 2em;
    padding: 0.6em 1em;
    border: 1px solid var(--border);
    border-radius: 6px;
}
details.methodology summary { cursor: pointer; font-weight: 600; }
details.methodology dl { margin: 0.6em 0 0; }
details.methodology dt { color: var(--muted); float: left; min-width: 14em; }
details.methodology dd { margin: 0 0 0.3em 0; }

footer {
    margin-top: 2.5em;
    padding-top: 1em;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.9em;
}
a { color: var(--link); }

@media print {
    body { padding: 1em; max-width: none; }
    #theme-toggle { display: none; }
    section.figures img { page-break-inside: avoid; }
    details.methodology[open] summary { page-break-after: avoid; }
    details.methodology { break-inside: avoid; }
}
"""


_THEME_SCRIPT = """
(function() {
    var KEY = "lmdiff-theme";
    var btn = document.getElementById("theme-toggle");
    if (!btn) return;
    var current = localStorage.getItem(KEY) || "auto";
    document.documentElement.setAttribute("data-theme", current);
    function label(t) { return t === "auto" ? "🌓" : t === "dark" ? "🌙" : "☀"; }
    btn.textContent = label(current);
    btn.addEventListener("click", function() {
        current = current === "auto" ? "light" : current === "light" ? "dark" : "auto";
        document.documentElement.setAttribute("data-theme", current);
        localStorage.setItem(KEY, current);
        btn.textContent = label(current);
    });
})();
"""


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


def _drift_class(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if value >= 0.10:
        return "cell-large"
    if value < 0.025:
        return "cell-small"
    if value < 0.05:
        return "cell-dim"
    return ""


def _share_class(share: float) -> str:
    if share is None or (isinstance(share, float) and math.isnan(share)):
        return ""
    if share >= 0.30:
        return "cell-peak"
    if share >= 0.22:
        return "cell-mid"
    if share < 0.10:
        return "cell-dim"
    return ""


def _cosine_class(value: float, diagonal: bool = False) -> str:
    if diagonal:
        return "cell-diag"
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if value >= 0.85:
        return "cell-large"
    if value < 0.70:
        return "cell-purple"
    return ""


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


# ── Section builders ──────────────────────────────────────────────────


def _build_summary(findings: tuple, n_probes: int, n_domains: int) -> str:
    from lmdiff.report._pipeline import _compose_one_liner
    from lmdiff._findings import (
        BiggestMoveFinding,
        DirectionClusterFinding,
        DirectionOutlierFinding,
        MostLikeBaseFinding,
        SpecializationPeakFinding,
    )

    one_liner = _compose_one_liner(findings, n_probes=n_probes, n_domains=n_domains)
    one_liner_html = escape(one_liner).replace("\n", "<br>")

    label_map = [
        (MostLikeBaseFinding, "Most like base"),
        (BiggestMoveFinding, "Biggest single move"),
        (DirectionClusterFinding, "Direction cluster"),
        (DirectionOutlierFinding, "Direction outlier"),
        (SpecializationPeakFinding, "Specialization peak"),
    ]

    items: list[str] = []
    for f in findings:
        for cls, label in label_map:
            if isinstance(f, cls):
                items.append(
                    f'<li><strong>{escape(label)}</strong>: '
                    f'{escape(f.summary)}</li>'
                )
                break

    items_html = "\n      ".join(items) if items else "<li><em>(no findings)</em></li>"

    return (
        '\n  <section class="summary">'
        '\n    <h2>Summary</h2>'
        f'\n    <p class="one-liner">{one_liner_html}</p>'
        f'\n    <ul class="headlines">\n      {items_html}\n    </ul>'
        '\n  </section>'
    )


def _build_share_table(
    variants: list[str], domains: list[str], share: dict,
) -> str:
    if not variants or not domains or not share:
        return ""
    head_cells = "".join(f"<th>{escape(d)}</th>" for d in domains)
    rows: list[str] = []
    for v in variants:
        row_share = share.get(v, {})
        cells = []
        peak_dom = (
            max(row_share, key=lambda d: row_share.get(d, 0.0)) if row_share else None
        )
        for d in domains:
            val = row_share.get(d, 0.0)
            cls = _share_class(val)
            attr = f' class="{cls}"' if cls else ""
            cells.append(f'<td{attr}>{_fmt_pct(val)}</td>')
        peak_cell = (
            f'<td class="cell-dim">{escape(peak_dom)}</td>'
            if peak_dom else '<td class="cell-dim">—</td>'
        )
        rows.append(
            f'<tr><td>{escape(v)}</td>{"".join(cells)}{peak_cell}</tr>'
        )
    return (
        '\n    <h3>Where each variant acts biggest</h3>'
        '\n    <table>'
        '\n      <caption>share of total drift; rows sum to 100%</caption>'
        f'\n      <thead><tr><th>variant</th>{head_cells}<th>peak</th></tr></thead>'
        f'\n      <tbody>{"".join(rows)}</tbody>'
        '\n    </table>'
    )


def _build_drift_table(
    variants: list[str], domains: list[str], drift: dict, totals: dict,
) -> str:
    if not variants or not domains or not drift:
        return ""
    head_cells = "".join(f"<th>{escape(d)}</th>" for d in domains)
    rows: list[str] = []
    for v in variants:
        row = drift.get(v, {})
        cells = []
        for d in domains:
            val = row.get(d, float("nan"))
            cls = _drift_class(val)
            attr = f' class="{cls}"' if cls else ""
            cells.append(f'<td{attr}>{_fmt_float(val, 4)}</td>')
        total_val = totals.get(v, float("nan"))
        rows.append(
            f'<tr><td>{escape(v)}</td>{"".join(cells)}'
            f'<td><strong>{_fmt_float(total_val, 4)}</strong></td></tr>'
        )
    return (
        '\n    <h3>How big is each move</h3>'
        '\n    <table>'
        '\n      <caption>per-domain drift magnitude</caption>'
        f'\n      <thead><tr><th>variant</th>{head_cells}<th>total</th></tr></thead>'
        f'\n      <tbody>{"".join(rows)}</tbody>'
        '\n    </table>'
    )


def _build_cosine_table(variants: list[str], cosine: dict) -> str:
    if not variants or not cosine:
        return ""
    head_cells = "".join(f"<th>{escape(v)}</th>" for v in variants)
    rows: list[str] = []
    for a in variants:
        row = cosine.get(a, {})
        cells = []
        for b in variants:
            diagonal = (a == b)
            val = row.get(b, float("nan"))
            cls = _cosine_class(val, diagonal=diagonal)
            attr = f' class="{cls}"' if cls else ""
            text = "—" if diagonal else _fmt_signed(val, 2)
            cells.append(f'<td{attr}>{text}</td>')
        rows.append(f'<tr><td>{escape(a)}</td>{"".join(cells)}</tr>')
    return (
        '\n    <h3>Direction agreement</h3>'
        '\n    <table>'
        '\n      <caption>cosine of δ vectors (red = same direction; gray-purple = different)</caption>'
        f'\n      <thead><tr><th></th>{head_cells}</tr></thead>'
        f'\n      <tbody>{"".join(rows)}</tbody>'
        '\n    </table>'
    )


def _build_accuracy_table(
    variants: list[str], accuracy: dict, artifact_tasks: set,
) -> str:
    if not accuracy or not variants:
        return ""
    tasks: list[str] = []
    for v in variants:
        for t in (accuracy.get(v) or {}):
            if t not in tasks:
                tasks.append(t)
    if not tasks:
        return ""
    head_cells = "".join(f"<th>{escape(t)}</th>" for t in tasks)
    rows: list[str] = []
    for v in variants:
        row = accuracy.get(v) or {}
        cells = []
        for t in tasks:
            val = row.get(t)
            artifact = t in artifact_tasks
            text = (
                _fmt_float(val, 2) if val is not None else "n/a"
            )
            if artifact and val is not None:
                cells.append(
                    f'<td><span class="cell-dim">{text}</span>'
                    f'<span class="cell-artifact-marker">*</span></td>'
                )
            else:
                cells.append(f'<td>{text}</td>')
        rows.append(f'<tr><td>{escape(v)}</td>{"".join(cells)}</tr>')
    return (
        '\n    <h3>Per-task accuracy</h3>'
        '\n    <table>'
        f'\n      <thead><tr><th>variant</th>{head_cells}</tr></thead>'
        f'\n      <tbody>{"".join(rows)}</tbody>'
        '\n    </table>'
    )


def _build_caveats(findings: tuple) -> str:
    from lmdiff._findings import (
        AccuracyArtifactFinding,
        BaseAccuracyMissingFinding,
        TokenizerMismatchFinding,
    )

    items: list[str] = []
    for f in findings:
        if isinstance(f, AccuracyArtifactFinding):
            items.append(
                f'<li class="severity-caveat">'
                f'<span class="cell-artifact-marker">*</span> {escape(f.summary)}'
                f'</li>'
            )
        elif isinstance(f, TokenizerMismatchFinding):
            items.append(
                f'<li class="severity-warning">⚠ {escape(f.summary)}</li>'
            )
        elif isinstance(f, BaseAccuracyMissingFinding):
            items.append(
                f'<li class="severity-caveat">{escape(f.summary)}</li>'
            )
    items.append(
        '<li class="severity-caveat">Drift magnitude shows where variants change, '
        'not whether changes help.</li>'
    )
    return (
        '\n  <section class="caveats">'
        '\n    <h2>Caveats</h2>'
        f'\n    <ul>{"".join(items)}</ul>'
        '\n  </section>'
    )


def _build_methodology(result: "GeoResult", lmdiff_version: str) -> str:
    meta = result.metadata or {}
    rows: list[str] = []

    def _row(label: str, value: Any) -> None:
        if value is None:
            return
        rows.append(
            f'<dt>{escape(label)}</dt><dd>{escape(str(value))}</dd>'
        )

    _row("Probe set", meta.get("probe_set_name"))
    _row("Probe set version", meta.get("probe_set_version"))
    _row("n_probes", result.n_probes)
    _row("n_skipped", meta.get("n_skipped"))
    _row("max_new_tokens", meta.get("max_new_tokens"))
    _row("base", result.base_name)
    _row("variants", ", ".join(result.variant_names))
    _row("schema_version", "5")
    _row("lmdiff", lmdiff_version)

    return (
        '\n  <details class="methodology">'
        '\n    <summary>Methodology (click to expand)</summary>'
        f'\n    <dl>{"".join(rows)}</dl>'
        '\n  </details>'
    )


def _figure_to_b64(figure_renderer, result, **kwargs) -> str:
    """Render a figure to a tempfile then read+base64 the bytes back.

    The viz renderers accept a ``Path`` and call ``savefig(path)``
    internally, so they can't write to a ``BytesIO`` directly.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        path = Path(tf.name)
    try:
        figure_renderer(result, path, **kwargs)
        data = path.read_bytes()
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_figures_section(
    result: "GeoResult",
    embed_images: bool,
    figs_dir: Path | None,
    findings: tuple,
) -> str:
    from lmdiff.viz.change_size import render_change_size
    from lmdiff.viz.direction import render_direction
    from lmdiff.viz.drift_share import render_drift_share

    titles = [
        ("Where each variant acts biggest",
         "drift and share heatmaps",
         "drift_share_dual.png", render_drift_share),
        ("Direction agreement",
         "cosine matrix",
         "direction_agreement.png", render_direction),
        ("How big is each move",
         "raw vs normalized magnitude bars",
         "change_size_bars.png", render_change_size),
    ]

    parts: list[str] = ['\n  <section class="figures">']
    for h2, alt, filename, fn in titles:
        if embed_images:
            src = _figure_to_b64(fn, result, findings=findings)
        else:
            assert figs_dir is not None
            out_path = figs_dir / filename
            fn(result, out_path, findings=findings)
            src = f"figs/{filename}"
        parts.append(f'\n    <h2>{escape(h2)}</h2>')
        parts.append(
            f'\n    <img src="{src}" alt="{escape(alt)}" loading="lazy">'
        )
    parts.append('\n  </section>')
    return "".join(parts)


# ── Entry point ───────────────────────────────────────────────────────


def render(
    result: "GeoResult",
    out_path: str | Path | None = None,
    *,
    findings: tuple | None = None,
    tables: dict | None = None,  # noqa: ARG001 (built locally)
    embed_images: bool = True,
    theme: Literal["auto", "light", "dark"] = "auto",
    **_unused,
) -> Any:
    """Render a GeoResult as a self-contained HTML document.

    Returns the HTML string when ``out_path`` is None, else the written
    ``Path``. ``embed_images=False`` requires ``out_path``.
    """
    if not embed_images and out_path is None:
        raise ValueError("embed_images=False requires out_path")
    if theme not in ("auto", "light", "dark"):
        raise ValueError(
            f"theme must be 'auto', 'light', or 'dark'; got {theme!r}"
        )

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

    figs_dir: Path | None = None
    if not embed_images:
        figs_dir = Path(out_path).parent / "figs"
        figs_dir.mkdir(parents=True, exist_ok=True)

    title = (
        f"lmdiff report — {result.base_name} vs "
        f"{', '.join(result.variant_names)}"
    )
    n_probes = result.n_probes
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    head = (
        f'<!DOCTYPE html>\n'
        f'<html lang="en" data-theme="{theme}">\n'
        f'<head>\n'
        f'  <meta charset="utf-8">\n'
        f'  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'  <title>{escape(title)}</title>\n'
        f'  <style>{_CSS}</style>\n'
        f'</head>'
    )
    if not embed_images:
        head_comment = (
            '<!-- This HTML references PNG files in ./figs/ — keep them '
            'together. -->\n'
        )
        head = head_comment + head

    summary_html = _build_summary(findings, n_probes, len(domains))
    figures_html = _build_figures_section(result, embed_images, figs_dir, findings)
    tables_section = (
        '\n  <section class="tables">'
        '\n    <h2>Numeric tables</h2>'
        f'{_build_share_table(variants, domains, share)}'
        f'{_build_drift_table(variants, domains, drift, totals)}'
        f'{_build_cosine_table(variants, cosine)}'
        f'{_build_accuracy_table(variants, accuracy, artifact_tasks)}'
        '\n  </section>'
    )
    caveats_html = _build_caveats(findings)
    methodology_html = _build_methodology(result, lmdiff_version)

    body_subtitle = (
        f"<strong>{escape(result.base_name)}</strong> vs "
        f"{len(result.variant_names)} variants · "
        f"{n_probes} probes · {escape(timestamp)}"
    )

    body = (
        '\n<body>'
        '\n<div class="container">'
        '\n  <header>'
        '\n    <button id="theme-toggle" type="button" '
        'aria-label="Toggle theme">🌓</button>'
        '\n    <h1>lmdiff Family Report</h1>'
        f'\n    <p class="subtitle">{body_subtitle}</p>'
        '\n  </header>'
        f'{summary_html}'
        f'{figures_html}'
        f'{tables_section}'
        f'{caveats_html}'
        f'{methodology_html}'
        '\n  <footer>'
        f'\n    <p>Generated by '
        f'<a href="https://github.com/MaiqiVerse/lmdiff">lmdiff-kit</a> '
        f'v{escape(lmdiff_version)} on {escape(timestamp)}.</p>'
        '\n  </footer>'
        '\n</div>'
        f'\n<script>{_THEME_SCRIPT}</script>'
        '\n</body>\n</html>\n'
    )

    html = head + body

    if out_path is not None:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(html, encoding="utf-8")
        return out_p
    return html


__all__ = ["render"]
