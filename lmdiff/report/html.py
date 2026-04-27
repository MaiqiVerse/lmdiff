"""HTML renderer skeleton (commit 1.5).

v0.3.0 ships a minimal but valid HTML5 page (``<!DOCTYPE html>``, body
containing the magnitudes table and findings list). The polished
self-contained report with theme toggle and base64-embedded figures
lands in commit 1.10. Conforms to the
:class:`~lmdiff.report._protocols.Renderer` protocol.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lmdiff.geometry import GeoResult


_PLACEHOLDER_BANNER = (
    '<p class="placeholder"><em>v0.3.0 application-tier HTML renderer '
    'arrives in commit 1.10.</em></p>'
)


def render(
    result: "GeoResult",
    *,
    findings: tuple = (),
    tables: dict | None = None,  # noqa: ARG001
    path: str | Path | None = None,
    **_unused,
) -> str:
    """Render a ``GeoResult`` to a self-contained HTML5 string.

    When ``path`` is given, also writes UTF-8 to disk.
    """
    title = (
        f"lmdiff: {escape(result.base_name)} vs "
        f"{escape(', '.join(result.variant_names))}"
    )

    rows: list[str] = []
    if result.magnitudes:
        for v in result.variant_names:
            mag = result.magnitudes.get(v, float("nan"))
            rows.append(
                f"<tr><td>{escape(str(v))}</td>"
                f"<td>{mag:.4f}</td></tr>"
            )
    table_html = (
        '<table><thead><tr><th>variant</th><th>‖δ‖</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        if rows
        else "<p>(no magnitudes)</p>"
    )

    findings_html: str
    if findings:
        items: list[str] = []
        for f in findings:
            sev = escape(str(getattr(f, "severity", "info")))
            summary = escape(str(getattr(f, "summary", str(f))))
            items.append(
                f'<li class="finding-{sev}">'
                f'<span class="severity">[{sev}]</span> {summary}'
                f"</li>"
            )
        findings_html = (
            "<h2>Findings</h2><ul>" + "".join(items) + "</ul>"
        )
    else:
        findings_html = ""

    body = (
        f'<h1>{title}</h1>'
        f'{_PLACEHOLDER_BANNER}'
        f'<p><strong>Probes</strong>: {result.n_probes}</p>'
        f'<h2>Magnitudes</h2>{table_html}'
        f'{findings_html}'
    )

    html = (
        '<!DOCTYPE html>\n'
        '<html lang="en"><head>'
        '<meta charset="utf-8">'
        f'<title>{title}</title>'
        '<style>body{font-family:system-ui,sans-serif;margin:2em;max-width:900px}'
        'table{border-collapse:collapse}td,th{padding:4px 10px;border:1px solid #ccc}'
        '.placeholder{color:#888}</style>'
        '</head><body>'
        f'{body}'
        '</body></html>\n'
    )

    if path is not None:
        Path(path).write_text(html, encoding="utf-8")
    return html


__all__ = ["render"]
