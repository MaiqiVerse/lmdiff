"""Markdown renderer skeleton (commit 1.5).

v0.3.0 ships a minimal Markdown serialiser — heading + magnitudes table
+ findings list + a banner pointing at commit 1.11 where the polished
narrative-style renderer lands. Conforms to the
:class:`~lmdiff.report._protocols.Renderer` protocol via the
module-level :func:`render` function.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lmdiff.geometry import GeoResult


_PLACEHOLDER_BANNER = (
    "> _v0.3.0 application-tier markdown renderer arrives in commit 1.11._"
)


def render(
    result: "GeoResult",
    *,
    findings: tuple = (),
    tables: dict | None = None,  # noqa: ARG001
    path: str | Path | None = None,
    **_unused,
) -> str:
    """Render a ``GeoResult`` to a Markdown string.

    When ``path`` is given, also writes UTF-8 to disk. Always returns the
    string.
    """
    lines: list[str] = []
    title = (
        f"# lmdiff: {result.base_name} vs "
        f"{', '.join(result.variant_names)}"
    )
    lines.append(title)
    lines.append("")
    lines.append(_PLACEHOLDER_BANNER)
    lines.append("")
    lines.append(f"**Probes**: {result.n_probes}")
    lines.append("")

    if result.magnitudes:
        lines.append("## Magnitudes")
        lines.append("")
        lines.append("| variant | ‖δ‖ |")
        lines.append("|---|---|")
        for v in result.variant_names:
            mag = result.magnitudes.get(v, float("nan"))
            lines.append(f"| {v} | {mag:.4f} |")
        lines.append("")

    if findings:
        lines.append("## Findings")
        lines.append("")
        for f in findings:
            sev = getattr(f, "severity", "info")
            summary = getattr(f, "summary", str(f))
            lines.append(f"- **[{sev}]** {summary}")
        lines.append("")

    output = "\n".join(lines).rstrip() + "\n"
    if path is not None:
        Path(path).write_text(output, encoding="utf-8")
    return output


__all__ = ["render"]
