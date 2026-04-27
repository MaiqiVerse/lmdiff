"""Standard render pipeline (commit 1.5).

::

    GeoResult → findings → tables → renderer.render(result, ...)

``findings`` come from :func:`lmdiff._findings.extract_findings` once
commit 1.6 lands. Until then, the pipeline passes an empty tuple so the
skeleton renderers don't crash.

``tables`` is a renderer-agnostic dict of standard views (drift, share,
direction, zscore, accuracy). Each renderer formats the entries as
appropriate. Numeric values are not formatted here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lmdiff.geometry import GeoResult


_VALID_CHANNELS = ("terminal", "markdown", "html", "json", "figures")


def build_tables(result: "GeoResult") -> dict[str, Any]:
    """Produce the standard renderer-agnostic table views.

    Keys (always present, may be empty when the underlying GeoResult
    lacks the prerequisite fields):

      ``magnitudes``       — ``dict[variant, float]`` (raw ‖δ‖)
      ``magnitudes_norm``  — ``dict[variant, float]`` (per-token-normalised)
      ``cosine``           — ``dict[variant, dict[variant, float]]`` (raw)
      ``selective_cosine`` — same shape as ``cosine``, mean-removed
      ``share``            — ``dict[variant, dict[domain, float]]`` (rows
        sum to 1)
      ``zscore``           — ``dict[variant, dict[domain, float]]`` row-z
        of per-domain normalised magnitudes (commit 1.4 → v5)
      ``accuracy``         — ``dict[variant, dict[task, float]]`` from
        ``result.metadata.get("accuracy_by_variant", {})``
    """
    out: dict[str, Any] = {
        "magnitudes": dict(result.magnitudes),
        "magnitudes_norm": dict(result.magnitudes_normalized),
        "cosine": result.cosine_matrix,
        "selective_cosine": result.selective_cosine_matrix,
        "share": dict(result.share_per_domain),
        "zscore": {},
        "accuracy": result.metadata.get("accuracy_by_variant", {}),
    }
    if result.probe_domains:
        try:
            out["zscore"] = result.magnitudes_specialization_zscore()
        except (ValueError, AttributeError):
            out["zscore"] = {}
    return out


def render(result: "GeoResult", channel: str, **kwargs: Any) -> Any:
    """Top-level entry: dispatch to a named renderer with shared
    pre-processing.

    Channels: ``"terminal"``, ``"markdown"``, ``"html"``, ``"json"``,
    ``"figures"``. Unknown channels raise ``ValueError`` with the valid
    list.
    """
    findings = _try_extract_findings(result)
    tables = build_tables(result)

    renderer = _get_renderer(channel)
    return renderer.render(result, findings=findings, tables=tables, **kwargs)


def _try_extract_findings(result: "GeoResult") -> tuple:
    """Extract findings if the findings module is available; else ``()``.

    Defers the import so ``lmdiff.report`` can ship without findings
    landing — useful during the 1.5 → 1.6 transition. Once 1.6 lands the
    import always succeeds and the empty-tuple fallback never fires.
    """
    try:
        from lmdiff._findings import extract_findings
    except ImportError:
        return ()
    return extract_findings(result)


def _get_renderer(channel: str):
    """Look up a renderer module by channel name."""
    if channel not in _VALID_CHANNELS:
        raise ValueError(
            f"unknown render channel {channel!r}. Valid channels: "
            f"{list(_VALID_CHANNELS)}"
        )
    if channel == "terminal":
        from lmdiff.report import terminal as mod
    elif channel == "markdown":
        from lmdiff.report import markdown as mod
    elif channel == "html":
        from lmdiff.report import html as mod
    elif channel == "json":
        from lmdiff.report import json_report as mod
    elif channel == "figures":
        from lmdiff.report import figures as mod
    return mod


__all__ = ["build_tables", "render"]
