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


def _compose_one_liner(findings: tuple, n_probes: int = 0, n_domains: int = 0) -> str:
    """Compose the Layer-1 one-liner for the terminal renderer.

    Dispatch order (first match wins):

      1. ``DirectionClusterFinding`` + ``DirectionOutlierFinding`` both fire →
         ``"<k> variants align (cluster), 1 stands apart (outlier)."``.
      2. ≥3 ``SpecializationPeakFinding`` fire on distinct domains →
         ``"Each variant acts biggest on a different domain: <peaks>"``.
      3. ``BiggestMoveFinding`` fires alone → echoes its summary.
      4. Generic fallback → ``"Variants compared on N probes across M domains."``.
    """
    from lmdiff._findings import (
        BiggestMoveFinding,
        DirectionClusterFinding,
        DirectionOutlierFinding,
        SpecializationPeakFinding,
    )

    cluster = next(
        (f for f in findings if isinstance(f, DirectionClusterFinding)), None,
    )
    outlier = next(
        (f for f in findings if isinstance(f, DirectionOutlierFinding)), None,
    )
    if cluster is not None and outlier is not None:
        k = len(cluster.details.get("variants", ()))
        return f"{k} variants align (cluster), 1 stands apart (outlier)."

    peaks = [f for f in findings if isinstance(f, SpecializationPeakFinding)]
    if len(peaks) >= 3:
        domains = {p.details.get("domain") for p in peaks}
        if len(domains) >= 3:
            # Per v6 §12.6: each Finding.summary must appear verbatim in
            # every renderer. Layer 1 prints the summaries directly.
            joined = "   ".join(p.summary for p in peaks)
            return f"Each variant acts biggest on a different domain:\n  {joined}"

    biggest = next(
        (f for f in findings if isinstance(f, BiggestMoveFinding)), None,
    )
    if biggest is not None:
        return biggest.summary

    return f"Variants compared on {n_probes} probes across {n_domains} domains."


__all__ = ["build_tables", "render", "_compose_one_liner"]
