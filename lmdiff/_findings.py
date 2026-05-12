"""Findings extraction (commit 1.6).

Eight finding type dataclasses + the rule logic that extracts them from
a :class:`~lmdiff.geometry.GeoResult`. Single source of truth for
narrative content that all renderers (commits 1.7-1.10) consume — the
``summary`` text is identical across channels; only presentation differs.

Stable extraction order::

  1. info findings
     a. MostLikeBaseFinding
     b. BiggestMoveFinding
     c. DirectionClusterFinding (≥3 variants only)
     d. DirectionOutlierFinding (only if cluster fired)
     e. SpecializationPeakFinding (one per variant, alphabetical)
  2. warning findings
     - TokenizerMismatchFinding
  3. caveat findings
     - AccuracyArtifactFinding (one per affected task)
     - BaseAccuracyMissingFinding

Renderers print findings in the order this module returns them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from lmdiff.geometry import GeoResult


__all__ = [
    "Finding",
    "MostLikeBaseFinding",
    "BiggestMoveFinding",
    "DirectionClusterFinding",
    "DirectionOutlierFinding",
    "SpecializationPeakFinding",
    "AccuracyArtifactFinding",
    "TokenizerMismatchFinding",
    "BaseAccuracyMissingFinding",
    "extract_findings",
]


# ── Severity-tunable thresholds ───────────────────────────────────────


_CLUSTER_COSINE_THRESHOLD = 0.90
_OUTLIER_COSINE_THRESHOLD = 0.85
_SPECIALIZATION_PEAK_SHARE_THRESHOLD = 0.30
_GENERATIVE_TASKS = frozenset({
    "gsm8k",
    "longbench_2wikimqa",
    "longbench_hotpotqa",
    "longbench_narrativeqa",
    "longbench_qasper",
})
_ACCURACY_ARTIFACT_MAX_NEW_TOKENS = 32
_ACCURACY_ARTIFACT_THRESHOLD = 0.05


# ── Finding base + 8 types ────────────────────────────────────────────


@dataclass(frozen=True)
class Finding:
    """Base for all findings.

    All concrete subclasses share the same field shape; the type itself
    discriminates them. Frozen so findings can be cached on a
    :class:`~lmdiff.geometry.GeoResult` instance.
    """

    severity: Literal["info", "caveat", "warning"]
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MostLikeBaseFinding(Finding):
    """The (variant, domain) cell with the smallest per-domain drift."""


@dataclass(frozen=True)
class BiggestMoveFinding(Finding):
    """The (variant, domain) cell with the largest per-domain drift."""


@dataclass(frozen=True)
class DirectionClusterFinding(Finding):
    """≥3 variants whose pairwise raw cosines all exceed 0.90."""


@dataclass(frozen=True)
class DirectionOutlierFinding(Finding):
    """A variant with cosine < 0.85 to every member of the cluster."""


@dataclass(frozen=True)
class SpecializationPeakFinding(Finding):
    """A variant whose share_per_domain peak exceeds 30%."""


@dataclass(frozen=True)
class AccuracyArtifactFinding(Finding):
    """A generative-task accuracy is suspiciously near 0 with a low
    max_new_tokens — likely truncation artifact."""


@dataclass(frozen=True)
class TokenizerMismatchFinding(Finding):
    """Cross-tokenizer comparison detected; some metrics unavailable."""


@dataclass(frozen=True)
class BaseAccuracyMissingFinding(Finding):
    """Variant accuracies present but base accuracy missing — Δacc skipped."""


# ── Helpers ───────────────────────────────────────────────────────────


def _per_domain_drift(result: "GeoResult") -> dict[str, dict[str, float]]:
    """Per-variant per-domain drift (raw L2 magnitude)."""
    if not result.probe_domains:
        return {}
    try:
        return result.domain_heatmap()
    except (ValueError, AttributeError):
        return {}


def _flatten_drift_cells(
    drift: dict[str, dict[str, float]],
) -> list[tuple[str, str, float]]:
    """Flatten ``{variant: {domain: drift}}`` to a sortable triple list."""
    out: list[tuple[str, str, float]] = []
    for variant, per_dom in drift.items():
        for domain, value in per_dom.items():
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            if fv != fv:  # NaN
                continue
            out.append((variant, domain, fv))
    return out


# ── Rule extractors ───────────────────────────────────────────────────


def _extract_most_like_base(result: "GeoResult") -> list[Finding]:
    cells = _flatten_drift_cells(_per_domain_drift(result))
    if not cells:
        return []
    variant, domain, value = min(cells, key=lambda t: (t[2], t[0], t[1]))
    summary = f"{variant} on {domain} (drift {value:.4f})"
    return [
        MostLikeBaseFinding(
            severity="info",
            summary=summary,
            details={"variant": variant, "domain": domain, "drift": value},
        )
    ]


def _extract_biggest_move(result: "GeoResult") -> list[Finding]:
    cells = _flatten_drift_cells(_per_domain_drift(result))
    if not cells:
        return []
    variant, domain, value = max(cells, key=lambda t: (t[2], t[0], t[1]))
    summary = f"{variant} on {domain} (drift {value:.4f})"
    return [
        BiggestMoveFinding(
            severity="info",
            summary=summary,
            details={"variant": variant, "domain": domain, "drift": value},
        )
    ]


def _extract_direction_cluster_and_outlier(
    result: "GeoResult",
) -> list[Finding]:
    """Cluster needs ≥3 variants with all pairwise raw cosines > 0.90.
    Outlier needs the cluster to fire AND a variant with cos < 0.85 to all
    cluster members."""
    variants = list(result.variant_names)
    if len(variants) < 3:
        return []

    cos = result.cosine_matrix

    def _safe_cos(a: str, b: str) -> float:
        try:
            v = float(cos[a][b])
        except (KeyError, TypeError, ValueError):
            return float("nan")
        return v

    # Greedy: take the largest k-clique above threshold (k from len downward).
    cluster: list[str] | None = None
    for k in range(len(variants), 2, -1):
        for combo in combinations(variants, k):
            ok = True
            for a, b in combinations(combo, 2):
                v = _safe_cos(a, b)
                if v != v or v <= _CLUSTER_COSINE_THRESHOLD:
                    ok = False
                    break
            if ok:
                cluster = list(combo)
                break
        if cluster is not None:
            break
    if cluster is None:
        return []

    # Mean cosine within the cluster.
    pair_vals = [_safe_cos(a, b) for a, b in combinations(cluster, 2)]
    avg_cos = sum(pair_vals) / len(pair_vals)
    cluster_set = tuple(sorted(cluster))
    cluster_summary = f"{{{', '.join(cluster_set)}}} agree (cos ~{avg_cos:.2f})"
    findings: list[Finding] = [
        DirectionClusterFinding(
            severity="info",
            summary=cluster_summary,
            details={
                "variants": cluster_set,
                "mean_cosine": avg_cos,
                "method": "pairwise_threshold_0.90",
            },
        )
    ]

    # Outlier: variants outside the cluster with cos < 0.85 to ALL members.
    cluster_members = set(cluster)
    for v in variants:
        if v in cluster_members:
            continue
        pair_to_cluster = [_safe_cos(v, c) for c in cluster]
        finite = [p for p in pair_to_cluster if p == p]
        if not finite or any(p >= _OUTLIER_COSINE_THRESHOLD for p in finite):
            continue
        mean_cos = sum(finite) / len(finite)
        findings.append(
            DirectionOutlierFinding(
                severity="info",
                summary=f"{v} stands apart (cos ~{mean_cos:.2f} to cluster)",
                details={
                    "variant": v,
                    "mean_cosine_to_cluster": mean_cos,
                    "cluster": cluster_set,
                },
            )
        )
    return findings


def _extract_specialization_peaks(result: "GeoResult") -> list[Finding]:
    if not result.share_per_domain:
        return []
    out: list[Finding] = []
    for variant in sorted(result.share_per_domain):
        per_dom = result.share_per_domain[variant]
        if not per_dom:
            continue
        # v0.4.1: skip None entries (out_of_range / variant_only) — they
        # carry no share signal and ``max`` over a None vs float raises.
        valid = [(d, s) for d, s in per_dom.items() if s is not None]
        if not valid:
            continue
        peak_dom, peak_share = max(valid, key=lambda kv: kv[1])
        if peak_share <= _SPECIALIZATION_PEAK_SHARE_THRESHOLD:
            continue
        summary = f"{variant}: {peak_share * 100:.0f}% on {peak_dom}"
        out.append(
            SpecializationPeakFinding(
                severity="info",
                summary=summary,
                details={
                    "variant": variant,
                    "domain": peak_dom,
                    "share": peak_share,
                },
            )
        )
    return out


def _extract_tokenizer_mismatch(result: "GeoResult") -> list[Finding]:
    """Look at metadata for explicit tokenizer-id pairs.

    The v0.2.x bridge writes ``bpb_normalized: {variant: bool}`` into
    metadata when tokenizers differ. Use that as the signal for v0.3.0.
    """
    meta = result.metadata or {}
    bpb_flags = meta.get("bpb_normalized") or {}
    if not isinstance(bpb_flags, dict):
        return []
    mismatched_pairs = [
        (result.base_name, v) for v, flag in bpb_flags.items() if flag
    ]
    if not mismatched_pairs:
        return []
    return [
        TokenizerMismatchFinding(
            severity="warning",
            summary="Cross-tokenizer comparison; representation metrics unavailable",
            details={"pairs": mismatched_pairs},
        )
    ]


def _effective_max_new_tokens(meta: dict, task: str) -> int | None:
    """Resolve the max_new_tokens that was actually used for ``task``.

    Per spec invariant #6: a per-task override (``task_max_new_tokens``
    in metadata) wins over the global ``max_new_tokens`` so the rule
    doesn't fire on tasks the user already fixed.
    """
    overrides = meta.get("task_max_new_tokens") or {}
    if isinstance(overrides, dict) and task in overrides:
        try:
            return int(overrides[task])
        except (TypeError, ValueError):
            pass
    g = meta.get("max_new_tokens")
    if g is None:
        return None
    try:
        return int(g)
    except (TypeError, ValueError):
        return None


def _extract_accuracy_findings(result: "GeoResult") -> list[Finding]:
    """AccuracyArtifactFinding (caveat) + BaseAccuracyMissingFinding (caveat)."""
    meta = result.metadata or {}
    acc_by_variant = meta.get("accuracy_by_variant") or {}
    if not isinstance(acc_by_variant, dict) or not acc_by_variant:
        return []

    findings: list[Finding] = []

    # AccuracyArtifactFinding — one per (task) where every variant ~0
    # and effective max_new_tokens is suspiciously low.
    tasks_seen: set[str] = set()
    for per_task in acc_by_variant.values():
        if isinstance(per_task, dict):
            tasks_seen.update(per_task.keys())

    for task in sorted(tasks_seen):
        if task not in _GENERATIVE_TASKS:
            continue
        mnt = _effective_max_new_tokens(meta, task)
        if mnt is None or mnt > _ACCURACY_ARTIFACT_MAX_NEW_TOKENS:
            continue
        accs: list[float] = []
        for per_task in acc_by_variant.values():
            if not isinstance(per_task, dict):
                continue
            v = per_task.get(task)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != fv:  # NaN
                continue
            accs.append(fv)
        if not accs:
            continue
        if max(accs) > _ACCURACY_ARTIFACT_THRESHOLD:
            continue
        findings.append(
            AccuracyArtifactFinding(
                severity="caveat",
                summary=f"{task} accuracy ~0 likely max_new_tokens={mnt} artifact",
                details={
                    "task": task,
                    "accuracy_by_variant": {
                        k: v.get(task) for k, v in acc_by_variant.items()
                        if isinstance(v, dict) and task in v
                    },
                    "max_new_tokens": mnt,
                    "suggestion": (
                        "Re-run with --task-max-new-tokens "
                        f"{task}=128 (or higher)"
                    ),
                },
            )
        )

    # BaseAccuracyMissingFinding — variants have accuracy data, base does not.
    base_acc = meta.get("base_accuracy") or meta.get("accuracy_base")
    variants_with_accuracy = [
        k for k, v in acc_by_variant.items()
        if isinstance(v, dict) and v
    ]
    if variants_with_accuracy and not base_acc:
        findings.append(
            BaseAccuracyMissingFinding(
                severity="caveat",
                summary="Base accuracy not measured; Δaccuracy comparison skipped",
                details={"variants_with_accuracy": variants_with_accuracy},
            )
        )

    return findings


# ── Top-level entry ───────────────────────────────────────────────────


def extract_findings(result: "GeoResult") -> tuple[Finding, ...]:
    """Run all 8 rules against ``result``. See module docstring for order."""
    findings: list[Finding] = []
    findings.extend(_extract_most_like_base(result))
    findings.extend(_extract_biggest_move(result))
    findings.extend(_extract_direction_cluster_and_outlier(result))
    findings.extend(_extract_specialization_peaks(result))
    findings.extend(_extract_tokenizer_mismatch(result))
    findings.extend(_extract_accuracy_findings(result))
    return tuple(findings)
