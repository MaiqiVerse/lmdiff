"""JSON serialization for all lmdiff result dataclasses.

Design rules:
- Deterministic output (sorted keys, no nondeterministic floats).
- numpy arrays → lists, torch tensors → error.
- NaN / inf → None (JSON has no standard representation).
- Config.model → string or "<object>" placeholder.
- All top-level outputs include schema_version and generated_at.

Key ordering note (see LESSONS L-018):
- ``to_json`` calls ``json.dumps(d, sort_keys=True, ...)`` so the emitted
  JSON is byte-exact across runs. The side effect is that every JSON
  object's keys come out **alphabetically sorted, not in insertion order**.
- List fields preserve order. In particular, ``GeoResult.change_vectors``
  is a ``{variant: list[float]}`` dict whose inner lists retain the
  prompt/probe order from ChangeGeometry.analyze().
- Dict fields do NOT preserve order after a JSON round-trip. In
  particular, ``GeoResult.per_probe`` is ``{variant: {probe_text: float}}``
  and its inner keys (probe texts) are alphabetical in the emitted JSON.
  Do NOT use ``list(per_probe[v].keys())`` as a proxy for probe order.
- Downstream analysis that needs probe order + probe text simultaneously
  should source both from the original ProbeSet / probe-set JSON and use
  ``change_vectors[v]`` for values.
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from functools import singledispatch
from pathlib import Path
from typing import Any

import numpy as np

from lmdiff.config import Config
from lmdiff.metrics.base import MetricLevel, MetricResult
from lmdiff.tasks.base import EvalResult, TaskResult
from lmdiff.tasks.capability_radar import DomainRadarResult, RadarResult
from lmdiff.diff import DiffReport, FullReport, PairTaskResult
from lmdiff.geometry import (
    GeoResult,
    _compute_overall_normalized_from_pdn,
    _compute_per_domain_normalized,
    _compute_share_per_domain,
)

SCHEMA_VERSION = "5"
"""Current GeoResult on-disk schema. Reader accepts v1-v5; writer emits v5
exclusively. v4 emits DeprecationWarning on load (will hard-fail in v0.4.0).
v5 added ``share_per_domain`` (per-variant per-domain energy shares,
synthesisable from existing v4 data; see L-023 / L-022)."""


def _clean_value(v: Any) -> Any:
    """Recursively clean a value for JSON serialization."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, np.floating):
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.ndarray):
        return [_clean_value(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _clean_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_value(x) for x in v]
    if isinstance(v, MetricLevel):
        return v.value
    if hasattr(v, "__class__") and v.__class__.__name__ == "Tensor":
        raise TypeError(
            "torch.Tensor found in result data — metrics should not "
            "store raw tensors in MetricResult"
        )
    return v


def _config_to_dict(cfg: Config) -> dict[str, Any]:
    model_val = cfg.model if isinstance(cfg.model, str) else "<object>"
    return {
        "adapter": cfg.adapter,
        "context": cfg.context,
        "decode": cfg.decode,
        "display_name": cfg.display_name,
        "dtype": cfg.dtype,
        "model": model_val,
        "name": cfg.name,
        "system_prompt": cfg.system_prompt,
    }


@singledispatch
def to_json_dict(obj: Any) -> dict[str, Any]:
    """Convert a lmdiff result to a JSON-serializable dict."""
    raise TypeError(f"to_json_dict does not support {type(obj).__name__}")


@to_json_dict.register(MetricResult)
def _metric_result(r: MetricResult) -> dict[str, Any]:
    return {
        "details": _clean_value(r.details),
        "level": r.level.value,
        "metadata": _clean_value(r.metadata),
        "name": r.name,
        "value": _clean_value(r.value),
    }


@to_json_dict.register(EvalResult)
def _eval_result(r: EvalResult) -> dict[str, Any]:
    return {
        "correct": r.correct,
        "expected": r.expected,
        "metadata": _clean_value(r.metadata),
        "output": r.output,
        "probe_id": r.probe_id,
        "score": _clean_value(r.score),
    }


@to_json_dict.register(TaskResult)
def _task_result(r: TaskResult) -> dict[str, Any]:
    return {
        "accuracy": _clean_value(r.accuracy),
        "engine_name": r.engine_name,
        "metadata": _clean_value(r.metadata),
        "n_correct": r.n_correct,
        "n_probes": r.n_probes,
        "per_domain": _clean_value(r.per_domain),
        "per_probe": [to_json_dict(p) for p in r.per_probe],
        "probe_set_name": r.probe_set_name,
        "task_name": r.task_name,
    }


@to_json_dict.register(DomainRadarResult)
def _domain_radar_result(r: DomainRadarResult) -> dict[str, Any]:
    return {
        "accuracy": _clean_value(r.accuracy),
        "bd_vs_baseline": _clean_value(r.bd_vs_baseline),
        "domain": r.domain,
        "n_probes": r.n_probes,
    }


@to_json_dict.register(RadarResult)
def _radar_result(r: RadarResult) -> dict[str, Any]:
    return {
        "a_by_domain": {k: to_json_dict(v) for k, v in r.a_by_domain.items()},
        "b_by_domain": (
            {k: to_json_dict(v) for k, v in r.b_by_domain.items()}
            if r.b_by_domain is not None else None
        ),
        "bd_by_domain": _clean_value(r.bd_by_domain),
        "bd_healthy_by_domain": _clean_value(r.bd_healthy_by_domain),
        "degeneracy_rates": _clean_value(r.degeneracy_rates),
        "domains": r.domains,
        "engine_a_name": r.engine_a_name,
        "engine_b_name": r.engine_b_name,
        "metadata": _clean_value(r.metadata),
    }


@to_json_dict.register(PairTaskResult)
def _pair_task_result(r: PairTaskResult) -> dict[str, Any]:
    return {
        "delta_accuracy": _clean_value(r.delta_accuracy),
        "metadata": _clean_value(r.metadata),
        "per_domain_delta": _clean_value(r.per_domain_delta),
        "result_a": to_json_dict(r.result_a),
        "result_b": to_json_dict(r.result_b),
        "task_name": r.task_name,
    }


@to_json_dict.register(DiffReport)
def _diff_report(r: DiffReport) -> dict[str, Any]:
    return {
        "config_a": _config_to_dict(r.config_a),
        "config_b": _config_to_dict(r.config_b),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": _clean_value(r.metadata),
        "results": [to_json_dict(m) for m in r.results],
        "schema_version": SCHEMA_VERSION,
    }


@to_json_dict.register(GeoResult)
def _geo_result(r: GeoResult) -> dict[str, Any]:
    return {
        "avg_tokens_per_probe": list(r.avg_tokens_per_probe) if r.avg_tokens_per_probe else [],
        "base_name": r.base_name,
        "change_vectors": _clean_value(r.change_vectors),
        "cosine_matrix": _clean_value(r.cosine_matrix),
        "delta_means": _clean_value(r.delta_means),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "magnitudes": _clean_value(r.magnitudes),
        "magnitudes_normalized": _clean_value(r.magnitudes_normalized),
        "magnitudes_per_domain_normalized": _clean_value(
            r.magnitudes_per_domain_normalized,
        ),
        "metadata": _clean_value(r.metadata),
        "n_probes": r.n_probes,
        "per_probe": _clean_value(r.per_probe),
        "probe_domains": list(r.probe_domains) if r.probe_domains else [],
        "schema_version": SCHEMA_VERSION,
        "selective_cosine_matrix": _clean_value(r.selective_cosine_matrix),
        "selective_magnitudes": _clean_value(r.selective_magnitudes),
        "share_per_domain": _clean_value(r.share_per_domain),
        "variant_names": list(r.variant_names),
    }


def geo_result_from_json_dict(d: dict[str, Any]) -> GeoResult:
    """Reconstruct a GeoResult from a to_json_dict / json.loads output.

    Accepts schema versions 1 through 5:

    * v1: legacy; decomposition fields empty.
    * v2: populates ``delta_means`` / ``selective_magnitudes`` /
      ``selective_cosine_matrix``.
    * v3: adds ``probe_domains``.
    * v4: adds ``avg_tokens_per_probe`` + ``magnitudes_normalized``;
      emits ``DeprecationWarning`` and synthesises the v5
      ``share_per_domain`` field on the fly. Will hard-fail in v0.4.0.
    * v5: adds ``share_per_domain``; loaded as-is.

    Numeric ``None`` (JSON ``null``) values in cosine / selective cosine
    matrices are restored to ``float('nan')`` so the in-memory result
    behaves identically whether it came from ``analyze()`` or a round-trip.
    """
    sv = str(d.get("schema_version", "1"))
    if sv not in ("1", "2", "3", "4", "5"):
        raise ValueError(f"unsupported GeoResult schema_version: {sv!r}")

    def _nan_of(v: Any) -> float:
        return float("nan") if v is None else float(v)

    def _nan_matrix(m: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, float]]:
        if not m:
            return {}
        return {a: {b: _nan_of(val) for b, val in row.items()} for a, row in m.items()}

    kwargs: dict[str, Any] = dict(
        base_name=d["base_name"],
        variant_names=list(d["variant_names"]),
        n_probes=int(d["n_probes"]),
        magnitudes={k: float(v) for k, v in d["magnitudes"].items()},
        cosine_matrix=_nan_matrix(d["cosine_matrix"]),
        change_vectors={k: [float(x) for x in v] for k, v in d["change_vectors"].items()},
        per_probe={k: {p: float(val) for p, val in row.items()} for k, row in d["per_probe"].items()},
        metadata=dict(d.get("metadata", {})),
    )
    if sv in ("2", "3", "4", "5"):
        kwargs["delta_means"] = {k: float(v) for k, v in d.get("delta_means", {}).items()}
        kwargs["selective_magnitudes"] = {
            k: float(v) for k, v in d.get("selective_magnitudes", {}).items()
        }
        kwargs["selective_cosine_matrix"] = _nan_matrix(d.get("selective_cosine_matrix"))
    if sv in ("3", "4", "5"):
        raw = d.get("probe_domains", [])
        kwargs["probe_domains"] = tuple(raw) if raw else ()
    if sv in ("4", "5"):
        raw_tokens = d.get("avg_tokens_per_probe", [])
        kwargs["avg_tokens_per_probe"] = (
            tuple(float(x) for x in raw_tokens) if raw_tokens else ()
        )
        kwargs["magnitudes_normalized"] = {
            k: float(v) for k, v in d.get("magnitudes_normalized", {}).items()
        }
    if sv == "5":
        raw_share = d.get("share_per_domain", {}) or {}
        kwargs["share_per_domain"] = {
            v: {dom: float(val) for dom, val in row.items()}
            for v, row in raw_share.items()
        }
        # v0.3.2 additive field — present in saves from v0.3.2-post-fix
        # onward, absent in v0.3.0 / v0.3.1 / pre-fix v0.3.2.
        raw_pdn = d.get("magnitudes_per_domain_normalized") or {}
        if raw_pdn:
            kwargs["magnitudes_per_domain_normalized"] = {
                v: {dom: float(val) for dom, val in row.items()}
                for v, row in raw_pdn.items()
            }

    result = GeoResult(**kwargs)

    # v4 → v5: synthesise share_per_domain (legacy path).
    # v0.3.2 share/overall-normalized correction:
    #   Whether sv is "4" (no share at all) or "5" with old length-biased
    #   share, we recompute share + overall + pdn from the raw inputs
    #   when those inputs are available. The old saved values get
    #   overwritten because the v0.3.0–v0.3.2 formulas are corrected.
    _ensure_per_domain_normalized_views(
        result,
        loaded_pdn_was_present=bool(
            sv == "5" and d.get("magnitudes_per_domain_normalized"),
        ),
        from_schema_version=sv,
    )

    return result


def _ensure_per_domain_normalized_views(
    result: GeoResult,
    *,
    loaded_pdn_was_present: bool,
    from_schema_version: str,
) -> None:
    """Recompute pdn / share / overall when the load preceded the
    v0.3.2 formula correction.

    Pre-correction (v0.3.0–v0.3.2) saves used:
      - ``share_per_domain[v][d] = ‖δ_{v|d}‖² / Σ ‖δ_{v|d'}‖²`` (raw,
        length-weighted — long-context domains dominate)
      - ``magnitudes_normalized[v] = raw / sqrt(n_probes·mean_T)``
        (single per-token RMS; under-weights short-prompt domains)

    Post-correction (v0.3.2 fix forward):
      - ``magnitudes_per_domain_normalized[v][d]`` = per-token RMS in d
      - ``share_per_domain[v][d]`` = pdn[v][d]² / Σ pdn²
      - ``magnitudes_normalized[v]`` = sqrt(mean over d of pdn²)

    If the new ``magnitudes_per_domain_normalized`` field was present in
    the JSON, we trust the on-disk values and skip recomputation. If
    it's absent but the inputs (probe_domains + avg_tokens_per_probe)
    are available, recompute and emit a single ``DeprecationWarning``.
    Without inputs, leave the loaded values alone (the bulk overall is
    still meaningful for single-domain experiments).
    """
    if loaded_pdn_was_present:
        return
    if not result.probe_domains or not result.avg_tokens_per_probe:
        return
    pdn = _compute_per_domain_normalized(
        result.variant_names,
        result.change_vectors,
        result.probe_domains,
        result.avg_tokens_per_probe,
    )
    if not pdn:
        return
    result.magnitudes_per_domain_normalized = pdn
    result.share_per_domain = _compute_share_per_domain(result)
    result.magnitudes_normalized = _compute_overall_normalized_from_pdn(pdn)
    extra = (
        " v4 load support will be removed in v0.4.0."
        if from_schema_version == "4" else ""
    )
    warnings.warn(
        f"loaded GeoResult (schema v{from_schema_version}) was saved before "
        "the v0.3.2 share_per_domain / overall-normalized formula correction. "
        "Recomputed magnitudes_per_domain_normalized + share_per_domain + "
        "magnitudes_normalized using per-domain per-token formulas — "
        "long-context-heavy probe sets see substantially different shares "
        "(this is the corrected behavior, matching v6 §13). "
        f"Re-save with result.save(path) to upgrade the on-disk file.{extra}",
        DeprecationWarning,
        stacklevel=3,
    )


@to_json_dict.register(FullReport)
def _full_report(r: FullReport) -> dict[str, Any]:
    return {
        "config_a": _config_to_dict(r.config_a),
        "config_b": _config_to_dict(r.config_b),
        "diff_report": to_json_dict(r.diff_report) if r.diff_report else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": _clean_value(r.metadata),
        "radar_result": to_json_dict(r.radar_result) if r.radar_result else None,
        "schema_version": SCHEMA_VERSION,
        "task_results": [to_json_dict(t) for t in r.task_results],
    }


def to_json(obj: Any, indent: int = 2) -> str:
    """Serialize any lmdiff result to a JSON string."""
    d = to_json_dict(obj)
    return json.dumps(d, indent=indent, sort_keys=True, ensure_ascii=False)


def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Serialize and write to disk. Auto-creates parent directories."""
    path = Path(path)
    text = to_json(obj, indent=indent)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ── v0.3.0 Renderer Protocol adapter (commit 1.5) ─────────────────────


def render(
    result: Any,
    *,
    findings: tuple = (),  # noqa: ARG001  reserved for commit 1.6 emission
    tables: dict | None = None,  # noqa: ARG001
    path: str | Path | None = None,
    indent: int = 2,
    **_unused,
) -> dict[str, Any]:
    """Render a GeoResult / DiffReport / FullReport as a v5 JSON dict.

    When ``path`` is given, the dict is also written to disk (UTF-8,
    sort_keys=True). Returns the dict either way.
    """
    payload = to_json_dict(result)
    if path is not None:
        text = json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=False)
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    return payload


def load_result(path: str | Path) -> "GeoResult":
    """Load a GeoResult JSON file (any schema 1-5).

    v4 emits ``DeprecationWarning`` and synthesises the v5
    ``share_per_domain`` field; v5 loads as-is.
    """
    text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    return geo_result_from_json_dict(payload)
