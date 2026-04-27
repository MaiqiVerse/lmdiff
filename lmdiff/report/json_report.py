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
from lmdiff.geometry import GeoResult, _compute_share_per_domain

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

    result = GeoResult(**kwargs)

    if sv == "4":
        warnings.warn(
            "GeoResult JSON schema v4 is deprecated since v0.3.0; the v5 "
            "share_per_domain field is being synthesised on the fly. "
            "Re-save with `result.save(path)` (or write_json) to upgrade. "
            "v4 load support will be removed in v0.4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Synthesise v5 share_per_domain from the v4 per-domain magnitudes.
        result.share_per_domain = _compute_share_per_domain(result)

    return result


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
    """Serialize and write to disk."""
    path = Path(path)
    text = to_json(obj, indent=indent)
    path.write_text(text, encoding="utf-8")
