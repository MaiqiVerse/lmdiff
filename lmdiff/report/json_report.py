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

SCHEMA_VERSION = "6"
"""Current GeoResult on-disk schema (v0.4.1+).

Reader accepts v1-v6; writer emits v6 exclusively. Per-version notes:
  v1-v4: legacy formats handled by the existing upgrade path; load
    emits ``DeprecationWarning``.
  v5 (v0.3.2 - v0.4.0): values **preserved as saved** on load per
    Q9.8 (saved means saved). Loader synthesizes empty
    ``probe_validity`` and full-status ``domain_status`` to honor
    the v6 schema shape but does NOT recompute the saved
    ``share_per_domain`` / ``magnitudes_per_domain_normalized`` —
    those still reflect the pre-v0.4.1 √T̄ formula, with a
    DeprecationWarning advising re-run for v0.4.1 numerics.
  v6 (v0.4.1+): full schema. Adds ``probe_validity``,
    ``domain_status``, ``variant_only_metrics`` (stub). Field
    semantics: ``share_per_domain[v][d]`` and
    ``magnitudes_per_domain_normalized[v][d]`` may be ``None`` for
    out_of_range / variant_only domains."""


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


def _engine_validity_to_dict(ev: Any) -> dict[str, Any]:
    """Serialize an EngineValidity dataclass. Defensive against partially
    populated (legacy / synthesized) records — every field has a sane
    fallback."""
    return {
        "engine_name": getattr(ev, "engine_name", ""),
        "max_context": getattr(ev, "max_context", None),
        "T_i": int(getattr(ev, "T_i", 0)),
        "is_valid": bool(getattr(ev, "is_valid", True)),
        "reason": getattr(ev, "reason", "unknown_limit"),
    }


def _probe_validity_to_dict(pv: Any) -> dict[str, Any]:
    """Serialize a ProbeValidity dataclass."""
    per_engine = getattr(pv, "per_engine", {}) or {}
    return {
        "probe_id": getattr(pv, "probe_id", ""),
        "domain": getattr(pv, "domain", None),
        "per_engine": {
            k: _engine_validity_to_dict(v) for k, v in per_engine.items()
        },
    }


@to_json_dict.register(GeoResult)
def _geo_result(r: GeoResult) -> dict[str, Any]:
    payload = {
        "avg_tokens_per_probe": list(r.avg_tokens_per_probe) if r.avg_tokens_per_probe else [],
        "base_name": r.base_name,
        "change_vectors": _clean_value(r.change_vectors),
        "cosine_matrix": _clean_value(r.cosine_matrix),
        "delta_means": _clean_value(r.delta_means),
        "domain_status": _clean_value(r.domain_status),
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
        "probe_validity": {
            pid: _probe_validity_to_dict(pv)
            for pid, pv in (r.probe_validity or {}).items()
        },
        "schema_version": SCHEMA_VERSION,
        "selective_cosine_matrix": _clean_value(r.selective_cosine_matrix),
        "selective_magnitudes": _clean_value(r.selective_magnitudes),
        "share_per_domain": _clean_value(r.share_per_domain),
        "variant_names": list(r.variant_names),
        "variant_only_metrics": (
            _clean_value(r.variant_only_metrics)
            if r.variant_only_metrics is not None else None
        ),
    }
    return payload


def geo_result_from_json_dict(d: dict[str, Any]) -> GeoResult:
    """Reconstruct a GeoResult from a to_json_dict / json.loads output.

    Accepts schema versions 1 through 6:

    * v1: legacy; decomposition fields empty.
    * v2: populates ``delta_means`` / ``selective_magnitudes`` /
      ``selective_cosine_matrix``.
    * v3: adds ``probe_domains``.
    * v4: adds ``avg_tokens_per_probe`` + ``magnitudes_normalized``;
      emits ``DeprecationWarning`` and synthesises the v5
      ``share_per_domain`` field on the fly.
    * v5 (v0.3.2 - v0.4.0): pre-v0.4.1 ``share_per_domain`` /
      ``magnitudes_per_domain_normalized`` values **preserved as
      saved** (Q9.8 — saved means saved). Loader synthesizes empty
      ``probe_validity`` and full-status ``domain_status`` to satisfy
      the v6 schema, but does NOT recompute the saved pdn / share
      with the v0.4.1 formula. Emits a ``DeprecationWarning``
      directing the user to re-run for v0.4.1 numerics.
    * v6 (v0.4.1+): full schema. ``probe_validity`` /
      ``domain_status`` / ``variant_only_metrics`` deserialized.
      ``share_per_domain`` / ``magnitudes_per_domain_normalized`` may
      contain ``None`` for invalid domains.

    Numeric ``None`` (JSON ``null``) values in cosine / selective cosine
    matrices are restored to ``float('nan')`` so the in-memory result
    behaves identically whether it came from ``analyze()`` or a round-trip.
    """
    sv = str(d.get("schema_version", "1"))
    if sv not in ("1", "2", "3", "4", "5", "6"):
        raise ValueError(f"unsupported GeoResult schema_version: {sv!r}")

    def _nan_of(v: Any) -> float:
        return float("nan") if v is None else float(v)

    def _nan_matrix(m: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, float]]:
        if not m:
            return {}
        return {a: {b: _nan_of(val) for b, val in row.items()} for a, row in m.items()}

    def _nullable_float_matrix(
        m: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, float | None]]:
        """For v0.4.1 share / pdn — values may be JSON null (sentinel
        for out_of_range / variant_only). Preserve as Python None."""
        if not m:
            return {}
        out: dict[str, dict[str, float | None]] = {}
        for k, row in m.items():
            row_out: dict[str, float | None] = {}
            for sk, val in row.items():
                row_out[sk] = None if val is None else float(val)
            out[k] = row_out
        return out

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
    if sv in ("2", "3", "4", "5", "6"):
        kwargs["delta_means"] = {k: float(v) for k, v in d.get("delta_means", {}).items()}
        kwargs["selective_magnitudes"] = {
            k: float(v) for k, v in d.get("selective_magnitudes", {}).items()
        }
        kwargs["selective_cosine_matrix"] = _nan_matrix(d.get("selective_cosine_matrix"))
    if sv in ("3", "4", "5", "6"):
        raw = d.get("probe_domains", [])
        kwargs["probe_domains"] = tuple(raw) if raw else ()
    if sv in ("4", "5", "6"):
        raw_tokens = d.get("avg_tokens_per_probe", [])
        kwargs["avg_tokens_per_probe"] = (
            tuple(float(x) for x in raw_tokens) if raw_tokens else ()
        )
        kwargs["magnitudes_normalized"] = {
            k: float(v) for k, v in d.get("magnitudes_normalized", {}).items()
        }
    if sv in ("5", "6"):
        # share_per_domain & pdn may carry None values starting v6;
        # _nullable_float_matrix handles both v5 (always-float) and
        # v6 (float | None).
        kwargs["share_per_domain"] = _nullable_float_matrix(
            d.get("share_per_domain", {}) or {},
        )
        raw_pdn = d.get("magnitudes_per_domain_normalized") or {}
        if raw_pdn:
            kwargs["magnitudes_per_domain_normalized"] = (
                _nullable_float_matrix(raw_pdn)
            )

    if sv == "6":
        # v0.4.1 fields. validity records use the dataclasses from
        # lmdiff._validity; reconstruct them with float coercion.
        from lmdiff._validity import EngineValidity, ProbeValidity

        raw_pv = d.get("probe_validity") or {}
        probe_validity: dict[str, ProbeValidity] = {}
        for pid, pv_dict in raw_pv.items():
            per_eng: dict[str, EngineValidity] = {}
            for ename, ev_dict in (pv_dict.get("per_engine") or {}).items():
                per_eng[ename] = EngineValidity(
                    engine_name=ev_dict.get("engine_name", ename),
                    max_context=ev_dict.get("max_context"),
                    T_i=int(ev_dict.get("T_i", 0)),
                    is_valid=bool(ev_dict.get("is_valid", True)),
                    reason=ev_dict.get("reason", "unknown_limit"),
                )
            probe_validity[pid] = ProbeValidity(
                probe_id=pv_dict.get("probe_id", pid),
                domain=pv_dict.get("domain"),
                per_engine=per_eng,
            )
        kwargs["probe_validity"] = probe_validity
        kwargs["domain_status"] = {
            v: dict(row) for v, row in (d.get("domain_status") or {}).items()
        }
        vom = d.get("variant_only_metrics")
        kwargs["variant_only_metrics"] = vom  # nullable dict, kept as-is

    result = GeoResult(**kwargs)

    # Legacy upgrade paths: v1-v4 didn't have share / pdn at all → must
    # synthesize. v5 → v6: per Q9.8, PRESERVE the saved values; only
    # synthesize the validity stubs that v5 didn't have. v0.4.1 formula
    # numerics are NOT applied to v5 saves.
    if sv in ("5", "6"):
        if sv == "5":
            _stub_validity_for_v5_load(result)
            warnings.warn(
                "loaded GeoResult schema v5 (v0.3.2 - v0.4.0); values use "
                "the pre-v0.4.1 formula (sqrt(Σδ²/ΣT), dimensionally "
                "inconsistent — see L-033). share_per_domain and "
                "magnitudes_per_domain_normalized preserved as saved per "
                "v0.4.1 Q9.8 (saved means saved). Re-run with v0.4.1+ for "
                "corrected numerics + per-probe validity records. See "
                "docs/methodology/normalization.md and "
                "docs/migration/v040-to-v041.md.",
                DeprecationWarning,
                stacklevel=3,
            )
        # v6: no recompute, no warning — values are already v0.4.1-formula.
    else:
        # v1-v4: pre-v5 saves had no share / pdn. Synthesize them — this
        # uses the v0.4.1 formula, which is the only formula remaining
        # in the codebase. The legacy DeprecationWarning still fires.
        _ensure_per_domain_normalized_views(
            result,
            loaded_pdn_was_present=False,
            from_schema_version=sv,
        )

    return result


def _stub_validity_for_v5_load(result: GeoResult) -> None:
    """Synthesize v0.4.1 validity fields on a v5-loaded GeoResult.

    v5 saves predate the validity framework. The v0.4.1 schema requires
    ``probe_validity`` and ``domain_status`` to be present (the
    GeoResult dataclass defaults to empty dicts). For consumers that
    iterate ``domain_status`` or call ``share_per_domain`` accessors,
    we populate ``domain_status`` with ``"full"`` for every (variant,
    domain) pair the saved data has — matches the legacy assumption
    "all probes valid for all engines." ``probe_validity`` stays empty
    (no per-probe records to reconstruct from a v5 save).
    """
    if result.probe_domains and result.variant_names:
        domains = sorted({d for d in result.probe_domains if d is not None})
        result.domain_status = {
            v: {d: "full" for d in domains}
            for v in result.variant_names
        }
    result.variant_only_metrics = None


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
    warnings.warn(
        f"loaded GeoResult (schema v{from_schema_version}) predates v0.4.1; "
        "magnitudes_per_domain_normalized + share_per_domain + "
        "magnitudes_normalized synthesized using the v0.4.1 formula "
        "(plain unweighted RMS over valid probes, Q9.10 Formula A). "
        "validity records (probe_validity, domain_status) cannot be "
        "reconstructed — re-run with v0.4.1+ for accurate validity "
        "classification. See docs/methodology/normalization.md and L-033.",
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
