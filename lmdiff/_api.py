"""Top-level v0.3.0 user API: ``compare`` and ``family``.

These are the entry points the rest of v0.3.0 hangs off. They glue
:class:`~lmdiff._config.Config` (PR #1) and :class:`~lmdiff._engine.Engine`
(PR #2) into a one-call workflow.

Conventions:
  - String arguments coerce to ``Config(model=str)``.
  - When the caller doesn't pass ``engine=``, this module constructs an
    :class:`~lmdiff._engine.HFEngine` per variant and **closes them in a
    finally block** — leaks would be 7B-class.
  - When the caller does pass ``engine=``, it is treated as a *template*:
    per-variant engines are built via ``template.with_config(cfg)``. Caller
    owns the template; this module never closes it.
  - Capability negotiation runs **before** any inference: a capability
    mismatch raises :class:`~lmdiff._engine.CapabilityError` immediately,
    not after spending compute on probes.

This module does NOT load torch at import time. ``_geometry`` /
``_engine`` imports happen inside the function bodies.
"""
from __future__ import annotations

from typing import Any, Optional, Union

from lmdiff._config import Config, DecodeSpec
from lmdiff._engine import CapabilityError, Engine, RESERVED_CAPABILITIES

__all__ = ["compare", "family"]


# v0.3.0 default-metric whitelist. The names below match what the existing
# v0.2.x ChangeGeometry pipeline already computes; Phase 4's metric registry
# replaces this hardcoded list. Capability requirements per metric are also
# hardcoded here for the same reason.
_V03_DEFAULT_METRICS: frozenset[str] = frozenset({
    "bd",
    "drift",
    "share",
    "direction",
    "specialization_zscore",
})

# Per-metric capability requirements. Phase 4 moves this into the metric
# class definitions; v0.3.0 keeps it here so we can enforce up-front.
_METRIC_CAPABILITY_REQUIREMENTS: dict[str, frozenset[str]] = {
    "bd": frozenset({"score", "generate"}),
    "drift": frozenset({"score", "generate"}),
    "share": frozenset({"score", "generate"}),
    "direction": frozenset({"score", "generate"}),
    "specialization_zscore": frozenset({"score", "generate"}),
}


# ── Coercion helpers ──────────────────────────────────────────────────


def _coerce_to_config(value: Union[str, Config]) -> Config:
    """Accept a model-id string or a Config; return a Config."""
    if isinstance(value, Config):
        return value
    if isinstance(value, str):
        return Config(model=value)
    raise TypeError(
        f"expected str or lmdiff.Config, got {type(value).__name__}"
    )


def _coerce_to_probe_set(probes: Union[str, "Any", None]) -> "Any":
    """Resolve the ``probes`` argument to a ``ProbeSet`` instance.

    Supported forms:
      - ``None``                              → bundled ``v01``
      - ``"v01"``                             → bundled ``v01.json``
      - ``"lm_eval:hellaswag+arc_challenge"`` → load+concat via the
        ``from_lm_eval`` adapter
      - ``ProbeSet`` instance                 → returned unchanged

    File-path probes / YAML / the new ``task_type``-aware ``ProbeSet`` land
    in commit 1.4. For commit 1.1 we re-use the v0.2.x probe handling.
    """
    from lmdiff.probes.loader import ProbeSet

    if probes is None or probes == "v01":
        from pathlib import Path
        v01_path = Path(__file__).parent / "probes" / "v01.json"
        return ProbeSet.from_json(v01_path)
    if isinstance(probes, ProbeSet):
        return probes
    if isinstance(probes, str):
        if probes.startswith("lm_eval:"):
            from lmdiff.probes.adapters import from_lm_eval
            tail = probes.split(":", 1)[1]
            task_names = [t.strip() for t in tail.split("+") if t.strip()]
            if not task_names:
                raise ValueError(
                    f"lm_eval probe spec {probes!r} contains no task names"
                )
            sets = [from_lm_eval(t) for t in task_names]
            from lmdiff.probes.loader import Probe
            merged_probes: list[Probe] = []
            for ps in sets:
                merged_probes.extend(ps)
            return ProbeSet(
                merged_probes,
                name=f"lm_eval:{'+'.join(task_names)}",
                version="lm-eval-harness",
            )
        # Bundled name fallback (e.g., a future "v02"):
        from pathlib import Path
        bundled = Path(__file__).parent / "probes" / f"{probes}.json"
        if bundled.exists():
            return ProbeSet.from_json(bundled)
        raise ValueError(
            f"unrecognized probes spec {probes!r}; pass None for the bundled "
            f"v01 set, an 'lm_eval:<task>+<task>' spec, or a ProbeSet instance"
        )
    raise TypeError(
        f"probes must be None, str, or ProbeSet; got {type(probes).__name__}"
    )


# ── Metric resolution + capability negotiation ────────────────────────


def _resolve_metrics(metrics: Union[str, list[str]]) -> list[str]:
    """Translate the ``metrics`` argument into a list of v0.3.0 metric names.

    ``"default"`` expands to ``_V03_DEFAULT_METRICS`` in canonical order.
    Names containing ``lmdiff.contrib.`` raise ``NotImplementedError``
    pointing at Phase 6. Unknown names raise ``ValueError``.
    """
    if metrics == "default" or metrics is None:
        return ["bd", "drift", "share", "direction", "specialization_zscore"]
    if isinstance(metrics, str):
        metrics = [metrics]
    if not isinstance(metrics, list):
        raise TypeError(
            f"metrics must be 'default', a str, or a list of str; "
            f"got {type(metrics).__name__}"
        )

    resolved: list[str] = []
    for name in metrics:
        if not isinstance(name, str):
            raise TypeError(
                f"metric names must be str; got {type(name).__name__}"
            )
        if name.startswith("lmdiff.contrib."):
            raise NotImplementedError(
                f"experimental metric {name!r} is not available until Phase 6. "
                f"v0.3.0 supports the default metric set; pass "
                f"metrics='default' or a subset of "
                f"{sorted(_V03_DEFAULT_METRICS)}."
            )
        if name not in _V03_DEFAULT_METRICS:
            raise ValueError(
                f"unknown metric {name!r}. v0.3.0 supports: "
                f"{sorted(_V03_DEFAULT_METRICS)}. Phase 4 introduces a real "
                f"metric registry."
            )
        resolved.append(name)
    return resolved


def _check_capabilities(
    metric_names: list[str],
    *engines: Engine,
) -> None:
    """Raise :class:`CapabilityError` if any engine is missing a required cap.

    Validation runs before any inference so users see the failure within
    milliseconds of calling, not after waiting for an HFEngine load.
    """
    for name in metric_names:
        required = _METRIC_CAPABILITY_REQUIREMENTS.get(name, frozenset())
        for eng in engines:
            missing = required - eng.capabilities
            if missing:
                raise CapabilityError(
                    missing=missing,
                    engine_name=eng.name,
                    metric_name=name,
                )


# ── Engine construction ───────────────────────────────────────────────


def _build_engine_for_config(
    config: Config,
    template: Optional[Engine],
) -> tuple[Engine, bool]:
    """Build an Engine for ``config``.

    Returns ``(engine, owned)``. ``owned=True`` means the caller (this
    module) is responsible for ``.close()``-ing the engine — the user
    didn't pass an Engine template, so we built it ourselves.

    When ``template`` is given, this module never closes it; the caller
    owns its lifecycle.
    """
    if template is None:
        from lmdiff._engine import HFEngine
        return HFEngine(config), True
    if not hasattr(template, "with_config"):
        raise AttributeError(
            f"engine template {type(template).__name__} does not implement "
            f"`with_config(config) -> Engine`. HFEngine, MinimalEngine, and "
            f"MockEngine all do; custom backends must add it to be usable as "
            f"an `engine=` template."
        )
    return template.with_config(config), False


def _close_owned(engines: list[Engine], owned_flags: list[bool]) -> None:
    """Best-effort close every owned engine, swallowing exceptions so a
    cleanup failure doesn't mask the original error."""
    for eng, owned in zip(engines, owned_flags):
        if not owned:
            continue
        try:
            eng.close()
        except Exception:  # noqa: BLE001
            pass


# ── Public entry points ───────────────────────────────────────────────


def compare(
    base: Union[str, Config],
    variant: Union[str, Config],
    *,
    probes: Union[str, "Any", None] = None,
    n_probes: int = 100,
    metrics: Union[str, list[str]] = "default",
    max_new_tokens: int = 16,
    task_overrides: Optional[dict[str, dict]] = None,
    engine: Optional[Engine] = None,
    seed: Optional[int] = None,
) -> "Any":
    """Pairwise behavioral comparison between ``base`` and ``variant``.

    Both arguments accept either a model-id string (coerced to
    ``Config(model=...)``) or a fully-specified
    :class:`~lmdiff._config.Config`.

    Returns a :class:`~lmdiff.geometry.GeoResult` (one variant). Commit 1.4
    expands the result to schema v5; for v0.3.0 commit 1.1 the existing
    GeoResult shape is what callers receive.

    Parameters
    ----------
    base, variant : str | Config
        The two configurations to compare.
    probes : str | ProbeSet | None
        ``None`` loads the bundled ``v01`` set. A string like
        ``"lm_eval:hellaswag+arc_challenge"`` loads via the lm-eval adapter.
        A ``ProbeSet`` is used as-is.
    n_probes : int
        Truncate the probe set to the first ``n_probes`` probes.
    metrics : "default" | list[str]
        Either ``"default"`` or a subset of ``_V03_DEFAULT_METRICS``. The
        full registry arrives in Phase 4.
    max_new_tokens : int
        Generation length passed through to the underlying ChangeGeometry.
    task_overrides : dict[str, dict] | None
        Per-task hyperparameter overrides; v0.3.0 validates the type only,
        actual handling lands in Phase 4.
    engine : Engine | None
        Optional engine *template*. When supplied, per-variant engines are
        constructed via ``template.with_config(cfg)`` and the template is
        not closed by ``compare``.
    seed : int | None
        Reserved for future randomized metrics; v0.3.0 ignores it.

    Notes
    -----
    When ``engine`` is omitted, ``compare`` constructs HFEngine instances
    internally and closes them in a ``finally`` block, even on error.
    """
    base_cfg = _coerce_to_config(base)
    variant_cfg = _coerce_to_config(variant)

    metric_names = _resolve_metrics(metrics)

    if task_overrides is not None and not isinstance(task_overrides, dict):
        raise TypeError(
            f"task_overrides must be dict | None; got "
            f"{type(task_overrides).__name__}"
        )

    probe_set = _coerce_to_probe_set(probes)
    if n_probes is not None and len(probe_set) > n_probes:
        probe_set = probe_set[:n_probes]

    base_engine, base_owned = _build_engine_for_config(base_cfg, engine)
    engines: list[Engine] = [base_engine]
    owned_flags: list[bool] = [base_owned]
    try:
        variant_engine, var_owned = _build_engine_for_config(variant_cfg, engine)
        engines.append(variant_engine)
        owned_flags.append(var_owned)

        _check_capabilities(metric_names, base_engine, variant_engine)

        # v0.3.0 routes through the existing ChangeGeometry pipeline. The
        # variant Engine is not yet plugged in — ChangeGeometry currently
        # builds its own InferenceEngine from the v0.2.x Config. Phase 4
        # rewires this to consume the new Engine instances directly.
        from lmdiff.geometry import ChangeGeometry
        v02_base = _to_v02_config(base_cfg)
        v02_variant = _to_v02_config(variant_cfg, fallback_name="variant")

        cg = ChangeGeometry(
            base=v02_base,
            variants={v02_variant.display_name: v02_variant},
            prompts=probe_set,
        )
        result = cg.analyze(max_new_tokens=max_new_tokens)
    finally:
        _close_owned(engines, owned_flags)

    return result


def family(
    base: Union[str, Config],
    variants: dict[str, Union[str, Config]],
    *,
    probes: Union[str, "Any", None] = None,
    n_probes: int = 100,
    metrics: Union[str, list[str]] = "default",
    max_new_tokens: int = 16,
    task_overrides: Optional[dict[str, dict]] = None,
    engine: Optional[Engine] = None,
    seed: Optional[int] = None,
) -> "Any":
    """Multi-variant ChangeGeometry against a single ``base``.

    Same call shape as :func:`compare` but ``variants`` is a
    ``{name: str | Config}`` mapping. Returns a
    :class:`~lmdiff.geometry.GeoResult` whose ``variant_names`` list the
    keys of ``variants`` in insertion order.

    Variants run **sequentially** in v0.3.0 — multi-GPU + cross-variant
    parallelism is Phase 5.
    """
    if not isinstance(variants, dict) or not variants:
        raise ValueError("variants must be a non-empty dict[str, str | Config]")

    base_cfg = _coerce_to_config(base)
    variant_cfgs = {name: _coerce_to_config(v) for name, v in variants.items()}

    metric_names = _resolve_metrics(metrics)

    if task_overrides is not None and not isinstance(task_overrides, dict):
        raise TypeError(
            f"task_overrides must be dict | None; got "
            f"{type(task_overrides).__name__}"
        )

    probe_set = _coerce_to_probe_set(probes)
    if n_probes is not None and len(probe_set) > n_probes:
        probe_set = probe_set[:n_probes]

    base_engine, base_owned = _build_engine_for_config(base_cfg, engine)
    engines: list[Engine] = [base_engine]
    owned_flags: list[bool] = [base_owned]
    try:
        for name, vcfg in variant_cfgs.items():
            ve, owned = _build_engine_for_config(vcfg, engine)
            engines.append(ve)
            owned_flags.append(owned)

        _check_capabilities(metric_names, *engines)

        from lmdiff.geometry import ChangeGeometry
        v02_base = _to_v02_config(base_cfg)
        v02_variants = {
            name: _to_v02_config(vcfg, fallback_name=name)
            for name, vcfg in variant_cfgs.items()
        }

        cg = ChangeGeometry(
            base=v02_base,
            variants=v02_variants,
            prompts=probe_set,
        )
        result = cg.analyze(max_new_tokens=max_new_tokens)
    finally:
        _close_owned(engines, owned_flags)

    return result


# ── v0.2.x bridge helpers ─────────────────────────────────────────────


def _to_v02_config(cfg: Config, fallback_name: Optional[str] = None) -> "Any":
    """Translate a v0.3.0 ``Config`` into the v0.2.x dataclass that
    ``ChangeGeometry`` currently consumes.

    This is a temporary bridge: Phase 4 rewires ChangeGeometry to read
    v0.3.0 ``Config`` and an :class:`Engine` directly, removing the need
    for this translation. Until then, only the fields the v0.2.x pipeline
    actually reads — ``model``, ``system_prompt``, ``decode``, ``name`` —
    are propagated.
    """
    import warnings as _w
    # Suppress the v0.2.x DeprecationWarning we'd otherwise emit on every
    # internal bridge construction; the user already opted in by calling
    # the v0.3.0 entry point.
    with _w.catch_warnings():
        _w.simplefilter("ignore", DeprecationWarning)
        from lmdiff.config import Config as V02Config

        decode_spec: DecodeSpec = cfg.decode
        decode_dict: dict[str, Any] = {"strategy": decode_spec.strategy}
        if decode_spec.strategy != "greedy":
            decode_dict["temperature"] = decode_spec.temperature
            decode_dict["top_p"] = decode_spec.top_p
            decode_dict["top_k"] = decode_spec.top_k
            if decode_spec.seed is not None:
                decode_dict["seed"] = decode_spec.seed

        return V02Config(
            model=cfg.model,
            system_prompt=cfg.system_prompt,
            decode=decode_dict,
            name=cfg.name or fallback_name,
        )
