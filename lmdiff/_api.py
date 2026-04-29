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


def _coerce_to_probe_set(
    probes: Union[str, "Any", None],
    *,
    n_probes: Optional[int] = None,
) -> tuple["Any", bool, dict]:
    """Resolve the ``probes`` argument to a ``ProbeSet`` instance.

    Returns ``(probe_set, is_lm_eval_multitask, info)`` where:
      - ``probe_set`` is the resolved set
      - ``is_lm_eval_multitask`` flags whether ``n_probes`` was already
        applied per-task (caller should NOT slice further)
      - ``info`` is a metadata dict — empty for non-lm_eval inputs;
        for lm_eval includes ``{"n_probes_per_task": N, "task_breakdown":
        {task_name: count}, "tasks": [...]}``.

    Supported forms:
      - ``None``                              → bundled ``v01``
      - ``"v01"``                             → bundled ``v01.json``
      - ``"lm_eval:hellaswag+arc_challenge"`` → load+concat via the
        ``from_lm_eval`` adapter; ``n_probes`` applies **per task**
        (so ``n_probes=100`` on a 5-task spec yields 500 probes)
      - ``ProbeSet`` instance                 → returned unchanged

    Per-task application matches the v0.2.x calibration convention
    (100 probes/task × 5 tasks = 500) and matches user expectation. For
    flat probe sets (``"v01"``, ``ProbeSet`` instance) the caller is
    expected to slice to ``n_probes`` for the "total" semantics.
    """
    from lmdiff.probes.loader import ProbeSet

    if probes is None or probes == "v01":
        from pathlib import Path
        v01_path = Path(__file__).parent / "probes" / "v01.json"
        return ProbeSet.from_json(v01_path), False, {}
    if isinstance(probes, ProbeSet):
        return probes, False, {}
    if isinstance(probes, str):
        if probes.startswith("lm_eval:"):
            from lmdiff.probes.adapters import from_lm_eval
            tail = probes.split(":", 1)[1]
            task_names = [t.strip() for t in tail.split("+") if t.strip()]
            if not task_names:
                raise ValueError(
                    f"lm_eval probe spec {probes!r} contains no task names"
                )
            # Per-task limit — the key v0.3.2 fix. Without this the merged
            # set was concatenated whole and then sliced from the front,
            # so on multi-task specs only the first task's probes survived.
            sets = [from_lm_eval(t, limit=n_probes) for t in task_names]
            from lmdiff.probes.loader import Probe
            merged_probes: list[Probe] = []
            task_breakdown: dict[str, int] = {}
            for task_name, ps in zip(task_names, sets):
                task_breakdown[task_name] = len(ps)
                merged_probes.extend(ps)
            ps = ProbeSet(
                merged_probes,
                name=f"lm_eval:{'+'.join(task_names)}",
                version="lm-eval-harness",
            )
            info = {
                "n_probes_per_task": n_probes,
                "task_breakdown": task_breakdown,
                "tasks": list(task_names),
            }
            return ps, True, info
        # Bundled name fallback (e.g., a future "v02"):
        from pathlib import Path
        bundled = Path(__file__).parent / "probes" / f"{probes}.json"
        if bundled.exists():
            return ProbeSet.from_json(bundled), False, {}
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


_BASE_ANCHOR = "__base__"


def _compute_anchor_map(
    items: list[tuple[str, Config]],
) -> dict[str, str]:
    """For each (name, config) in ``items`` (iteration order), pick the
    earliest preceding name whose config is runtime-compatible — that's
    this name's "anchor". A name that's its own anchor starts a new
    group (forces a fresh engine load); two names with the same anchor
    share a single loaded engine.

    The first item's name is conventionally ``"__base__"``. If a variant
    is runtime-compatible with base, its anchor is ``"__base__"`` and
    the base engine serves it for free.
    """
    anchor: dict[str, str] = {}
    representatives: list[tuple[str, Config]] = []
    for name, cfg in items:
        chosen = name
        for rep_name, rep_cfg in representatives:
            if cfg.is_runtime_only_modification_of(rep_cfg):
                chosen = rep_name
                break
        anchor[name] = chosen
        if chosen == name:
            representatives.append((name, cfg))
    return anchor


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
    progress: Optional[bool] = None,
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
        Probe-count cap. Semantics depend on the ``probes`` argument:

          - For a flat probe set (``"v01"``, a ``ProbeSet`` instance):
            **total** number of probes (head-of-list slice).
          - For a multi-task ``"lm_eval:task1+task2+..."`` string:
            **per-task** limit (so ``n_probes=100`` on a 5-task spec
            loads 500 probes — 100 from each task). Matches the v0.2.x
            calibration convention.

        The asymmetry is documented; the per-task expansion is the
        v0.3.2 fix for the v0.3.1 surprise where multi-task strings
        loaded only the first task's first ``n_probes`` rows.
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
    progress : bool | None
        Render per-probe progress bars and per-variant phase markers.
        ``None`` (default) auto-enables on a tty and stays silent in
        pipelines / log redirection. ``True`` forces progress on
        regardless of tty; ``False`` disables it. Override via
        ``LMDIFF_PROGRESS=0`` / ``LMDIFF_PROGRESS=1`` env var.

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

    probe_set, lm_eval_multitask, probe_info = _coerce_to_probe_set(
        probes, n_probes=n_probes,
    )
    # Flat probe sets (bundled / ProbeSet instance) keep the v0.3.0 "total"
    # semantics; multi-task lm_eval strings already had the limit applied
    # per-task inside the loader, so we skip the secondary slice to avoid
    # double-truncation.
    if (
        not lm_eval_multitask
        and n_probes is not None
        and len(probe_set) > n_probes
    ):
        probe_set = probe_set[:n_probes]

    # Engine preflight: when an explicit engine template was provided we
    # build HFEngines here so that capability checks run before any heavy
    # lifting. With ``engine=None`` (default) the geometry path uses the
    # v0.2.x ``InferenceEngine`` directly and the HFEngine instances we'd
    # build here are never used for inference — they were only kept around
    # to feed ``_check_capabilities``. Skip the eager build in that case
    # to avoid loading every model TWICE (once as zombie HFEngine, once as
    # the real InferenceEngine inside ChangeGeometry). Capability checks
    # are deferred to the engine that actually runs inference.
    engines: list[Engine] = []
    owned_flags: list[bool] = []
    if engine is not None:
        base_engine, base_owned = _build_engine_for_config(base_cfg, engine)
        engines.append(base_engine)
        owned_flags.append(base_owned)
    try:
        if engine is not None:
            variant_engine, var_owned = _build_engine_for_config(variant_cfg, engine)
            engines.append(variant_engine)
            owned_flags.append(var_owned)
            _check_capabilities(metric_names, *engines)

        # v0.3.0 routes through the existing ChangeGeometry pipeline. The
        # variant Engine is not yet plugged in — ChangeGeometry currently
        # builds its own InferenceEngine from the v0.2.x Config. Phase 4
        # rewires this to consume the new Engine instances directly.
        from lmdiff.geometry import ChangeGeometry
        v02_base = _to_v02_config(base_cfg)
        v02_variant = _to_v02_config(variant_cfg, fallback_name="variant")
        v02_variant_name = v02_variant.display_name
        if v02_variant_name == _BASE_ANCHOR:
            raise ValueError(
                f"variant name {_BASE_ANCHOR!r} is reserved by lmdiff "
                f"for the base-engine sentinel; rename your variant"
            )

        # When the single variant is runtime-compatible with base it
        # reuses base's engine — otherwise it loads its own. Either way
        # the anchor map records the decision for analyze().
        anchor_map = _compute_anchor_map(
            [(_BASE_ANCHOR, base_cfg), (v02_variant_name, variant_cfg)],
        )

        cg = ChangeGeometry(
            base=v02_base,
            variants={v02_variant_name: v02_variant},
            prompts=probe_set,
        )
        result = cg.analyze(
            max_new_tokens=max_new_tokens,
            progress=progress,
            engine_groups=anchor_map,
        )
        if probe_info:
            result.metadata.update(probe_info)
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
    progress: Optional[bool] = None,
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

    probe_set, lm_eval_multitask, probe_info = _coerce_to_probe_set(
        probes, n_probes=n_probes,
    )
    if (
        not lm_eval_multitask
        and n_probes is not None
        and len(probe_set) > n_probes
    ):
        probe_set = probe_set[:n_probes]

    # See ``compare()`` — only build HFEngine instances when an explicit
    # engine template was passed. With the default ``engine=None`` path,
    # ChangeGeometry loads its own InferenceEngine and the HFEngines we
    # built here would be dead weight (one full model load per variant).
    # On a 7-variant Llama-2 demo this halves peak VRAM at minimum and
    # is the difference between completing and OOMing.
    engines: list[Engine] = []
    owned_flags: list[bool] = []
    if engine is not None:
        base_engine, base_owned = _build_engine_for_config(base_cfg, engine)
        engines.append(base_engine)
        owned_flags.append(base_owned)
    try:
        if engine is not None:
            for name, vcfg in variant_cfgs.items():
                ve, owned = _build_engine_for_config(vcfg, engine)
                engines.append(ve)
                owned_flags.append(owned)
            _check_capabilities(metric_names, *engines)

        # Engine reuse: walk the (base + variants) sequence and find
        # which variants are runtime-compatible with an earlier config.
        # Same-anchor variants will share one loaded engine in
        # ``ChangeGeometry.analyze`` (saves a full model load each).
        if any(name == _BASE_ANCHOR for name in variant_cfgs):
            raise ValueError(
                f"variant name {_BASE_ANCHOR!r} is reserved by lmdiff "
                f"for the base-engine sentinel; rename your variant"
            )
        anchor_map = _compute_anchor_map(
            [(_BASE_ANCHOR, base_cfg), *variant_cfgs.items()],
        )

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
        result = cg.analyze(
            max_new_tokens=max_new_tokens,
            progress=progress,
            engine_groups=anchor_map,
        )
        if probe_info:
            result.metadata.update(probe_info)
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
