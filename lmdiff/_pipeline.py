"""Family-pipeline for the v0.4.0 backend cutover.

Port of ``lmdiff.geometry.ChangeGeometry.analyze`` that consumes the
public Engine Protocol (HFEngine, MockEngine, MinimalEngine, custom)
instead of the v0.2.x ``InferenceEngine``.

The math is unchanged from ``ChangeGeometry.analyze`` — same change-vector
computation, same per-token / per-domain normalization, same share /
overall formulas (the v0.3.2 corrected ones from PR #11). The only
difference is the call shape into the engine layer:

  - v0.2.x ``InferenceEngine.score(prompts: list[str], …)`` was batched
    internally with a per-probe Python loop. The new pipeline runs the
    per-probe loop *here*, calling ``Engine.score(prompt, continuation)``
    once per probe. Same total Python overhead, no per-call regression.
  - Runtime-only Config fields (``system_prompt``, ``context``,
    ``decode``) are applied at the **prompt-assembly** layer here, not
    via Engine kwargs. The Engine sees an already-assembled prompt
    string — it doesn't need to know what a "system prompt" is.
  - Per-probe token counts come from the Protocol's ``token_count``
    method (added v0.4.0); cross-tokenizer fallback uses
    ``tokenizers_equivalent_to`` (also v0.4.0). No more direct attribute
    access on private engine state.

Engine ownership: this module **does not** close engines it doesn't
create. Engine lifecycle is the caller's responsibility (the
``_api.compare`` / ``_api.family`` helpers already handle this in their
``try/finally`` blocks).
"""
from __future__ import annotations

import gc
import math
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from lmdiff._config import Config
from lmdiff._engine import Engine
from lmdiff._validity import EngineValidity, ProbeValidity, compute_domain_status

# Engine factory contract (Fix 4, v0.4.0 PR #15):
#   factory(config) -> (engine, pipeline_owns_lifecycle)
# When ``pipeline_owns_lifecycle`` is True, ``run_family_pipeline``
# calls ``engine.close()`` after look-ahead-by-one release. When False,
# the caller (typically ``_api.family``) handed us a long-lived engine
# (e.g. the user's ``engine=`` template's ``with_config`` result) and
# we leave its lifetime alone.
EngineFactory = Callable[[Config], "tuple[Engine, bool]"]
from lmdiff.geometry import (
    GeoResult,
    _compute_overall_normalized_from_pdn,
    _compute_per_domain_normalized,
    _compute_share_per_domain,
    _cosine_matrix,
    _selective_decomposition,
    _universally_valid_indices,
)
from lmdiff.tokenizer_utils import bpb_from_ce
from lmdiff.probes.loader import ProbeSet


_BASE_ANCHOR = "__base__"


# ── Prompt assembly (runtime-only Config fields) ─────────────────────


def _prefix_text(config: Config) -> str:
    """Build the prefix that precedes a probe, from a v0.3 ``Config``.

    Mirrors v0.2.x ``InferenceEngine._prefix_text`` byte-for-byte:
    ``"\\n".join([system_prompt, *context_contents]) + "\\n"`` when
    any prefix material is present; ``""`` otherwise.

    Note: keeping the trailing ``"\\n"`` matters for byte-equivalence
    with the v0.2.x calibration baseline. Don't strip.
    """
    parts: list[str] = []
    if config.system_prompt:
        parts.append(config.system_prompt)
    if config.context:
        for msg in config.context:
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if content:
                parts.append(content)
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def _assemble_prompt(config: Config, probe_text: str) -> str:
    """Concatenate prefix + probe. The Engine sees a single string."""
    return _prefix_text(config) + probe_text


# ── Decode → generate kwargs ─────────────────────────────────────────


def _generate_kwargs(config: Config, max_new_tokens: int) -> dict[str, Any]:
    """Translate the v0.3 ``DecodeSpec`` into Engine.generate kwargs.

    The Engine Protocol's ``generate`` signature (v0.4.0) is:
        generate(prompt, *, max_new_tokens, temperature, top_p,
                 top_k, seed, prefix_text)

    Mirrors v0.2.x ``InferenceEngine._decode_params`` — temperature,
    top_p, top_k all flow through. ``top_k`` defaults to 0 (no
    filtering); HF's ``model.generate`` defaults to top_k=50 when the
    kwarg is omitted, which silently truncates sample-decode
    distributions. Passing top_k=0 explicitly is what makes ``temp_1.5``
    variants byte-equivalent to v0.3.2.

    NB: ``seed`` is **not** included here. Seed is applied once per
    variant at probe 0 by ``_delta_for_variant`` (see ``_resolve_seed``);
    repeating it on every probe call would reset RNG between probes
    and force every probe in a sampling variant to see the same RNG
    state, which is the wrong granularity (lab convention is "pin
    once per experiment, let RNG advance naturally").
    """
    decode = config.decode
    out: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if decode.strategy == "greedy":
        # Defaults are temperature=1.0, top_p=1.0 → HFEngine sets do_sample=False.
        return out
    if decode.strategy == "sample":
        out["temperature"] = decode.temperature
        out["top_p"] = decode.top_p
        out["top_k"] = decode.top_k
        return out
    # beam / best_of_n / self_consistency aren't yet wired through
    # HFEngine.generate; the v0.2.x path didn't support them either.
    # Fall through to greedy defaults for byte-equivalence with v0.3.2.
    return out


def _resolve_seed(
    v_config: Config, family_seed: Optional[int],
) -> Optional[int]:
    """Effective seed for one variant's generate phase.

    Precedence (Fix 3, v0.4.0 PR #15):
      1. ``v_config.decode.seed`` if set explicitly — per-variant
         override (a user can pin one variant while leaving others
         unpinned).
      2. ``family_seed`` if set — applies to every variant whose
         DecodeSpec didn't override.
      3. ``None`` — no ``manual_seed`` call; RNG state is whatever
         prior work left it as. Matches PyTorch convention ("no seed
         arg = no seeding"); the cost is non-reproducibility for
         sampling variants, which the user is responsible for if they
         pass neither family seed nor DecodeSpec seed.
    """
    decode_seed = getattr(v_config.decode, "seed", None)
    if decode_seed is not None:
        return decode_seed
    return family_seed


# ── Per-variant change-vector computation ────────────────────────────


def _delta_for_variant(
    base_engine: Engine,
    base_config: Config,
    v_engine: Engine,
    v_config: Config,
    prompts: list[str],
    *,
    max_new_tokens: int,
    progress: Optional[bool],
    progress_label: str,
    seed: Optional[int] = None,
    base_validity_per_probe: Optional[list[EngineValidity]] = None,
    all_probe_tokens: Optional[list[int]] = None,
) -> tuple[list[float], bool, list[EngineValidity]]:
    """Compute the raw δ vector (possibly containing NaN) for one variant.

    v0.4.1 measurement validity (Q9.1, audit §1): when
    ``base_validity_per_probe`` and ``all_probe_tokens`` are provided,
    we compute per-probe ``EngineValidity`` for the variant and skip
    the per-probe sub-loop work for probes flagged invalid for the
    relevant engine. Skipped probes contribute ``δ = NaN`` and are
    dropped by the existing ``_universally_valid_indices`` filter.

    Returns
    -------
    (deltas, use_bpb, variant_validity_per_probe) where
    ``variant_validity_per_probe[i]`` is the variant engine's record
    for ``prompts[i]``. When validity inputs are None (legacy callers,
    e.g. existing unit tests), every probe is treated as valid and the
    returned list is built with ``reason="unknown_limit"`` records that
    don't influence aggregation.
    """
    from lmdiff._progress import iterate as _progress_iter

    n = len(prompts)
    prefix = f"{progress_label} " if progress_label else ""

    # ── Validity precompute (Q9.1, audit §1) ───────────────────────
    # Build the variant's per-probe EngineValidity. Base records are
    # passed in by run_family_pipeline (computed once across variants).
    # When no validity inputs are supplied, we synthesize all-valid
    # records with reason="unknown_limit" so callers that don't care
    # about validity (existing unit tests) continue to work.
    variant_max = v_engine.max_context_length()
    base_max = base_engine.max_context_length()  # for the base reuse fallback below

    base_prefix_text = _prefix_text(base_config)
    v_prefix_text = _prefix_text(v_config)
    base_prefix_T = (
        base_engine.token_count(base_prefix_text) if base_prefix_text else 0
    )
    v_prefix_T = (
        v_engine.token_count(v_prefix_text) if v_prefix_text else 0
    )

    same_tokenizer = base_engine.tokenizers_equivalent_to(v_engine)

    variant_validity: list[EngineValidity] = []
    base_skip: list[bool] = [False] * n  # skip score-base loop for this probe
    var_skip: list[bool] = [False] * n   # skip generate / score-variant loops
    for i in range(n):
        # Variant T_i = prefix_var + token_count_for_variant + max_new_tokens
        if all_probe_tokens is not None and same_tokenizer:
            var_T = v_prefix_T + all_probe_tokens[i] + max_new_tokens
        elif all_probe_tokens is not None:
            var_T = v_prefix_T + v_engine.token_count(prompts[i]) + max_new_tokens
        else:
            var_T = -1  # unknown — default to valid

        if variant_max is None or var_T < 0:
            v_is_valid = True
            v_reason = "unknown_limit"
        else:
            v_is_valid = var_T <= variant_max
            v_reason = "valid" if v_is_valid else "exceeds_context"

        variant_validity.append(EngineValidity(
            engine_name=v_engine.name,
            max_context=variant_max,
            T_i=var_T if var_T >= 0 else 0,
            is_valid=v_is_valid,
            reason=v_reason,
        ))

        if not v_is_valid:
            var_skip[i] = True

        if base_validity_per_probe is not None:
            base_ev = base_validity_per_probe[i]
            if not base_ev.is_valid:
                base_skip[i] = True

    # Per-Config prefix (system_prompt + context). Passed to the engine
    # via ``prefix_text=`` kwarg so the engine can split-tokenize
    # prefix vs probe — matches v0.2.x ``InferenceEngine._encode_for_model``
    # byte-for-byte. Pre-concatenating the prefix into the prompt and
    # using single-tokenize would cause SentencePiece boundary effects
    # (Llama in particular) that drift the variant's δ values away
    # from v0.3.2 — see L-030. When prefix_text is empty (most common
    # case: variants without ``system_prompt`` / ``context``), the
    # split-tokenize path collapses to ``[BOS] + tokens(probe)``,
    # byte-identical to the empty-prefix single-tokenize path.
    base_prefix = base_prefix_text  # alias; same value, kept for readability
    v_prefix = v_prefix_text

    gen_kwargs = _generate_kwargs(v_config, max_new_tokens)

    # Once-per-variant seed pinning (Fix 3, v0.4.0 PR #15). Resolved
    # seed (DecodeSpec.seed > family seed > None) is passed to
    # ``engine.generate(seed=…)`` ONLY on probe 0; subsequent probes
    # pass ``seed=None`` so RNG advances naturally through the loop.
    # This matches the user mental model "set seed, run experiment"
    # and avoids the per-probe variant where every probe sees the
    # same RNG state (over-correlated sampling). If both family seed
    # and DecodeSpec.seed are None, ``effective_seed`` is None and no
    # ``manual_seed`` is called anywhere — that's the intentional
    # PyTorch-convention fallback (non-reproducible by design).
    effective_seed = _resolve_seed(v_config, seed)

    # ── 1. Generate variant outputs ──
    # Skip generation for probes flagged invalid for the variant
    # engine (v0.4.1 validity framework). Empty output then
    # short-circuits the score loops below to NaN δ, which the global
    # _universally_valid_indices filter drops.
    v_outputs: list[str] = []
    v_ids_per_probe: list[list[int]] = []
    for i in _progress_iter(
        range(n), desc=f"{prefix}generate", total=n, enable=progress,
    ):
        if var_skip[i]:
            v_outputs.append("")
            v_ids_per_probe.append([])
            continue
        gen_seed = effective_seed if i == 0 else None
        try:
            gen = v_engine.generate(
                prompts[i], prefix_text=v_prefix, seed=gen_seed,
                **gen_kwargs,
            )
        except TypeError:
            # Engine without prefix_text kwarg — fall back to single-
            # tokenize via concatenation. (e.g. MockEngine in unit
            # tests.) For real backends the calibration test would catch
            # any byte-level divergence. Seed kwarg is in the Protocol,
            # so it's safe to pass even on the fallback path.
            gen = v_engine.generate(
                v_prefix + prompts[i], seed=gen_seed, **gen_kwargs,
            )
        v_outputs.append(gen.text)
        v_ids_per_probe.append(list(gen.tokens))

    # ── 2. Score base on variant's outputs (cross) ──
    # base uses *its own* prefix (base_config.system_prompt etc.), not
    # the variant's — that's what makes ce_b_of_v meaningful as
    # "how surprised is base by what variant produced".
    ce_b_of_v: list[float] = [0.0] * n
    ntok_b_of_v: list[int] = [0] * n
    for i in _progress_iter(
        range(n), desc=f"{prefix}score base|v", total=n, enable=progress,
    ):
        # Skip when probe is out-of-context for base (v0.4.1) OR when
        # variant produced no output (existing behavior).
        if base_skip[i] or not v_outputs[i]:
            ce_b_of_v[i] = float("nan")
            ntok_b_of_v[i] = 0
            continue
        try:
            sr = base_engine.score(
                prompts[i], v_outputs[i], prefix_text=base_prefix,
            )
        except TypeError:
            sr = base_engine.score(base_prefix + prompts[i], v_outputs[i])
        if len(sr.tokens) == 0:
            ce_b_of_v[i] = float("nan")
            ntok_b_of_v[i] = 0
            continue
        # ScoreResult.logprobs is the per-token log-probability of the
        # continuation. ce = -mean(logprobs) per the v0.2.x convention.
        lp = np.asarray(sr.logprobs, dtype=np.float64)
        ce_b_of_v[i] = float(-lp.sum() / len(lp))
        ntok_b_of_v[i] = len(sr.tokens)

    # ── 3. Score variant on its own outputs (self-score) ──
    # Use the pre-tokenized continuation_ids when the engine supports
    # it (HFEngine does — added v0.3.1 PR #7), otherwise fall back to
    # text. Self-scoring with continuation_ids avoids decode→retokenize
    # drift (the lm-eval-convention guarantee).
    ce_v_self: list[float] = [0.0] * n
    ntok_v_self: list[int] = [0] * n
    for i in _progress_iter(
        range(n), desc=f"{prefix}score v|v", total=n, enable=progress,
    ):
        # var_skip[i] already implies v_outputs[i] == "" via the
        # generate skip above; check anyway for clarity.
        if var_skip[i] or not v_outputs[i]:
            ce_v_self[i] = float("nan")
            ntok_v_self[i] = 0
            continue
        try:
            # Self-score with pre-tokenized continuation_ids and
            # variant-specific prefix. Don't pass ``continuation`` at
            # all — HFEngine's validator requires *exactly one* of
            # (continuation, continuation_ids).
            sr = v_engine.score(
                prompts[i],
                continuation_ids=v_ids_per_probe[i],
                prefix_text=v_prefix,
            )
        except TypeError:
            # Engine without continuation_ids and/or prefix_text
            # support — fall back to text-based scoring with
            # concatenated prefix. (e.g. MockEngine in unit tests.)
            sr = v_engine.score(v_prefix + prompts[i], v_outputs[i])
        if len(sr.tokens) == 0:
            ce_v_self[i] = float("nan")
            ntok_v_self[i] = 0
            continue
        lp = np.asarray(sr.logprobs, dtype=np.float64)
        ce_v_self[i] = float(-lp.sum() / len(lp))
        ntok_v_self[i] = len(sr.tokens)

    # ── 4. BPB normalization when tokenizers differ ──
    # Fast path: tokenizer_id match. Slow path: canary-string check via
    # the engine's tokenizers_equivalent_to (added v0.4.0).
    use_bpb = not base_engine.tokenizers_equivalent_to(v_engine)

    deltas: list[float] = []
    for i in range(n):
        bv = ce_b_of_v[i]
        vv = ce_v_self[i]
        if math.isnan(bv) or math.isnan(vv):
            deltas.append(float("nan"))
            continue
        if use_bpb:
            bv = bpb_from_ce(bv, n_tokens=ntok_b_of_v[i], text=v_outputs[i])
            vv = bpb_from_ce(vv, n_tokens=ntok_v_self[i], text=v_outputs[i])
        deltas.append(float(bv - vv))

    return deltas, use_bpb, variant_validity


# ── Top-level pipeline ────────────────────────────────────────────────


def run_family_pipeline(
    base_engine: Engine,
    base_config: Config,
    variant_configs: dict[str, Config],
    probe_set: ProbeSet,
    *,
    variant_engines: Optional[dict[str, Engine]] = None,
    engine_factory: Optional[EngineFactory] = None,
    max_new_tokens: int = 16,
    progress: Optional[bool] = None,
    engine_groups: Optional[dict[str, str]] = None,
    seed: Optional[int] = None,
) -> GeoResult:
    """Run the family pipeline using only the Engine Protocol.

    Two engine-supply modes — exactly one of ``variant_engines`` or
    ``engine_factory`` must be supplied:

    - **Eager** (``variant_engines={name: Engine}``): caller pre-built
      every variant engine. Pipeline never closes them — caller owns
      lifecycle. Backward-compatible with v0.4.0-Fix-3 callsites and
      with unit tests that inject Mock engines explicitly.

    - **Lazy** (``engine_factory=callable``): callable takes one
      ``Config`` and returns ``(engine, pipeline_owns_lifecycle)``.
      Engines are constructed on cache-miss inside the variant loop —
      one at a time. After look-ahead-by-one release fires, if the
      pipeline owns the engine it's ``.close()``-ed before moving on.
      Peak resident engines = 2 (base + active variant). This is the
      v0.3.2 ``ChangeGeometry.analyze`` behavior, restored in v0.4.0
      via Fix 4. Without this mode, ``_api.family`` would pre-load
      every unique-model variant before pipeline start (6+ model loads
      sitting in memory simultaneously for the 7-variant Llama family).

    Parameters
    ----------
    base_engine : Engine
        Already-loaded base engine (lifecycle owned by caller).
    base_config : Config
        v0.3 Config carrying base's runtime-only fields (system_prompt,
        context, decode) for prompt assembly.
    variant_configs : dict[str, Config]
        Per-variant v0.3 Configs in iteration order. The variant order
        determines the look-ahead-by-one release schedule. Carries each
        variant's runtime-only fields, applied at prompt-assembly time.
    probe_set : ProbeSet
        The probe set. Domains and per-probe text are read from here.
    variant_engines : dict[str, Engine] | None
        Eager-mode engine map. Keys must match ``variant_configs``.
        Mutually exclusive with ``engine_factory``.
    engine_factory : Callable[[Config], (Engine, bool)] | None
        Lazy-mode engine factory. Called by the pipeline on first
        cache-miss for each unique anchor (not per-variant — variants
        sharing an anchor share the loaded engine). Returns
        ``(engine, pipeline_owns_lifecycle)``. When the second value
        is True, the pipeline ``.close()``-s the engine on release.
        Mutually exclusive with ``variant_engines``.
    max_new_tokens : int
        Generation length cap.
    progress : bool | None
        Forwarded to per-probe progress bars and per-variant phase
        markers (see ``lmdiff._progress``).
    engine_groups : dict[str, str] | None
        ``variant_name → anchor_name`` map for the look-ahead-by-one
        release decision. ``"__base__"`` denotes the base engine
        (never released). When ``None``, every variant is its own
        anchor — no reuse.
    seed : int | None
        Top-level RNG seed (Fix 3, v0.4.0 PR #15). When set, each
        variant whose ``DecodeSpec.seed is None`` will pin RNG via
        a single ``torch.manual_seed`` call at probe 0 of that
        variant's generate phase; subsequent probes advance RNG
        naturally. ``DecodeSpec.seed`` takes precedence per variant.
        ``None`` (default) leaves RNG unpinned — sample-decode
        variants are non-reproducible across runs (PyTorch convention).

    Returns
    -------
    GeoResult
        Schema v5, fully populated. Numerically equivalent to what
        ``ChangeGeometry.analyze`` would produce on the same inputs.
    """
    from lmdiff._progress import phase as _phase, lifecycle_log

    prompts = list(probe_set.texts)
    n_total = len(prompts)
    if n_total == 0:
        raise ValueError("cannot analyze on an empty probe set")

    if (variant_engines is None) == (engine_factory is None):
        raise ValueError(
            "run_family_pipeline requires exactly one of "
            "`variant_engines` (eager) or `engine_factory` (lazy)",
        )

    variant_names = list(variant_configs.keys())
    n_v = len(variant_names)
    if n_v == 0:
        raise ValueError("variant_configs must contain at least one entry")

    if variant_engines is not None and set(variant_engines.keys()) != set(variant_names):
        raise ValueError(
            "variant_engines and variant_configs must share the same keys",
        )

    # ── 1. Per-probe token counts (Protocol-clean — no _tokenizer) ──
    all_probe_tokens: list[int] = [base_engine.token_count(p) for p in prompts]

    # ── 1b. Per-probe base validity (v0.4.1, Q9.6 worst-case bound) ──
    # Built once across all variants since base validity is variant-
    # independent. T_i = T_prefix + T_prompt + max_new_tokens (worst
    # case before generation). Variant-side validity is built inside
    # _delta_for_variant where the variant engine is in scope.
    base_max = base_engine.max_context_length()
    base_prefix_text = _prefix_text(base_config)
    base_prefix_T = (
        base_engine.token_count(base_prefix_text) if base_prefix_text else 0
    )
    base_validity_per_probe: list[EngineValidity] = []
    for i in range(n_total):
        t_i = base_prefix_T + all_probe_tokens[i] + max_new_tokens
        if base_max is None:
            is_valid, reason = True, "unknown_limit"
        else:
            is_valid = t_i <= base_max
            reason = "valid" if is_valid else "exceeds_context"
        base_validity_per_probe.append(EngineValidity(
            engine_name=base_engine.name,
            max_context=base_max,
            T_i=t_i,
            is_valid=is_valid,
            reason=reason,
        ))

    # ── 2. Engine cache + look-ahead release loop ──
    # Mirrors ChangeGeometry.analyze's loop with the same anchor map
    # semantics. The cache key is the *anchor* name, not the variant
    # name — variants sharing an anchor share a single loaded engine.
    engine_cache: dict[str, Engine] = {_BASE_ANCHOR: base_engine}
    # Tracks which cached engines were built BY this pipeline call (lazy
    # mode only). On release, pipeline-owned engines are .close()-d;
    # caller-owned engines (eager mode, or lazy factories that returned
    # owned=False) are left alone — caller manages their lifecycle.
    pipeline_owned: dict[str, bool] = {}

    if engine_groups is None:
        engine_groups = {n: n for n in variant_names}

    def _anchor_of(name: str) -> str:
        return engine_groups.get(name, name)

    raw_deltas: dict[str, list[float]] = {}
    bpb_flags: dict[str, bool] = {}
    # Per-variant variant-side validity records (one EngineValidity per
    # probe). Merged with base records into the master probe_validity
    # dict after the variant loop completes.
    variant_validity_records: dict[str, list[EngineValidity]] = {}

    try:
        for v_idx, name in enumerate(variant_names, 1):
            v_config = variant_configs[name]
            anchor = _anchor_of(name)
            with _phase(
                f"variant {v_idx}/{n_v} ({name}): "
                f"{'reuse '+anchor if anchor in engine_cache else 'load'}"
                f"+generate+score",
                enable=progress,
            ):
                if anchor in engine_cache:
                    v_engine = engine_cache[anchor]
                    lifecycle_log(
                        "engine_reuse",
                        variant=name,
                        anchor=anchor,
                    )
                else:
                    # Cache miss — produce an engine for this anchor.
                    # Eager: pull from the dict the caller pre-built.
                    # Lazy: construct via factory, remember ownership.
                    if variant_engines is not None:
                        v_engine = variant_engines[anchor]
                    else:
                        anchor_config = variant_configs[anchor]
                        v_engine, owns = engine_factory(anchor_config)
                        if owns:
                            pipeline_owned[anchor] = True
                        lifecycle_log(
                            "engine_load",
                            variant=name,
                            anchor=anchor,
                            owned_by_pipeline=owns,
                        )
                    engine_cache[anchor] = v_engine

                deltas, use_bpb, var_validity = _delta_for_variant(
                    base_engine=base_engine,
                    base_config=base_config,
                    v_engine=v_engine,
                    v_config=v_config,
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    progress=progress,
                    progress_label=f"v{v_idx}/{n_v} {name}",
                    seed=seed,
                    base_validity_per_probe=base_validity_per_probe,
                    all_probe_tokens=all_probe_tokens,
                )
                raw_deltas[name] = deltas
                bpb_flags[name] = use_bpb
                variant_validity_records[name] = var_validity

                # Look-ahead-by-one release. Same rule as
                # ChangeGeometry.analyze (v0.3.2 PR #10).
                if anchor != _BASE_ANCHOR:
                    next_idx = v_idx  # 0-based next position
                    keep = (
                        next_idx < n_v
                        and _anchor_of(variant_names[next_idx]) == anchor
                    )
                    if not keep:
                        released = engine_cache.pop(anchor, None)
                        # Lazy + pipeline-owned: actually free the
                        # weights now. Without this .close() the engine
                        # stays in CPU/GPU memory because some local
                        # variable in the caller's frame might still
                        # reference it (or accelerate's device_map kept
                        # tensors alive). v0.3.2's lazy path closed
                        # explicitly here — Fix 4 restores that.
                        if pipeline_owned.pop(anchor, False) and released is not None:
                            try:
                                released.close()
                            except Exception:  # noqa: BLE001
                                pass
                        gc.collect()
                        lifecycle_log(
                            "engine_release",
                            anchor=anchor,
                            after_variant=name,
                        )
    finally:
        # Drop every cached entry except base (which the caller still
        # holds via base_engine and may want to close on its own).
        # Pipeline-owned remnants (e.g. an exception aborted the loop
        # before look-ahead release fired) get .close()-d so they
        # don't leak weights.
        for cached_anchor in list(engine_cache):
            if cached_anchor == _BASE_ANCHOR:
                continue
            stale = engine_cache.pop(cached_anchor, None)
            if pipeline_owned.pop(cached_anchor, False) and stale is not None:
                try:
                    stale.close()
                except Exception:  # noqa: BLE001
                    pass
        gc.collect()

    # ── 3. Build the GeoResult — pure-function math reused as-is ──
    valid_indices = _universally_valid_indices(raw_deltas, n_total)
    n_valid = len(valid_indices)

    change_vectors: dict[str, list[float]] = {
        name: [raw_deltas[name][i] for i in valid_indices]
        for name in variant_names
    }
    per_probe: dict[str, dict[str, float]] = {}
    for name in variant_names:
        per_probe[name] = {
            prompts[i]: raw_deltas[name][i] for i in valid_indices
        }

    magnitudes: dict[str, float] = {
        name: float(np.linalg.norm(change_vectors[name])) if n_valid > 0 else 0.0
        for name in variant_names
    }
    cosine_matrix = _cosine_matrix(variant_names, change_vectors, magnitudes)

    delta_means, selective_magnitudes, selective_cosine_matrix = (
        _selective_decomposition(variant_names, change_vectors)
    )

    # probe_domains aligned with change_vectors after NaN filter (v3+).
    probe_domains: tuple[str | None, ...] = ()
    all_domains = [p.domain for p in probe_set]
    probe_domains = tuple(all_domains[i] for i in valid_indices)

    # Schema v4: per-probe token counts.
    avg_tokens_per_probe: tuple[float, ...] = tuple(
        float(all_probe_tokens[i]) for i in valid_indices
    )

    # ── Validity aggregation (v0.4.1) — must happen BEFORE pdn ──
    # because the pdn formula needs domain_status to know which
    # (variant, domain) cells should be None vs computed.
    base_name = base_engine.name
    probe_validity: dict[str, ProbeValidity] = {}
    for i in range(n_total):
        probe = probe_set[i]
        per_engine: dict[str, EngineValidity] = {
            base_name: base_validity_per_probe[i],
        }
        for vname in variant_names:
            var_rec = variant_validity_records[vname][i]
            per_engine[var_rec.engine_name] = var_rec
        probe_validity[probe.id] = ProbeValidity(
            probe_id=probe.id,
            domain=probe.domain,
            per_engine=per_engine,
        )

    # Per-(variant, domain) status — uses ALL probes (pre-filter), so
    # variant_only / out_of_range domains are correctly classified even
    # when their δ values are all NaN. Keyed by the variant DICT KEY
    # (vname, what the user passed to family()) for caller convenience;
    # the validity lookup itself uses engine.name (display name from
    # EngineValidity.engine_name) since per_engine is keyed that way.
    domain_status: dict[str, dict[str, str]] = {}
    domains_in_set = sorted({
        p.domain for p in probe_set if p.domain is not None
    })
    for vname in variant_names:
        # Look up this variant's engine display name from a sample
        # validity record (they all share engine_name within a variant).
        v_engine_name = variant_validity_records[vname][0].engine_name
        v_status: dict[str, str] = {}
        for d in domains_in_set:
            probes_in_d = [
                pv for pv in probe_validity.values() if pv.domain == d
            ]
            v_status[d] = compute_domain_status(
                probes_in_d, base_name, v_engine_name,
            )
        domain_status[vname] = v_status

    # ── Per-domain pdn + overall (v0.4.1 corrected formula, validity-aware) ──
    # mag_per_domain_norm[v][d] = sqrt(mean(δ²)) over valid probes;
    # None for out_of_range / variant_only domains. Overall =
    # per-domain RMS over valid (non-None) entries.
    mag_per_domain_norm = _compute_per_domain_normalized(
        variant_names, change_vectors, probe_domains, avg_tokens_per_probe,
        domain_status=domain_status,
    )
    magnitudes_normalized: dict[str, float] = {}
    if mag_per_domain_norm:
        magnitudes_normalized = _compute_overall_normalized_from_pdn(
            mag_per_domain_norm,
        )
    elif avg_tokens_per_probe:
        # Single-domain fallback — same as ChangeGeometry.analyze.
        mean_tokens = float(np.mean(avg_tokens_per_probe))
        denom = math.sqrt(n_valid * mean_tokens) if mean_tokens > 0 else 0.0
        if denom > 0:
            magnitudes_normalized = {
                name: magnitudes[name] / denom for name in variant_names
            }

    metadata = {
        "n_total_probes": n_total,
        "n_skipped": n_total - n_valid,
        "bpb_normalized": bpb_flags,
        "max_new_tokens": max_new_tokens,
        "base_max_context": base_max,
    }
    if probe_set.name:
        metadata["probe_set_name"] = probe_set.name
    if probe_set.version:
        metadata["probe_set_version"] = probe_set.version

    result = GeoResult(
        base_name=base_config.display_name,
        variant_names=variant_names,
        n_probes=n_valid,
        magnitudes=magnitudes,
        cosine_matrix=cosine_matrix,
        change_vectors=change_vectors,
        per_probe=per_probe,
        metadata=metadata,
        delta_means=delta_means,
        selective_magnitudes=selective_magnitudes,
        selective_cosine_matrix=selective_cosine_matrix,
        probe_domains=probe_domains,
        avg_tokens_per_probe=avg_tokens_per_probe,
        magnitudes_normalized=magnitudes_normalized,
        magnitudes_per_domain_normalized=mag_per_domain_norm,
        probe_validity=probe_validity,
        domain_status=domain_status,
    )
    result.share_per_domain = _compute_share_per_domain(result)
    return result


__all__ = ["EngineFactory", "run_family_pipeline"]
