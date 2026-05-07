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
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from lmdiff._config import Config
from lmdiff._engine import Engine
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
        if decode.seed is not None:
            out["seed"] = decode.seed
        return out
    # beam / best_of_n / self_consistency aren't yet wired through
    # HFEngine.generate; the v0.2.x path didn't support them either.
    # Fall through to greedy defaults for byte-equivalence with v0.3.2.
    return out


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
) -> tuple[list[float], bool]:
    """Compute the raw δ vector (possibly containing NaN) for one variant.

    Replaces ``ChangeGeometry._delta_for_variant`` byte-for-byte. The
    only structural difference is the per-probe Python loop here vs
    per-batch in the v0.2.x version — same total work, same numeric
    output.
    """
    from lmdiff._progress import iterate as _progress_iter

    n = len(prompts)
    prefix = f"{progress_label} " if progress_label else ""

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
    base_prefix = _prefix_text(base_config)
    v_prefix = _prefix_text(v_config)

    gen_kwargs = _generate_kwargs(v_config, max_new_tokens)

    # ── 1. Generate variant outputs ──
    v_outputs: list[str] = []
    v_ids_per_probe: list[list[int]] = []
    for i in _progress_iter(
        range(n), desc=f"{prefix}generate", total=n, enable=progress,
    ):
        try:
            gen = v_engine.generate(
                prompts[i], prefix_text=v_prefix, **gen_kwargs,
            )
        except TypeError:
            # Engine without prefix_text kwarg — fall back to single-
            # tokenize via concatenation. (e.g. MockEngine in unit
            # tests.) For real backends the calibration test would catch
            # any byte-level divergence.
            gen = v_engine.generate(v_prefix + prompts[i], **gen_kwargs)
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
        if not v_outputs[i]:
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
        if not v_outputs[i]:
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

    return deltas, use_bpb


# ── Top-level pipeline ────────────────────────────────────────────────


def run_family_pipeline(
    base_engine: Engine,
    base_config: Config,
    variant_engines: dict[str, Engine],
    variant_configs: dict[str, Config],
    probe_set: ProbeSet,
    *,
    max_new_tokens: int = 16,
    progress: Optional[bool] = None,
    engine_groups: Optional[dict[str, str]] = None,
) -> GeoResult:
    """Run the family pipeline using only the Engine Protocol.

    Parameters
    ----------
    base_engine : Engine
        Already-loaded base engine (lifecycle owned by caller).
    base_config : Config
        v0.3 Config carrying base's runtime-only fields (system_prompt,
        context, decode) for prompt assembly.
    variant_engines : dict[str, Engine]
        Per-variant engines. May share instances with each other or
        with ``base_engine`` when the configs are runtime-compatible
        (engine reuse, see v0.3.2 PR #10).
    variant_configs : dict[str, Config]
        Per-variant v0.3 Configs. Carries each variant's runtime-only
        fields, applied at prompt-assembly time. Matches the
        ``variant_engines`` keys.
    probe_set : ProbeSet
        The probe set. Domains and per-probe text are read from here.
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

    variant_names = list(variant_engines.keys())
    n_v = len(variant_names)
    if n_v == 0:
        raise ValueError("variant_engines must contain at least one entry")
    if set(variant_configs.keys()) != set(variant_names):
        raise ValueError(
            "variant_engines and variant_configs must share the same keys",
        )

    # ── 1. Per-probe token counts (Protocol-clean — no _tokenizer) ──
    all_probe_tokens: list[int] = [base_engine.token_count(p) for p in prompts]

    # ── 2. Engine cache + look-ahead release loop ──
    # Mirrors ChangeGeometry.analyze's loop with the same anchor map
    # semantics.
    engine_cache: dict[str, Engine] = {_BASE_ANCHOR: base_engine}
    if engine_groups is None:
        engine_groups = {n: n for n in variant_names}

    def _anchor_of(name: str) -> str:
        return engine_groups.get(name, name)

    raw_deltas: dict[str, list[float]] = {}
    bpb_flags: dict[str, bool] = {}

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
                    # Variant has its own engine; pull from the supplied
                    # variant_engines dict (caller built it). The cache
                    # then keeps it for any later variants sharing this
                    # anchor.
                    v_engine = variant_engines[anchor]
                    engine_cache[anchor] = v_engine

                raw_deltas[name], bpb_flags[name] = _delta_for_variant(
                    base_engine=base_engine,
                    base_config=base_config,
                    v_engine=v_engine,
                    v_config=v_config,
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    progress=progress,
                    progress_label=f"v{v_idx}/{n_v} {name}",
                )

                # Look-ahead-by-one release. Same rule as
                # ChangeGeometry.analyze (v0.3.2 PR #10).
                if anchor != _BASE_ANCHOR:
                    next_idx = v_idx  # 0-based next position
                    keep = (
                        next_idx < n_v
                        and _anchor_of(variant_names[next_idx]) == anchor
                    )
                    if not keep:
                        engine_cache.pop(anchor, None)
                        # Caller owns the engine lifecycle (it built
                        # them and will close them in finally) — we
                        # don't .close() here. The cache pop is enough
                        # to release the local reference for GC.
                        gc.collect()
                        lifecycle_log(
                            "engine_release",
                            anchor=anchor,
                            after_variant=name,
                        )
    finally:
        # Drop every cached entry except base (which the caller still
        # holds via base_engine and may want to close on its own).
        for cached_anchor in list(engine_cache):
            if cached_anchor == _BASE_ANCHOR:
                continue
            engine_cache.pop(cached_anchor, None)
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

    # Schema v4 corrected at v0.3.2 PR #11: per-domain per-token
    # normalization, with the overall as the per-domain RMS.
    mag_per_domain_norm = _compute_per_domain_normalized(
        variant_names, change_vectors, probe_domains, avg_tokens_per_probe,
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
    )
    result.share_per_domain = _compute_share_per_domain(result)
    return result


__all__ = ["run_family_pipeline"]
