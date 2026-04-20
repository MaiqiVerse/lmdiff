"""Extended partial family geometry — 3 weight mods + 2 config-only variants.

Phase A: yarn / long / code via ChangeGeometry (same as partial experiment).
Phase B: sysprompt / temp by reusing Phase A's base engine (no weight reload).
Final: merge both into a single 5-variant GeoResult and emit per-domain breakdown.

Outputs examples/family_geometry_extended.json + terminal render.

Run: mamba run -n lmdiff python scripts/run_family_geometry_extended.py
"""
from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import numpy as np
import torch

from lmdiff import ChangeGeometry, Config, GeoResult, ProbeSet
from lmdiff.engine import InferenceEngine, release_cuda_cache
from lmdiff.report.json_report import write_json
from lmdiff.report.terminal import print_geometry


V01_PATH = Path(__file__).parent.parent / "lmdiff" / "probes" / "v01.json"
OUTPUT_DIR = Path(__file__).parent.parent / "examples"
OUTPUT_DIR.mkdir(exist_ok=True)

BASE = "meta-llama/Llama-2-7b-hf"
WEIGHT_VARIANTS = {
    "yarn": "NousResearch/Yarn-Llama-2-7b-128k",
    "long": "togethercomputer/LLaMA-2-7B-32K",
    "code": "codellama/CodeLlama-7b-hf",
}
MAX_NEW_TOKENS = 16
SEED = 42


# ── Phase B helper ──────────────────────────────────────────────────────

def _run_config_only(
    base_engine: InferenceEngine,
    variant_config: Config,
    prompts: list[str],
    max_new_tokens: int,
) -> tuple[list[float], bool]:
    """Compute raw δ for a config-only variant by reusing base engine's weights.

    Builds an InferenceEngine via __new__ (no _load, no weight reload) and
    hands it the base engine's model + tokenizer. The variant's config is
    set so _prefix_text / _decode_params honor its system_prompt / decode.
    """
    variant_engine = InferenceEngine.__new__(InferenceEngine)
    variant_engine.config = variant_config
    variant_engine.device = base_engine.device
    variant_engine._model = base_engine._model
    variant_engine._tokenizer = base_engine._tokenizer

    gen_v = variant_engine.generate(
        prompts, n_samples=1, max_new_tokens=max_new_tokens,
    )
    v_outputs = [c[0] for c in gen_v.completions]
    v_ids = [t[0] for t in gen_v.token_ids]

    score_b_of_v = base_engine.score(prompts, continuations=v_outputs)
    score_v_self = variant_engine.score(prompts, continuation_ids=v_ids)

    deltas: list[float] = []
    for i in range(len(prompts)):
        ce_bv = score_b_of_v.cross_entropies[i]
        ce_vv = score_v_self.cross_entropies[i]
        if math.isnan(ce_bv) or math.isnan(ce_vv):
            deltas.append(float("nan"))
        else:
            deltas.append(float(ce_bv - ce_vv))

    # Drop wrapper but NOT _model / _tokenizer — those are base's refs.
    del variant_engine, gen_v, score_b_of_v, score_v_self
    gc.collect()
    # No release_cuda_cache: nothing new was allocated on GPU.
    return deltas, False  # same model string → same tokenizer → no BPB


# ── Merge ───────────────────────────────────────────────────────────────

def _merge_results(
    result_weight_mod: GeoResult,
    config_only_deltas: dict[str, list[float]],
    config_only_bpb: dict[str, bool],
    prompts: list[str],
    variant_types: dict[str, str],
) -> GeoResult:
    """Combine weight-mod GeoResult with config-only raw δ into one 5-variant GeoResult."""
    assert result_weight_mod.metadata["n_skipped"] == 0, (
        "Phase A skipped probes — merging logic assumes n_skipped == 0 so "
        "that per_probe lookups align with prompts order."
    )

    all_variants = list(result_weight_mod.variant_names) + list(config_only_deltas.keys())
    n_total = len(prompts)

    raw_deltas: dict[str, list[float]] = {}
    for name in result_weight_mod.variant_names:
        per_probe_v = result_weight_mod.per_probe[name]
        raw_deltas[name] = [per_probe_v[p] for p in prompts]
    for name, deltas in config_only_deltas.items():
        assert len(deltas) == n_total, f"variant {name}: len(deltas)={len(deltas)}, n_total={n_total}"
        raw_deltas[name] = deltas

    valid_indices = [
        i for i in range(n_total)
        if all(not math.isnan(raw_deltas[name][i]) for name in all_variants)
    ]
    n_valid = len(valid_indices)

    change_vectors = {
        name: [raw_deltas[name][i] for i in valid_indices]
        for name in all_variants
    }
    per_probe = {
        name: {prompts[i]: raw_deltas[name][i] for i in valid_indices}
        for name in all_variants
    }
    magnitudes = {
        name: float(np.linalg.norm(change_vectors[name])) if n_valid > 0 else 0.0
        for name in all_variants
    }

    cosine_matrix: dict[str, dict[str, float]] = {a: {} for a in all_variants}
    vec_arrays = {name: np.asarray(change_vectors[name], dtype=float) for name in all_variants}
    for i, a in enumerate(all_variants):
        cosine_matrix[a][a] = 1.0 if magnitudes[a] > 0 else float("nan")
        for b in all_variants[i + 1:]:
            if magnitudes[a] == 0 or magnitudes[b] == 0:
                cos = float("nan")
            else:
                dot = float(np.dot(vec_arrays[a], vec_arrays[b]))
                cos = dot / (magnitudes[a] * magnitudes[b])
                cos = max(-1.0, min(1.0, cos))
            cosine_matrix[a][b] = cos
            cosine_matrix[b][a] = cos

    bpb_normalized = dict(result_weight_mod.metadata["bpb_normalized"])
    bpb_normalized.update(config_only_bpb)

    return GeoResult(
        base_name=result_weight_mod.base_name,
        variant_names=all_variants,
        n_probes=n_valid,
        magnitudes=magnitudes,
        cosine_matrix=cosine_matrix,
        change_vectors=change_vectors,
        per_probe=per_probe,
        metadata={
            "n_total_probes": n_total,
            "n_skipped": n_total - n_valid,
            "bpb_normalized": bpb_normalized,
            "max_new_tokens": result_weight_mod.metadata["max_new_tokens"],
            "probe_set_name": result_weight_mod.metadata.get("probe_set_name"),
            "probe_set_version": result_weight_mod.metadata.get("probe_set_version"),
            "sampling_seed": SEED,
            "variant_types": variant_types,
        },
    )


# ── Per-domain breakdown (reporting only) ───────────────────────────────

def _print_per_domain_breakdown(result: GeoResult, probes: ProbeSet) -> None:
    text_to_domain = {p.text: p.domain for p in probes}
    bpb_flags = result.metadata["bpb_normalized"]

    print("\nPer-domain magnitude breakdown")
    print("-" * 76)
    print(f"{'variant':<12s} {'type':<12s} {'bpb':<5s}  "
          f"{'math ‖δ‖':>10s}  {'know ‖δ‖':>10s}  {'code ‖δ‖':>10s}  {'overall':>10s}")
    print("-" * 76)

    v_types = result.metadata.get("variant_types", {})
    for name in result.variant_names:
        by_domain: dict[str, list[float]] = {"math": [], "knowledge": [], "code": []}
        for ptext, delta in result.per_probe[name].items():
            d = text_to_domain.get(ptext)
            if d in by_domain:
                by_domain[d].append(delta)
        mags = {
            d: float(np.linalg.norm(by_domain[d])) for d in ["math", "knowledge", "code"]
        }
        overall = float(np.linalg.norm(
            by_domain["math"] + by_domain["knowledge"] + by_domain["code"]
        ))
        bpb = "yes" if bpb_flags.get(name, False) else "no"
        print(f"{name:<12s} {v_types.get(name, '?'):<12s} {bpb:<5s}  "
              f"{mags['math']:>10.4f}  {mags['knowledge']:>10.4f}  "
              f"{mags['code']:>10.4f}  {overall:>10.4f}")


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    probes = ProbeSet.from_json(V01_PATH)
    prompts = probes.texts
    print(f"Loaded {len(probes)} probes across {probes.domains}")
    print(f"Base:   {BASE}")
    print(f"Weight variants: {WEIGHT_VARIANTS}")
    print(f"Config-only variants: sysprompt, temp")
    print(f"Seed: {SEED}, max_new_tokens: {MAX_NEW_TOKENS}")

    base_config = Config(model=BASE)
    variant_configs_weight = {
        name: Config(model=mid, name=name) for name, mid in WEIGHT_VARIANTS.items()
    }
    variant_config_sysprompt = Config(
        model=BASE,
        system_prompt="You are a helpful assistant.",
        name="sysprompt",
    )
    variant_config_temp = Config(
        model=BASE,
        decode={"strategy": "sample", "temperature": 1.5},
        name="temp",
    )

    # ── Phase A ────────────────────────────────────────────────────────
    print(f"\n=== Phase A: 3 weight-mod variants via ChangeGeometry ===")
    t0 = time.time()
    cg = ChangeGeometry(
        base=base_config,
        variants=variant_configs_weight,
        prompts=probes,
    )
    result_weight_mod = cg.analyze(max_new_tokens=MAX_NEW_TOKENS)
    phase_a_time = time.time() - t0
    print(f"Phase A done in {phase_a_time/60:.2f} min")

    # Sanity on Phase A (mirrors the partial experiment's expected state)
    assert result_weight_mod.metadata["n_skipped"] == 0, "Phase A should have no NaN skips"
    bpb_a = result_weight_mod.metadata["bpb_normalized"]
    assert bpb_a == {"yarn": False, "long": False, "code": True}, (
        f"Phase A bpb flags unexpected: {bpb_a}"
    )

    # cg.base_engine is lazy; accessing the property returns the one already
    # created inside analyze(). It is still holding the base weights.
    base_engine = cg.base_engine

    # ── Phase B ────────────────────────────────────────────────────────
    print(f"\n=== Phase B: 2 config-only variants (reusing base engine) ===")
    t0 = time.time()
    config_only_deltas: dict[str, list[float]] = {}
    config_only_bpb: dict[str, bool] = {}
    for name, v_config in [
        ("sysprompt", variant_config_sysprompt),
        ("temp", variant_config_temp),
    ]:
        print(f"  computing δ for '{name}' ...")
        t_v = time.time()
        deltas, use_bpb = _run_config_only(
            base_engine, v_config, prompts, MAX_NEW_TOKENS,
        )
        print(f"    {name}: {time.time() - t_v:.1f} s")
        config_only_deltas[name] = deltas
        config_only_bpb[name] = use_bpb
    phase_b_time = time.time() - t0
    print(f"Phase B done in {phase_b_time/60:.2f} min")

    # Sanity on Phase B
    for name, deltas in config_only_deltas.items():
        assert len(deltas) == len(prompts), (
            f"Phase B variant '{name}' has {len(deltas)} deltas, expected {len(prompts)}"
        )
    for name, flag in config_only_bpb.items():
        assert flag is False, f"Phase B variant '{name}' should not be BPB-normalized"

    # ── Merge ──────────────────────────────────────────────────────────
    variant_types = {
        "yarn": "weight_mod",
        "long": "weight_mod",
        "code": "weight_mod",
        "sysprompt": "config_only",
        "temp": "config_only",
    }
    final_result = _merge_results(
        result_weight_mod, config_only_deltas, config_only_bpb, prompts, variant_types,
    )

    # Release base engine now that Phase B is done
    del base_engine, cg, result_weight_mod
    gc.collect()
    release_cuda_cache()

    # ── Output ─────────────────────────────────────────────────────────
    print_geometry(final_result)
    _print_per_domain_breakdown(final_result, probes)

    out_path = OUTPUT_DIR / "family_geometry_extended.json"
    write_json(final_result, out_path)
    print(f"\nJSON written to {out_path}")

    # Timing footer
    print(f"\n=== Timing ===")
    print(f"  Phase A (3 weight-mod variants, full reload each): {phase_a_time/60:.2f} min")
    print(f"  Phase B (2 config-only variants, reused base):     {phase_b_time/60:.2f} min")
    print(f"  Total analyze time:                                {(phase_a_time + phase_b_time)/60:.2f} min")


if __name__ == "__main__":
    main()
