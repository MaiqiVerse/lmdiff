#!/usr/bin/env python
"""Diagnose per-task magnitude normalization for lm-eval family experiments.

Read-only analysis. Given a family_geometry summary JSON (produced by
run_family_geometry_lm_eval.py), computes:

  1. Per-task average probe token count (using the same lm-eval task
     loader + seed as the original experiment, so the probe sample
     matches exactly).
  2. Pearson correlation between per-task avg probe length and per-task
     ‖δ‖ for each variant. Spearman too if scipy is available.
  3. A per-token normalized magnitude table:
         norm_mag[variant][task] = ‖δ_task‖ / sqrt(n_probes × avg_tokens)
     interpreting the L2 norm as RMS per-token CE difference.
  4. A task-feature PCA on the normalized magnitudes for comparison
     against the un-normalized PCA already in hand.

Outputs to stdout + writes a JSON with all intermediate numbers so the
result can be loaded into plotting later.

Usage:
    mamba run -n lmdiff python scripts/diagnose_probe_length_effect.py \\
        examples/family_lm_eval_weight_mods.json \\
        --output examples/family_lm_eval_weight_mods_diagnostic.json

Requires: pip install lmdiff-kit[lm-eval]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _import_transformers_tokenizer(model_id: str):
    """Load tokenizer; sys.exit(2) with hint on failure."""
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except ImportError:
        print(
            "ERROR: transformers not installed. Install with: pip install transformers",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to load tokenizer {model_id!r}: {exc}", file=sys.stderr)
        sys.exit(2)


def _import_from_lm_eval():
    try:
        from lmdiff.probes.adapters import from_lm_eval
    except ImportError as exc:
        print(
            f"ERROR importing from_lm_eval: {exc}\n"
            "Install with: pip install lmdiff-kit[lm-eval]",
            file=sys.stderr,
        )
        sys.exit(2)
    return from_lm_eval


def _pearson(x: list[float], y: list[float]) -> tuple[float, float]:
    """Pearson r and two-sided p-value via scipy if available, else r-only."""
    try:
        from scipy.stats import pearsonr  # type: ignore[import-not-found]
    except ImportError:
        # Fall back to numpy correlation; no p-value.
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.std() == 0 or y_arr.std() == 0:
            return float("nan"), float("nan")
        r = float(np.corrcoef(x_arr, y_arr)[0, 1])
        return r, float("nan")
    res = pearsonr(x, y)
    # scipy returns either (r, p) tuple or PearsonRResult depending on version.
    try:
        r = float(res.statistic)
        p = float(res.pvalue)
    except AttributeError:
        r, p = float(res[0]), float(res[1])
    return r, p


def _spearman(x: list[float], y: list[float]) -> tuple[float, float] | tuple[None, None]:
    try:
        from scipy.stats import spearmanr  # type: ignore[import-not-found]
    except ImportError:
        return None, None
    res = spearmanr(x, y)
    try:
        return float(res.statistic), float(res.pvalue)
    except AttributeError:
        return float(res[0]), float(res[1])


def _summary_stats(values: list[int]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-token magnitude normalization diagnostic for lm-eval family experiments.",
    )
    parser.add_argument("summary_json", type=Path,
                        help="Path to family_geometry summary JSON.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: <input_stem>_diagnostic.json).")
    parser.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf",
                        help="HF id for token counting (default: meta-llama/Llama-2-7b-hf).")
    args = parser.parse_args(argv)

    if not args.summary_json.exists():
        print(f"ERROR: {args.summary_json} not found", file=sys.stderr)
        return 2

    with open(args.summary_json, encoding="utf-8") as f:
        summary = json.load(f)

    tasks = summary["tasks"]
    limit_per_task = summary["limit_per_task"]
    seed = summary["seed"]
    variants = summary["variants"]
    delta_mag = summary["delta_magnitude_by_variant"]
    raw_total = summary["magnitudes_total"]
    accuracy = summary.get("accuracy_by_variant", {})

    output_path = args.output or args.summary_json.with_name(
        args.summary_json.stem + "_diagnostic.json"
    )

    print(f"=== probe-length diagnostic ===")
    print(f"Source: {args.summary_json}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Tasks: {tasks}")
    print(f"limit_per_task={limit_per_task}, seed={seed}")
    print(f"Variants: {sorted(variants)}")
    print()

    # --- Step 1: tokenize probes per task ---
    from_lm_eval = _import_from_lm_eval()
    tok = _import_transformers_tokenizer(args.tokenizer)

    print("Loading + tokenizing probe sets per task ...")
    per_task_token_stats: dict[str, dict[str, float | int]] = {}
    for task in tasks:
        print(f"  {task} ...", flush=True)
        ps = from_lm_eval(task, limit=limit_per_task, seed=seed)
        token_lens = [len(tok.encode(p.text, add_special_tokens=False)) for p in ps]
        per_task_token_stats[task] = _summary_stats(token_lens)
        s = per_task_token_stats[task]
        print(f"    n={s['n']}  mean={s['mean']:.1f}  min={s['min']}  "
              f"max={s['max']}  median={s['median']:.1f}  std={s['std']:.1f}")

    avg_tokens = {t: float(per_task_token_stats[t]["mean"]) for t in tasks}

    # --- Step 2: correlations ---
    print("\n" + "=" * 68)
    print("Diagnostic 1: Pearson / Spearman r between avg_tokens and ‖δ‖")
    print("=" * 68)
    print(f"{'variant':<12} {'pearson_r':>10} {'p-value':>11} "
          f"{'spearman_r':>11} {'p-value':>11}")
    print("-" * 60)
    correlations: dict[str, dict[str, float | None]] = {}
    for v in sorted(variants):
        per_task = delta_mag[v]
        x = [avg_tokens[t] for t in tasks]
        y = [per_task[t] for t in tasks]
        pr, pp = _pearson(x, y)
        sr, sp = _spearman(x, y)
        correlations[v] = {
            "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp,
        }
        sr_s = f"{sr:>+11.4f}" if sr is not None else f"{'n/a':>11}"
        sp_s = f"{sp:>11.4g}" if sp is not None else f"{'n/a':>11}"
        print(f"{v:<12} {pr:>+10.4f} {pp:>11.4g} {sr_s} {sp_s}")

    # --- Step 3: normalized magnitudes ---
    print("\n" + "=" * 68)
    print("Diagnostic 2: per-token normalized magnitudes")
    print("  norm_mag = raw / sqrt(n_probes × avg_tokens)")
    print("=" * 68)

    print("\n--- Raw ‖δ‖ per task ---")
    print(f"{'task':<35} " + " ".join(f"{v:>9}" for v in sorted(variants)))
    print("-" * 70)
    for t in tasks:
        row = [delta_mag[v][t] for v in sorted(variants)]
        print(f"{t:<35} " + " ".join(f"{x:>9.4f}" for x in row))

    norm_mag: dict[str, dict[str, float]] = {v: {} for v in sorted(variants)}
    for v in sorted(variants):
        for t in tasks:
            denom = math.sqrt(limit_per_task * avg_tokens[t])
            norm_mag[v][t] = delta_mag[v][t] / denom if denom > 0 else float("nan")

    print("\n--- Per-token normalized ‖δ‖ ---")
    print(f"{'task':<35} " + " ".join(f"{v:>9}" for v in sorted(variants)))
    print("-" * 70)
    for t in tasks:
        row = [norm_mag[v][t] for v in sorted(variants)]
        print(f"{t:<35} " + " ".join(f"{x:>9.4f}" for x in row))

    # Normalized total: equal-weight L2 over the 5 normalized per-task magnitudes.
    norm_total: dict[str, float] = {}
    for v in sorted(variants):
        norm_total[v] = math.sqrt(sum(x * x for x in norm_mag[v].values()))

    print("\n--- Total magnitude (raw vs normalized) ---")
    print(f"{'variant':<12} {'raw_total':>12} {'norm_total':>12} {'ratio':>10}")
    print("-" * 50)
    for v in sorted(variants):
        ratio = raw_total[v] / norm_total[v] if norm_total[v] > 0 else float("nan")
        print(f"{v:<12} {raw_total[v]:>12.4f} {norm_total[v]:>12.4f} {ratio:>10.2f}")

    # --- Step 4: task-feature PCA on normalized magnitudes ---
    print("\n" + "=" * 68)
    print("Diagnostic 2.5: task-feature PCA on normalized magnitudes")
    print("=" * 68)

    sorted_variants = sorted(variants)
    X = np.asarray(
        [[norm_mag[v][t] for t in tasks] for v in sorted_variants],
        dtype=float,
    )
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords_full = U * S
    n_components = 2
    total_var = float((S ** 2).sum())
    ratios = tuple(
        float((S[i] ** 2) / total_var) if i < len(S) else 0.0
        for i in range(n_components)
    )
    coords = {
        name: tuple(float(x) for x in coords_full[i, :n_components])
        for i, name in enumerate(sorted_variants)
    }

    print(f"singular values: {[round(s, 4) for s in S.tolist()]}")
    print(f"explained_variance_ratio (PC1, PC2): "
          f"({ratios[0]:.4f}, {ratios[1]:.4f})")

    print("\n--- PC1 loadings (which task drives PC1) ---")
    for i, t in enumerate(tasks):
        print(f"  {t:<35} {Vt[0, i]:+.4f}")
    print("\n--- PC2 loadings ---")
    for i, t in enumerate(tasks):
        print(f"  {t:<35} {Vt[1, i]:+.4f}")

    print("\n--- variant coords in normalized PC space ---")
    for v in sorted_variants:
        c = coords[v]
        print(f"  {v:<12} ({c[0]:+.4f}, {c[1]:+.4f})")

    # --- Write JSON ---
    diagnostic = {
        "source_json": str(args.summary_json),
        "tokenizer": args.tokenizer,
        "tasks": list(tasks),
        "limit_per_task": limit_per_task,
        "seed": seed,
        "variants": dict(variants),
        "per_task_token_stats": per_task_token_stats,
        "pearson_spearman_correlations": correlations,
        "raw_magnitude_by_variant": delta_mag,
        "raw_total_by_variant": raw_total,
        "normalized_magnitude_by_variant": norm_mag,
        "normalized_total_by_variant": norm_total,
        "task_feature_pca_normalized": {
            "loadings_pc1": {t: float(Vt[0, i]) for i, t in enumerate(tasks)},
            "loadings_pc2": {t: float(Vt[1, i]) for i, t in enumerate(tasks)},
            "explained_variance_ratio": list(ratios),
            "singular_values": [float(x) for x in S.tolist()],
            "coords": {v: list(coords[v]) for v in sorted_variants},
        },
        "accuracy_echo": accuracy,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nDiagnostic JSON: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
