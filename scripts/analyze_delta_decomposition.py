"""Decompose δ into constant + selective components for each variant.

Reads examples/family_geometry_extended.json.
Prints three analyses:
  1. Coefficient of variation per variant
  2. Constant-offset energy fraction per variant
  3. Selective cosine matrix (= Pearson correlation) + original for comparison

No side effects — pure numpy analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


JSON_PATH = Path(__file__).parent.parent / "examples" / "family_geometry_extended.json"


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)

    variant_names = list(data["variant_names"])
    change_vectors = data["change_vectors"]
    magnitudes = data["magnitudes"]
    original_cos = data["cosine_matrix"]

    vecs = {name: np.asarray(change_vectors[name], dtype=float) for name in variant_names}
    n = len(vecs[variant_names[0]])
    assert all(len(vecs[name]) == n for name in variant_names), "Length mismatch"

    print(f"Loaded {len(variant_names)} variants, {n} probes each")
    print(f"Variants: {variant_names}")

    # ============================================================
    # Check 1: Coefficient of variation
    # ============================================================
    print("\n" + "=" * 60)
    print("Check 1: Coefficient of variation (CV = std / |mean|)")
    print("=" * 60)
    print(f"{'variant':<12} {'mean':>10} {'std':>10} {'CV':>10} {'‖δ‖':>10}")
    print("-" * 56)
    for name in variant_names:
        v = vecs[name]
        mean = float(v.mean())
        std = float(v.std())
        mag = float(np.linalg.norm(v))
        cv = std / abs(mean) if abs(mean) > 1e-12 else float("inf")
        print(f"{name:<12} {mean:>+10.4f} {std:>10.4f} {cv:>10.4f} {mag:>10.4f}")

    # ============================================================
    # Check 2: Constant-offset energy fraction
    # ============================================================
    print("\n" + "=" * 60)
    print("Check 2: Constant-offset energy fraction")
    print("=" * 60)
    print("Decompose δ = c·𝟙 + ε where c = mean(δ)")
    print("constant_fraction = ‖c·𝟙‖² / ‖δ‖²  (high → uniform offset dominates)")
    print()
    print(f"{'variant':<12} {'c (mean)':>10} {'const_frac':>12} {'sel_frac':>10} {'‖ε‖':>10}")
    print("-" * 58)
    for name in variant_names:
        v = vecs[name]
        c = float(v.mean())
        total_energy = float((v ** 2).sum())
        constant_energy = c ** 2 * n  # ‖c·𝟙‖² = c² * n
        residual = v - c
        residual_energy = float((residual ** 2).sum())
        selective_norm = float(np.linalg.norm(residual))
        constant_fraction = constant_energy / total_energy if total_energy > 0 else 0.0
        selective_fraction = residual_energy / total_energy if total_energy > 0 else 0.0
        # sanity: constant_fraction + selective_fraction should equal 1.0
        assert abs(constant_fraction + selective_fraction - 1.0) < 1e-9, \
            f"Energy decomposition broken for {name}"
        print(
            f"{name:<12} {c:>+10.4f} {constant_fraction:>12.4f} "
            f"{selective_fraction:>10.4f} {selective_norm:>10.4f}"
        )

    # ============================================================
    # Check 3: Selective cosine matrix (Pearson correlation)
    # ============================================================
    print("\n" + "=" * 60)
    print("Check 3: Selective cosine matrix")
    print("=" * 60)
    print("cos(δ_A − mean_A·𝟙, δ_B − mean_B·𝟙)  (== Pearson r)")
    print("Low selective cos + high original cos → agreement was driven by")
    print("constant offset, not by selective pattern match.")

    centered = {name: vecs[name] - vecs[name].mean() for name in variant_names}

    print()
    print(f"{'Selective cos':<12}", end="")
    for name in variant_names:
        print(f"{name:>11}", end="")
    print()
    for a in variant_names:
        print(f"{a:<12}", end="")
        for b in variant_names:
            if a == b:
                val = 1.0
            else:
                dot = float((centered[a] * centered[b]).sum())
                na = float(np.linalg.norm(centered[a]))
                nb = float(np.linalg.norm(centered[b]))
                val = dot / (na * nb) if na > 0 and nb > 0 else float("nan")
                val = max(-1.0, min(1.0, val))
            print(f"{val:>+11.4f}", end="")
        print()

    # ============================================================
    # Side-by-side comparison
    # ============================================================
    print("\n" + "=" * 60)
    print("Side-by-side: original cos vs selective cos")
    print("=" * 60)
    print(f"{'pair':<30} {'original':>10} {'selective':>10} {'diff':>10}")
    print("-" * 62)
    for i, a in enumerate(variant_names):
        for b in variant_names[i + 1:]:
            orig = original_cos[a][b]
            dot = float((centered[a] * centered[b]).sum())
            na = float(np.linalg.norm(centered[a]))
            nb = float(np.linalg.norm(centered[b]))
            sel = dot / (na * nb) if na > 0 and nb > 0 else float("nan")
            sel = max(-1.0, min(1.0, sel))
            diff = sel - orig
            print(f"{a + ' vs ' + b:<30} {orig:>+10.4f} {sel:>+10.4f} {diff:>+10.4f}")


if __name__ == "__main__":
    main()
