"""Analyze partial family geometry JSON output.

Two diagnostics:
1. Per-domain magnitude breakdown (splits 90-d δ vector into math/knowledge/code)
2. High-δ probe sampling (top-N probes by |δ| per variant)

Run: mamba run -n lmdiff python scripts/analyze_family_geometry.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from lmdiff.probes.loader import ProbeSet

JSON_PATH = Path(__file__).parent.parent / "examples" / "family_geometry_partial.json"
V01_PATH = Path(__file__).parent.parent / "lmdiff" / "probes" / "v01.json"
TOP_N = 5  # top-N high-|δ| probes per variant


def main() -> None:
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    variant_names = data["variant_names"]
    per_probe = data["per_probe"]
    bpb_flags = data["metadata"]["bpb_normalized"]

    # Build probe_text -> domain map from v01
    ps = ProbeSet.from_json(V01_PATH)
    text_to_domain = {p.text: p.domain for p in ps}

    # ── Part 1: per-domain magnitude breakdown ─────────────────────

    print(f"\n{'=' * 70}")
    print("  PART 1: Per-domain magnitude breakdown")
    print(f"{'=' * 70}\n")

    print(f"{'variant':<10s} {'bpb':<5s}  "
          f"{'math ‖δ‖':>10s}  {'know ‖δ‖':>10s}  {'code ‖δ‖':>10s}  "
          f"{'overall':>10s}")
    print("-" * 70)

    per_domain_mag: dict[str, dict[str, float]] = {}
    for v in variant_names:
        per_domain_mag[v] = {}
        # Split this variant's δ values by domain
        by_domain: dict[str, list[float]] = {"math": [], "knowledge": [], "code": []}
        unknown = 0
        for probe_text, delta in per_probe[v].items():
            if delta is None:  # JSON null for NaN — shouldn't happen post-filter
                continue
            domain = text_to_domain.get(probe_text)
            if domain in by_domain:
                by_domain[domain].append(delta)
            else:
                unknown += 1

        for domain in ["math", "knowledge", "code"]:
            vec = np.array(by_domain[domain], dtype=float)
            per_domain_mag[v][domain] = float(np.linalg.norm(vec))

        # Recompute overall for cross-check (should match JSON magnitudes)
        all_deltas = by_domain["math"] + by_domain["knowledge"] + by_domain["code"]
        overall = float(np.linalg.norm(np.array(all_deltas, dtype=float)))

        bpb = "yes" if bpb_flags.get(v, False) else "no"
        print(f"{v:<10s} {bpb:<5s}  "
              f"{per_domain_mag[v]['math']:>10.4f}  "
              f"{per_domain_mag[v]['knowledge']:>10.4f}  "
              f"{per_domain_mag[v]['code']:>10.4f}  "
              f"{overall:>10.4f}")

        if unknown > 0:
            print(f"  [warn] {unknown} probes in variant '{v}' had no matching domain")

    # Cross-check: does ‖δ_math‖² + ‖δ_know‖² + ‖δ_code‖² == ‖δ_total‖²?
    # (Pythagorean, because the three subsets partition the 90-d space on disjoint axes)
    print("\n  [sanity check: per-domain² sum vs overall²]")
    for v in variant_names:
        mag_expected_sq = sum(per_domain_mag[v][d] ** 2 for d in ["math", "knowledge", "code"])
        mag_reported = data["magnitudes"][v]
        mag_reported_sq = mag_reported ** 2
        rel_diff = abs(mag_expected_sq - mag_reported_sq) / mag_reported_sq
        status = "OK" if rel_diff < 1e-6 else "MISMATCH"
        print(f"    {v}: Σd²={mag_expected_sq:.4f}  reported²={mag_reported_sq:.4f}  [{status}]")

    # ── Part 2: high-δ probe sampling ──────────────────────────────

    print(f"\n{'=' * 70}")
    print(f"  PART 2: Top-{TOP_N} high-|δ| probes per variant")
    print(f"{'=' * 70}\n")

    for v in variant_names:
        print(f"\n--- variant: {v} ({'BPB' if bpb_flags.get(v, False) else 'nat'}) ---")

        # Sort probes by absolute δ descending
        sorted_probes = sorted(
            per_probe[v].items(),
            key=lambda kv: abs(kv[1]) if kv[1] is not None else 0,
            reverse=True,
        )
        for probe_text, delta in sorted_probes[:TOP_N]:
            domain = text_to_domain.get(probe_text, "?")
            # Truncate probe text for display
            display = probe_text.replace("\n", "\\n")
            if len(display) > 50:
                display = display[:47] + "..."
            print(f"  δ={delta:+8.4f}  [{domain:>9s}]  {display!r}")

        # Also show domain distribution of top-N
        top_domains = [
            text_to_domain.get(p, "?") for p, _ in sorted_probes[:TOP_N]
        ]
        from collections import Counter
        dist = Counter(top_domains)
        print(f"  top-{TOP_N} domain distribution: {dict(dist)}")

    # ── Part 3: per-domain mean / median δ (sign matters) ──────────

    print(f"\n{'=' * 70}")
    print("  PART 3: Per-domain δ mean/median (positive = base more surprised)")
    print(f"{'=' * 70}\n")

    print(f"{'variant':<10s} {'domain':<10s}  "
          f"{'mean':>8s}  {'median':>8s}  {'std':>8s}  {'n':>4s}")
    print("-" * 60)

    for v in variant_names:
        for domain in ["math", "knowledge", "code"]:
            deltas = []
            for probe_text, delta in per_probe[v].items():
                if delta is None:
                    continue
                if text_to_domain.get(probe_text) == domain:
                    deltas.append(delta)
            if not deltas:
                continue
            arr = np.array(deltas)
            print(f"{v:<10s} {domain:<10s}  "
                  f"{arr.mean():+8.4f}  {np.median(arr):+8.4f}  "
                  f"{arr.std():>8.4f}  {len(arr):>4d}")

    print()


if __name__ == "__main__":
    main()
