"""4-variant calibration empirical verification — v0.4.1 audit.

Tests the §8.2 hypothesis that Formula B (current sqrt(Σδ²/ΣT)) and
Formula C (proposed sqrt(mean(δ²))) produce numerically-identical
results on the 4-variant calibration baseline (where every probe is
in-context and every (variant, domain) is `full`).

This script runs on tests/fixtures/calibration_v032_baseline.json,
which is the byte-equivalence fixture for the existing 4-variant
test. If B and C disagree by more than the test's 1e-6 tolerance,
the calibration fixture must be regenerated alongside v0.4.1.

Run:
    mamba run -n lmdiff python docs/internal/v041_audit_4variant_check.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CAL = REPO / "tests" / "fixtures" / "calibration_v032_baseline.json"
TOL = 1e-6  # Existing 4-variant test tolerance.


def load() -> dict:
    return json.loads(CAL.read_text(encoding="utf-8"))


def main() -> int:
    d = load()
    deltas_by_v: dict[str, list[float]] = d["change_vectors"]
    domains = list(d["probe_domains"])
    tokens = list(d["avg_tokens_per_probe"])

    n = len(domains)
    assert all(len(v) == n for v in deltas_by_v.values())
    assert len(tokens) == n

    by_d: dict[str, list[int]] = defaultdict(list)
    for i, dom in enumerate(domains):
        by_d[dom].append(i)

    variants = list(deltas_by_v.keys())
    domains_sorted = sorted(by_d)

    print("=" * 78)
    print(f"4-VARIANT CALIBRATION BASELINE — Formula B vs C empirical check")
    print(f"  fixture: {CAL.relative_to(REPO)}")
    print(f"  tolerance (existing test): {TOL}")
    print("=" * 78)

    # ── T_i variance per (variant, domain) — actually per domain since
    # all variants share the same probe set / token counts ──
    print()
    print("Within-domain T_i variance (same across variants — token counts")
    print("come from `base_engine.token_count(p)` once per probe):")
    print(f"  {'domain':<14} {'n':>4} {'T_min':>6} {'T_max':>6} {'T_mean':>7} "
          f"{'T_std':>7} {'CoV':>6}")
    for dom in domains_sorted:
        idxs = by_d[dom]
        ts = [tokens[i] for i in idxs]
        mean = sum(ts) / len(ts)
        var = sum((t - mean) ** 2 for t in ts) / len(ts)
        std = math.sqrt(var)
        cov = std / mean if mean > 0 else 0.0
        print(f"  {dom:<14} {len(idxs):>4} {min(ts):>6.0f} {max(ts):>6.0f} "
              f"{mean:>7.1f} {std:>7.1f} {cov:>6.3f}")

    # ── Theoretical prediction ──
    # pdn_B = sqrt(Σδ² / ΣT) = sqrt(σ²·n / (n·T̄)) = σ/√T̄
    # pdn_C = sqrt(Σδ² / n)  = σ
    # ratio: pdn_C / pdn_B = √T̄
    print()
    print("Theoretical: pdn_C / pdn_B = √T̄_d for each domain")
    print("  (because pdn_B has an extra /√T̄ factor vs the unit-clean form)")

    print()
    print("─" * 78)
    print(" pdn_B vs pdn_C per (variant, domain)")
    print("─" * 78)
    max_abs_diff_pdn = 0.0
    max_abs_diff_share = 0.0
    any_breaks_tolerance = False

    for v in variants:
        dv = deltas_by_v[v]
        b_row: dict[str, float] = {}
        c_row: dict[str, float] = {}
        for dom, idxs in by_d.items():
            ssq = sum(dv[i] * dv[i] for i in idxs)
            sum_t = sum(tokens[i] for i in idxs)
            b_row[dom] = math.sqrt(ssq / sum_t) if sum_t > 0 else 0.0
            c_row[dom] = math.sqrt(ssq / len(idxs)) if idxs else 0.0

        # shares
        b_sum_sq = sum(x * x for x in b_row.values())
        c_sum_sq = sum(x * x for x in c_row.values())
        b_share = {dom: (b_row[dom] ** 2) / b_sum_sq if b_sum_sq > 0 else 0.0
                   for dom in domains_sorted}
        c_share = {dom: (c_row[dom] ** 2) / c_sum_sq if c_sum_sq > 0 else 0.0
                   for dom in domains_sorted}

        print(f"\n▶ {v}")
        print(f"    {'domain':<14} {'pdn_B':>10} {'pdn_C':>10} {'C/B':>7} "
              f"{'√T̄':>7}  {'share_B':>9} {'share_C':>9}  {'Δshare':>8}")
        for dom in domains_sorted:
            ratio = c_row[dom] / b_row[dom] if b_row[dom] > 0 else 0.0
            T_mean = sum(tokens[i] for i in by_d[dom]) / len(by_d[dom])
            sqrt_T = math.sqrt(T_mean)
            pdn_diff = abs(c_row[dom] - b_row[dom])
            share_diff = abs(c_share[dom] - b_share[dom])
            max_abs_diff_pdn = max(max_abs_diff_pdn, pdn_diff)
            max_abs_diff_share = max(max_abs_diff_share, share_diff)
            tol_break = " *" if pdn_diff > TOL else ""
            if pdn_diff > TOL:
                any_breaks_tolerance = True
            print(f"    {dom:<14} {b_row[dom]:>10.6f} {c_row[dom]:>10.6f} "
                  f"{ratio:>7.3f} {sqrt_T:>7.3f}  "
                  f"{b_share[dom]*100:>7.2f}%  {c_share[dom]*100:>7.2f}%  "
                  f"{share_diff*100:>+7.2f}pp{tol_break}")

    # ── Verdict ──
    print()
    print("=" * 78)
    print(" VERDICT")
    print("=" * 78)
    print(f"  Max |pdn_C - pdn_B| over all (variant, domain): "
          f"{max_abs_diff_pdn:.4e}")
    print(f"  Max |share_C - share_B| (absolute, not pp): "
          f"{max_abs_diff_share:.4e}")
    print(f"  Existing 4-variant tolerance: {TOL}")
    print()
    if any_breaks_tolerance:
        print("  ✗ HYPOTHESIS BROKEN.")
        print("    pdn values shift by factor √T̄_d (theoretical prediction).")
        print("    Max |pdn diff| exceeds the 1e-6 calibration tolerance.")
        print("    Implication: 4-variant calibration fixture must be")
        print("    regenerated alongside the 7-variant fixture when v0.4.1")
        print("    implementation lands.")
    else:
        print("  ✓ HYPOTHESIS HOLDS — pdn values within 1e-6 across (v,d).")
        print("    4-variant fixture does NOT need regeneration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
