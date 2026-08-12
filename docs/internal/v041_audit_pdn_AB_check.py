"""4-variant calibration: plain vs token-weighted RMS empirical compare.

Per audit Q9.10, we choose between two pdn variants for v0.4.1:

  A. plain unweighted:  pdn_A[v][d] = sqrt(mean(δ²))            = sqrt(Σδ² / n_d)
  B. token-weighted:    pdn_B[v][d] = sqrt(Σ T_i·δ_i² / Σ T_i)

Both have units nats/token (clean). They diverge when within-domain T_i
variance is non-negligible AND T_i correlates with δ_i² across probes
(longer or shorter probes systematically having bigger or smaller per-
token drift). At CoV ≈ 0.4–0.5 in the 4-variant calibration, the gap is
empirically a few percent of the share value.

This script runs on tests/fixtures/calibration_v032_baseline.json and
prints (variant, domain) pdn_A, pdn_B, share_A, share_B side-by-side.

Run:
    mamba run -n lmdiff python docs/internal/v041_audit_pdn_AB_check.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CAL = REPO / "tests" / "fixtures" / "calibration_v032_baseline.json"


def load() -> dict:
    return json.loads(CAL.read_text(encoding="utf-8"))


def main() -> int:
    d = load()
    deltas_by_v: dict[str, list[float]] = d["change_vectors"]
    domains = list(d["probe_domains"])
    tokens = list(d["avg_tokens_per_probe"])

    n = len(domains)
    by_d: dict[str, list[int]] = defaultdict(list)
    for i, dom in enumerate(domains):
        by_d[dom].append(i)

    variants = list(deltas_by_v.keys())
    domains_sorted = sorted(by_d)

    print("=" * 78)
    print(f"4-VARIANT — plain vs token-weighted RMS (Q9.10)")
    print(f"  fixture: {CAL.relative_to(REPO)}")
    print("=" * 78)
    print()

    # ── correlation diagnostic per (variant, domain) ──
    # If T_i and δ_i² are uncorrelated within a domain, A ≈ B.
    # If positively correlated, B > A. Negative ⇒ B < A.
    print("Corr(T_i, δ_i²) within each (variant, domain) — predicts A vs B sign:")
    print(f"  {'variant':<10} " + " ".join(f"{d:>14}" for d in domains_sorted))
    for v in variants:
        dv = deltas_by_v[v]
        cors = []
        for dom in domains_sorted:
            idxs = by_d[dom]
            xs = [tokens[i] for i in idxs]
            ys = [dv[i] * dv[i] for i in idxs]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = math.sqrt(sum((x - mx) ** 2 for x in xs)
                            * sum((y - my) ** 2 for y in ys))
            cors.append(num / den if den > 0 else 0.0)
        print(f"  {v:<10} " + " ".join(f"{c:>+14.3f}" for c in cors))

    print()
    print("─" * 78)
    print(" pdn_A vs pdn_B per (variant, domain) and the resulting share")
    print("─" * 78)

    max_share_diff_pp = 0.0

    for v in variants:
        dv = deltas_by_v[v]
        pdn_a: dict[str, float] = {}
        pdn_b: dict[str, float] = {}
        for dom, idxs in by_d.items():
            ssq_unweighted = sum(dv[i] * dv[i] for i in idxs)
            n_d = len(idxs)
            pdn_a[dom] = math.sqrt(ssq_unweighted / n_d) if n_d else 0.0

            ssq_weighted = sum(tokens[i] * dv[i] * dv[i] for i in idxs)
            sum_t = sum(tokens[i] for i in idxs)
            pdn_b[dom] = math.sqrt(ssq_weighted / sum_t) if sum_t > 0 else 0.0

        # shares
        sa_total = sum(x * x for x in pdn_a.values())
        sb_total = sum(x * x for x in pdn_b.values())
        share_a = {d: pdn_a[d] ** 2 / sa_total if sa_total > 0 else 0.0
                   for d in domains_sorted}
        share_b = {d: pdn_b[d] ** 2 / sb_total if sb_total > 0 else 0.0
                   for d in domains_sorted}

        print(f"\n▶ {v}")
        print(f"    {'domain':<14} {'pdn_A':>10} {'pdn_B':>10} {'B/A':>7}  "
              f"{'share_A':>9} {'share_B':>9} {'Δshare_pp':>10}")
        for dom in domains_sorted:
            ratio = pdn_b[dom] / pdn_a[dom] if pdn_a[dom] > 0 else 0.0
            diff_pp = (share_b[dom] - share_a[dom]) * 100
            max_share_diff_pp = max(max_share_diff_pp, abs(diff_pp))
            print(f"    {dom:<14} {pdn_a[dom]:>10.5f} {pdn_b[dom]:>10.5f} "
                  f"{ratio:>7.3f}  {share_a[dom]*100:>7.2f}%  "
                  f"{share_b[dom]*100:>7.2f}%  {diff_pp:>+8.2f}pp")

    # ── ALSO: under v0.4.1 expected case (long-context dropped) ──
    print()
    print("─" * 78)
    print(" Under v0.4.1 validity framework (long-context dropped from base):")
    print("─" * 78)

    in_context_doms = [d for d in domains_sorted if d != "long-context"]
    max_share_diff_pp_post = 0.0
    for v in variants:
        dv = deltas_by_v[v]
        pdn_a: dict[str, float] = {}
        pdn_b: dict[str, float] = {}
        for dom in in_context_doms:
            idxs = by_d[dom]
            ssq_unweighted = sum(dv[i] * dv[i] for i in idxs)
            n_d = len(idxs)
            pdn_a[dom] = math.sqrt(ssq_unweighted / n_d) if n_d else 0.0
            ssq_weighted = sum(tokens[i] * dv[i] * dv[i] for i in idxs)
            sum_t = sum(tokens[i] for i in idxs)
            pdn_b[dom] = math.sqrt(ssq_weighted / sum_t) if sum_t > 0 else 0.0
        sa_total = sum(x * x for x in pdn_a.values())
        sb_total = sum(x * x for x in pdn_b.values())
        share_a = {d: pdn_a[d] ** 2 / sa_total if sa_total > 0 else 0.0
                   for d in in_context_doms}
        share_b = {d: pdn_b[d] ** 2 / sb_total if sb_total > 0 else 0.0
                   for d in in_context_doms}

        print(f"\n▶ {v}")
        print(f"    {'domain':<14} {'pdn_A':>10} {'pdn_B':>10} {'B/A':>7}  "
              f"{'share_A':>9} {'share_B':>9} {'Δshare_pp':>10}")
        for dom in in_context_doms:
            ratio = pdn_b[dom] / pdn_a[dom] if pdn_a[dom] > 0 else 0.0
            diff_pp = (share_b[dom] - share_a[dom]) * 100
            max_share_diff_pp_post = max(max_share_diff_pp_post, abs(diff_pp))
            print(f"    {dom:<14} {pdn_a[dom]:>10.5f} {pdn_b[dom]:>10.5f} "
                  f"{ratio:>7.3f}  {share_a[dom]*100:>7.2f}%  "
                  f"{share_b[dom]*100:>7.2f}%  {diff_pp:>+8.2f}pp")
        # Biggest-move story preservation check
        a_max = max(in_context_doms, key=lambda x: share_a[x])
        b_max = max(in_context_doms, key=lambda x: share_b[x])
        flip = " ⚠ ranking flipped" if a_max != b_max else ""
        print(f"      biggest under A: {a_max} ({share_a[a_max]*100:.1f}%)   "
              f"under B: {b_max} ({share_b[b_max]*100:.1f}%){flip}")

    # ── verdict ──
    print()
    print("=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    print(f"  Max |share_A − share_B| WITH long-context (pre-v0.4.1 view): "
          f"{max_share_diff_pp:.2f}pp")
    print(f"  Max |share_A − share_B| WITHOUT long-context (v0.4.1 view):  "
          f"{max_share_diff_pp_post:.2f}pp")
    print()
    print("  Interpretation (v0.4.1 view, the relevant one):")
    if max_share_diff_pp_post < 1.0:
        print("    Sub-percentage differences. Either choice is fine; A is one")
        print("    less code path.")
    elif max_share_diff_pp_post < 5.0:
        print("    Small but non-negligible (<5pp) differences. B is")
        print("    statistically principled; A is simpler. User decision.")
    else:
        print(f"    Large divergence (max {max_share_diff_pp_post:.1f}pp). B's per-probe")
        print("    statistical weighting matters here; recommend B.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
