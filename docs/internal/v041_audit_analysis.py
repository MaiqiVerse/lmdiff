"""Numerical impact analysis — v0.4.1 audit (one-off, read-only).

Compares three formulations of per-domain magnitude (pdn) on real
data from the v0.3.2 demo, to inform the v0.4.1 design decisions:

  Formula B (current): pdn_B[v][d] = sqrt(Σ_{i∈d} δ_i² / Σ_{i∈d} T_i)
  Formula B' (drop LC): same as B but long-context probes excluded
                        (mimics base-context-window validity filter)
  Formula C (proposed): pdn_C[v][d] = sqrt(mean_{i∈d}(δ_i²))
                        on *valid* probes only

Reads:
  _demo_check/runs/v032-rerendered/family_geometry.json (7-variant)

Writes nothing; prints tabular comparison to stdout.

Run:
  mamba run -n lmdiff python docs/internal/v041_audit_analysis.py
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "_demo_check" / "runs" / "v032-rerendered" / "family_geometry.json"

# Llama-2-7b-hf context = 4096 (max_position_embeddings). Anything longer is
# out-of-range for the base model. The 7-variant demo's `longbench_2wikimqa`
# task averages ~9000 tokens; under v0.4.1 those probes should be dropped
# from base-vs-variant comparisons.
BASE_CONTEXT = 4096


def load_demo() -> dict:
    return json.loads(DEMO.read_text(encoding="utf-8"))


def group_by_domain(probe_domains: list[str]) -> dict[str, list[int]]:
    by_d: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(probe_domains):
        by_d[d if d is not None else "unknown"].append(i)
    return by_d


def formula_b(deltas: list[float], tokens: list[float], idxs: list[int]) -> float:
    """Current (geometry.py:1073): sqrt(Σδ²/ΣT)."""
    s = sum(deltas[i] * deltas[i] for i in idxs)
    t = sum(tokens[i] for i in idxs)
    return math.sqrt(s / t) if t > 0 else 0.0


def formula_c(deltas: list[float], idxs: list[int]) -> float:
    """Proposed (Y.4 component 4): sqrt(mean(δ²))."""
    if not idxs:
        return 0.0
    s = sum(deltas[i] * deltas[i] for i in idxs)
    return math.sqrt(s / len(idxs))


def share_from(pdn_row: dict[str, float]) -> dict[str, float | None]:
    """Squared-share normalization. None when total is zero."""
    sq = {d: m * m for d, m in pdn_row.items() if m is not None}
    total = sum(sq.values())
    if total <= 0:
        return {d: None for d in pdn_row}
    return {d: q / total for d, q in sq.items()}


def main() -> int:
    d = load_demo()
    deltas_by_v: dict[str, list[float]] = d["change_vectors"]
    domains = list(d["probe_domains"])
    tokens = list(d["avg_tokens_per_probe"])

    n = len(domains)
    assert all(len(v) == n for v in deltas_by_v.values())
    assert len(tokens) == n

    by_d = group_by_domain(domains)
    variants = list(deltas_by_v.keys())

    # Per-domain token stats — confirms long-context >> 4096
    print("=" * 78)
    print(f"DEMO: {DEMO.relative_to(REPO)}")
    print("=" * 78)
    print(f"n_probes={n}  variants={variants}")
    print()
    print("Per-domain token stats (T_i means probe prompt tokens, not continuation):")
    print(f"  {'domain':<14} {'n':>4} {'T_min':>6} {'T_max':>6} {'T_mean':>7} {'>4K?':>8}")
    for dom in sorted(by_d):
        idxs = by_d[dom]
        ts = [tokens[i] for i in idxs]
        over = sum(1 for t in ts if t > BASE_CONTEXT)
        print(f"  {dom:<14} {len(idxs):>4} {min(ts):>6.0f} {max(ts):>6.0f} "
              f"{sum(ts)/len(ts):>7.0f} {over:>8d}")
    print()

    # ── Three formulas, each variant ──
    pdn_b: dict[str, dict[str, float]] = {}
    pdn_b_drop: dict[str, dict[str, float | None]] = {}
    pdn_c: dict[str, dict[str, float | None]] = {}

    for v in variants:
        dv = deltas_by_v[v]
        b_row: dict[str, float] = {}
        b_drop_row: dict[str, float | None] = {}
        c_row: dict[str, float | None] = {}
        for dom, idxs in by_d.items():
            valid = [i for i in idxs if tokens[i] <= BASE_CONTEXT]
            b_row[dom] = formula_b(dv, tokens, idxs)
            if valid:
                b_drop_row[dom] = formula_b(dv, tokens, valid)
                c_row[dom] = formula_c(dv, valid)
            else:
                b_drop_row[dom] = None  # all probes out-of-range
                c_row[dom] = None
        pdn_b[v] = b_row
        pdn_b_drop[v] = b_drop_row
        pdn_c[v] = c_row

    # ── Side-by-side share table ──
    print()
    print("─" * 78)
    print(" SHARE PER DOMAIN — three formulations side by side")
    print(f"  B      = current sqrt(Σδ²/ΣT) on ALL probes")
    print(f"  B'     = current formula on valid probes only (LC dropped, T>4K)")
    print(f"  C      = sqrt(mean(δ²)) on valid probes only (proposed Y.4)")
    print("─" * 78)

    domains_sorted = sorted(by_d.keys())
    for v in variants:
        sb = share_from({d: pdn_b[v][d] for d in domains_sorted})
        sb_drop = share_from({d: pdn_b_drop[v][d] for d in domains_sorted
                              if pdn_b_drop[v][d] is not None})
        sc = share_from({d: pdn_c[v][d] for d in domains_sorted
                         if pdn_c[v][d] is not None})

        print()
        print(f"▶ {v}")
        print(f"    {'domain':<14} {'B share':>9} {'B′ share':>10} {'C share':>9}"
              f"  {'pdn_B':>8} {'pdn_B′':>8} {'pdn_C':>8}")
        for dom in domains_sorted:
            shb = sb.get(dom)
            shbd = sb_drop.get(dom)
            shc = sc.get(dom)
            pb = pdn_b[v][dom]
            pbd = pdn_b_drop[v][dom]
            pc = pdn_c[v][dom]

            def fmt_share(x):
                return "  —    " if x is None else f"{x*100:5.1f}%  "

            def fmt_pdn(x):
                return "  —    " if x is None else f"{x:7.3f} "

            print(f"    {dom:<14} {fmt_share(shb):>9} {fmt_share(shbd):>10} "
                  f"{fmt_share(shc):>9}  {fmt_pdn(pb):>8} {fmt_pdn(pbd):>8} "
                  f"{fmt_pdn(pc):>8}")

        # biggest move
        b_max = max(domains_sorted, key=lambda x: sb.get(x) or 0)
        bd_dom = [k for k in sb_drop if sb_drop[k] is not None]
        c_dom = [k for k in sc if sc[k] is not None]
        bd_max = max(bd_dom, key=lambda x: sb_drop[x]) if bd_dom else "—"
        c_max = max(c_dom, key=lambda x: sc[x]) if c_dom else "—"
        print(f"      biggest under B:  {b_max:<14}    "
              f"under B′:  {bd_max:<14}    under C:  {c_max}")

    # ── Q1 summary: long-context share under each formula ──
    print()
    print("─" * 78)
    print(" Q1: long-context share under each formula")
    print("─" * 78)
    print(f"  {'variant':<15} {'B':>7} {'B′':>7} {'C':>7}  (B′/C drop LC entirely → '—')")
    for v in variants:
        sb = share_from({d: pdn_b[v][d] for d in domains_sorted})
        # In B′ and C, long-context is dropped — so its share is None
        lc_b = (sb.get("long-context") or 0) * 100
        # No share for excluded domain in B'/C — they're shown as "—"
        print(f"  {v:<15} {lc_b:>6.1f}%   —       —")

    # ── Q2 summary: did the *story* survive C? ──
    print()
    print("─" * 78)
    print(" Q2: where is each variant's BIGGEST share under each formula?")
    print("─" * 78)
    print(f"  {'variant':<15} {'B':<14} {'B′ (drop LC)':<18} {'C (drop LC)':<14}")
    for v in variants:
        sb = share_from({d: pdn_b[v][d] for d in domains_sorted})
        sb_drop = share_from({d: pdn_b_drop[v][d] for d in domains_sorted
                              if pdn_b_drop[v][d] is not None})
        sc = share_from({d: pdn_c[v][d] for d in domains_sorted
                         if pdn_c[v][d] is not None})
        b_max = max(sb, key=lambda x: sb[x] or 0)
        bd_dom = [k for k in sb_drop if sb_drop[k] is not None]
        c_dom = [k for k in sc if sc[k] is not None]
        bd_max = max(bd_dom, key=lambda x: sb_drop[x]) if bd_dom else "—"
        c_max = max(c_dom, key=lambda x: sc[x]) if c_dom else "—"
        b_pct = (sb[b_max] or 0) * 100
        bd_pct = (sb_drop[bd_max] or 0) * 100 if bd_max != "—" else 0
        c_pct = (sc[c_max] or 0) * 100 if c_max != "—" else 0
        print(f"  {v:<15} {b_max:<10} {b_pct:>3.0f}%  "
              f"{bd_max:<10} {bd_pct:>3.0f}%       "
              f"{c_max:<10} {c_pct:>3.0f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
