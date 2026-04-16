"""Diagnostic script: inspect v01 BD internals for gpt2 vs distilgpt2."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from modeldiff.config import Config
from modeldiff.engine import InferenceEngine
from modeldiff.metrics.output._degeneracy import is_degenerate_tokens as is_degenerate
from modeldiff.probes.loader import ProbeSet

V01_PATH = Path(__file__).parent.parent / "modeldiff" / "probes" / "v01.json"
MAX_NEW_TOKENS = 16


def main() -> None:
    ps = ProbeSet.from_json(V01_PATH)
    probes = ps.texts

    print("Loading gpt2...")
    engine_a = InferenceEngine(Config(model="gpt2"))
    print("Loading distilgpt2...")
    engine_b = InferenceEngine(Config(model="distilgpt2"))

    print(f"\nGenerating with max_new_tokens={MAX_NEW_TOKENS}...")
    gen_a = engine_a.generate(probes, n_samples=1, max_new_tokens=MAX_NEW_TOKENS)
    gen_b = engine_b.generate(probes, n_samples=1, max_new_tokens=MAX_NEW_TOKENS)

    outputs_a = [c[0] for c in gen_a.completions]
    outputs_b = [c[0] for c in gen_b.completions]
    ids_a = [t[0] for t in gen_a.token_ids]
    ids_b = [t[0] for t in gen_b.token_ids]

    print("Scoring (4 score calls)...")
    score_aa = engine_a.score(probes, continuation_ids=ids_a)
    score_bb = engine_b.score(probes, continuation_ids=ids_b)
    score_ab = engine_b.score(probes, continuations=outputs_a)
    score_ba = engine_a.score(probes, continuations=outputs_b)

    # ── CE > 5 outlier probe details ─────────────────────────────────────

    print(f"\n{'='*70}")
    print("  OUTLIER PROBES (any CE > 5 nats)")
    print(f"{'='*70}")

    found_outlier = False
    for i, probe in enumerate(ps):
        ce_vals = (
            score_aa.cross_entropies[i],
            score_ab.cross_entropies[i],
            score_ba.cross_entropies[i],
            score_bb.cross_entropies[i],
        )
        if any(not math.isnan(c) and c > 5.0 for c in ce_vals):
            found_outlier = True
            print(f"\n  [{probe.id}] {probe.text!r}")
            print(f"    output_a (gpt2):       {outputs_a[i]!r}")
            print(f"    output_b (distilgpt2): {outputs_b[i]!r}")
            print(f"    token_ids_a: {ids_a[i]}")
            print(f"    token_ids_b: {ids_b[i]}")
            print(
                f"    CE: aa={ce_vals[0]:.4f}  ab={ce_vals[1]:.4f}  "
                f"ba={ce_vals[2]:.4f}  bb={ce_vals[3]:.4f}"
            )
            print(f"    degenerate_a: {is_degenerate(ids_a[i])}  degenerate_b: {is_degenerate(ids_b[i])}")
    if not found_outlier:
        print("  (none)")

    # ── Degeneracy analysis ──────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("  DEGENERACY ANALYSIS (≥80% repeated token)")
    print(f"{'='*70}")

    by_domain = ps.by_domain()

    for model_name, all_ids, all_outputs in [
        ("gpt2", ids_a, outputs_a),
        ("distilgpt2", ids_b, outputs_b),
    ]:
        print(f"\n  --- {model_name} ---")
        total_degen = 0
        total_nonascii = 0

        for domain in sorted(by_domain):
            domain_ps = by_domain[domain]
            indices = [ps.ids.index(p.id) for p in domain_ps]
            degen_count = sum(1 for i in indices if is_degenerate(all_ids[i]))
            nonascii_count = sum(
                1 for i in indices
                if all_outputs[i].startswith("\xa0")
                or "????" in all_outputs[i]
                or "Â" in all_outputs[i]
            )
            total_degen += degen_count
            total_nonascii += nonascii_count
            print(
                f"    {domain:12s}: "
                f"degenerate={degen_count}/10, "
                f"non-ASCII-repetition={nonascii_count}/10"
            )

            if degen_count > 0:
                for i in indices:
                    if is_degenerate(all_ids[i]):
                        probe = ps[i]
                        counts = Counter(all_ids[i])
                        top_tok, top_cnt = counts.most_common(1)[0]
                        print(
                            f"      {probe.id}: {all_outputs[i]!r:.60s}  "
                            f"(tok {top_tok} × {top_cnt}/{len(all_ids[i])})"
                        )

        print(f"    TOTAL: degenerate={total_degen}/30, non-ASCII={total_nonascii}/30")

    # ── Healthy-probe BD ─────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("  BD ON HEALTHY PROBES ONLY (both outputs non-degenerate)")
    print(f"{'='*70}")

    healthy_bd: dict[str, list[float]] = {}

    for i, probe in enumerate(ps):
        ce_aa = score_aa.cross_entropies[i]
        ce_ab = score_ab.cross_entropies[i]
        ce_ba = score_ba.cross_entropies[i]
        ce_bb = score_bb.cross_entropies[i]

        if any(math.isnan(c) for c in (ce_aa, ce_ab, ce_ba, ce_bb)):
            continue

        degen_a = is_degenerate(ids_a[i])
        degen_b = is_degenerate(ids_b[i])

        domain = probe.domain or "unknown"
        if degen_a or degen_b:
            continue

        bd_i = 0.5 * (ce_ab - ce_bb) + 0.5 * (ce_ba - ce_aa)
        healthy_bd.setdefault(domain, []).append(bd_i)

    all_healthy = []
    for domain in sorted(healthy_bd):
        vals = healthy_bd[domain]
        mean = sum(vals) / len(vals)
        all_healthy.extend(vals)
        print(f"  {domain:12s}: BD={mean:.4f} ({len(vals)} probes)")

    if all_healthy:
        overall = sum(all_healthy) / len(all_healthy)
        print(f"  {'overall':12s}: BD={overall:.4f} ({len(all_healthy)} probes)")
    else:
        print("  (no healthy probes)")

    print()


if __name__ == "__main__":
    main()
