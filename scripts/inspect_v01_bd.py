"""Diagnostic script: inspect v01 BD internals for gpt2 vs distilgpt2."""

from __future__ import annotations

import math
from pathlib import Path

from modeldiff.config import Config
from modeldiff.engine import InferenceEngine
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

    eos_tok_a = engine_a.tokenizer.eos_token
    eos_tok_b = engine_b.tokenizer.eos_token
    eos_id_a = engine_a.tokenizer.eos_token_id
    eos_id_b = engine_b.tokenizer.eos_token_id

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

    # ── Per-domain detailed diagnostics ──────────────────────────────────

    by_domain = ps.by_domain()
    for domain in sorted(by_domain):
        domain_ps = by_domain[domain]
        print(f"\n{'='*70}")
        print(f"  DOMAIN: {domain} ({len(domain_ps)} probes)")
        print(f"{'='*70}")

        shown = 0
        for probe in domain_ps:
            if shown >= 3:
                break
            idx = ps.ids.index(probe.id)
            shown += 1

            out_a = outputs_a[idx]
            out_b = outputs_b[idx]
            tid_a = ids_a[idx]
            tid_b = ids_b[idx]

            ce_aa = score_aa.cross_entropies[idx]
            ce_ab = score_ab.cross_entropies[idx]
            ce_ba = score_ba.cross_entropies[idx]
            ce_bb = score_bb.cross_entropies[idx]

            n_tok_a_self = len(score_aa.token_ids[idx])
            n_tok_a_cross = len(score_ab.token_ids[idx])
            n_tok_b_self = len(score_bb.token_ids[idx])
            n_tok_b_cross = len(score_ba.token_ids[idx])

            has_eos_a = eos_id_a in tid_a
            has_eos_b = eos_id_b in tid_b

            print(f"\n  [{probe.id}] {probe.text!r}")
            print(f"    output_a (gpt2):      {out_a!r}")
            print(f"    output_b (distilgpt2): {out_b!r}")
            print(f"    tokens_a: {len(tid_a)} gen | {n_tok_a_self} score_aa | {n_tok_a_cross} score_ab")
            print(f"    tokens_b: {len(tid_b)} gen | {n_tok_b_self} score_bb | {n_tok_b_cross} score_ba")
            if has_eos_a:
                print(f"    ⚠ output_a contains EOS ({eos_tok_a!r})")
            if has_eos_b:
                print(f"    ⚠ output_b contains EOS ({eos_tok_b!r})")
            print(f"    CE: aa={ce_aa:.4f}  ab={ce_ab:.4f}  ba={ce_ba:.4f}  bb={ce_bb:.4f}")

            if not math.isnan(ce_aa) and not math.isnan(ce_bb):
                bd_i = 0.5 * (ce_ab - ce_bb) + 0.5 * (ce_ba - ce_aa)
                print(f"    BD={bd_i:.4f}")

    # ── Aggregate statistics ─────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("  AGGREGATE STATISTICS")
    print(f"{'='*70}")

    for domain in sorted(by_domain):
        domain_ps = by_domain[domain]
        indices = [ps.ids.index(p.id) for p in domain_ps]
        lens_a = [len(ids_a[i]) for i in indices]
        lens_b = [len(ids_b[i]) for i in indices]
        print(f"\n  {domain}:")
        print(f"    avg output length (gpt2):      {sum(lens_a)/len(lens_a):.1f} tokens")
        print(f"    avg output length (distilgpt2): {sum(lens_b)/len(lens_b):.1f} tokens")

    # Empty / whitespace outputs
    empty_a = sum(1 for o in outputs_a if not o.strip())
    empty_b = sum(1 for o in outputs_b if not o.strip())
    print(f"\n  Empty/whitespace outputs: gpt2={empty_a}, distilgpt2={empty_b}")

    # Outliers: any CE > 5 nats
    print(f"\n  Outlier probes (any CE > 5 nats):")
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
            print(
                f"    {probe.id:16s}  "
                f"aa={ce_vals[0]:.2f}  ab={ce_vals[1]:.2f}  "
                f"ba={ce_vals[2]:.2f}  bb={ce_vals[3]:.2f}"
            )
    if not found_outlier:
        print("    (none)")

    print()


if __name__ == "__main__":
    main()
