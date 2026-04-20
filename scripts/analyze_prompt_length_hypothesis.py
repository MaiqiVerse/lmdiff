"""Test hypothesis: long and sysprompt have similar selective δ pattern
because both are sensitive to prompt token length.

Reads examples/family_geometry_extended.json + lmdiff/probes/v01.json.
For each variant, computes Pearson r between:
  - selective δ per probe (= δ − mean(δ))
  - probe token count (measured with Llama-2 tokenizer)

Alignment note: the extended JSON's per_probe dicts are serialized with
json.dumps(sort_keys=True), so per_probe key order is alphabetical, not
prompts order. change_vectors is a list and stays in prompts order, which
matches v01.json probe order when n_skipped == 0. So we source probe
texts/domains from v01.json directly and read δ from change_vectors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).parent.parent
JSON_PATH = ROOT / "examples" / "family_geometry_extended.json"
V01_PATH = ROOT / "lmdiff" / "probes" / "v01.json"
TOKENIZER_ID = "meta-llama/Llama-2-7b-hf"


def main():
    # Load δ data
    with open(JSON_PATH) as f:
        geom = json.load(f)
    variant_names = list(geom["variant_names"])
    change_vectors = geom["change_vectors"]
    n_skipped = geom["metadata"]["n_skipped"]
    n_total = geom["metadata"]["n_total_probes"]

    assert n_skipped == 0, (
        f"n_skipped={n_skipped}; change_vectors may not align 1:1 with v01 probe "
        "order. This script's alignment assumption requires no probes were "
        "filtered out. Regenerate the JSON with a probe set that has no NaN CEs, "
        "or switch to key-based alignment."
    )

    # Load probes from v01 (original insertion order = change_vectors order)
    with open(V01_PATH) as f:
        v01 = json.load(f)
    probe_entries = v01["probes"]
    probe_texts = [p["text"] for p in probe_entries]
    probe_domains = [p.get("domain", "unknown") for p in probe_entries]
    n_probes = len(probe_texts)

    assert n_probes == n_total, (
        f"v01 has {n_probes} probes but metadata.n_total_probes={n_total}"
    )
    for name in variant_names:
        assert len(change_vectors[name]) == n_probes, (
            f"change_vectors[{name!r}] has {len(change_vectors[name])} entries, "
            f"expected {n_probes}"
        )

    print(f"Loaded {n_probes} probes, {len(variant_names)} variants")
    print(f"n_skipped={n_skipped} (alignment precondition satisfied)")

    # Tokenize probes
    print(f"Loading tokenizer {TOKENIZER_ID} ...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    # Strip BOS if present, count content tokens only (fair across probes)
    token_counts = np.asarray(
        [len(tok.encode(t, add_special_tokens=False)) for t in probe_texts],
        dtype=float,
    )

    print(f"Probe token count stats:")
    print(f"  min={int(token_counts.min())}  max={int(token_counts.max())}  "
          f"mean={token_counts.mean():.2f}  std={token_counts.std():.2f}")

    # For each variant, compute selective δ and correlate with token_counts
    print("\n" + "=" * 68)
    print("Pearson r between selective δ (= δ − mean(δ)) and probe token count")
    print("=" * 68)
    print(f"{'variant':<12} {'mean(δ)':>10} {'selective_norm':>14} "
          f"{'r_vs_length':>12} {'n':>4}")
    print("-" * 56)

    selective_by_variant = {}
    for name in variant_names:
        vec = np.asarray(change_vectors[name], dtype=float)
        c = float(vec.mean())
        selective = vec - c
        sel_norm = float(np.linalg.norm(selective))
        selective_by_variant[name] = selective

        if sel_norm == 0:
            r = float("nan")
        else:
            tc_centered = token_counts - token_counts.mean()
            tc_norm = float(np.linalg.norm(tc_centered))
            if tc_norm == 0:
                r = float("nan")
            else:
                r = float(np.dot(selective, tc_centered) / (sel_norm * tc_norm))
                r = max(-1.0, min(1.0, r))

        print(f"{name:<12} {c:>+10.4f} {sel_norm:>14.4f} {r:>+12.4f} {vec.size:>4d}")

    # Per-domain breakdown
    print("\n" + "=" * 68)
    print("Length correlation broken out by domain")
    print("=" * 68)

    unique_domains = sorted(set(probe_domains))
    print(f"{'variant':<12} " + " ".join(f"{d:>12}" for d in unique_domains))
    print("-" * (13 + 13 * len(unique_domains)))

    for name in variant_names:
        selective = selective_by_variant[name]
        row = [f"{name:<12}"]
        for dom in unique_domains:
            idx = [i for i, d in enumerate(probe_domains) if d == dom]
            if len(idx) < 2:
                row.append(f"{'n/a':>12}")
                continue
            sel_sub = selective[idx]
            tc_sub = token_counts[idx]
            sel_sub_c = sel_sub - sel_sub.mean()
            tc_sub_c = tc_sub - tc_sub.mean()
            sn = float(np.linalg.norm(sel_sub_c))
            tn = float(np.linalg.norm(tc_sub_c))
            if sn == 0 or tn == 0:
                r = float("nan")
            else:
                r = float(np.dot(sel_sub_c, tc_sub_c) / (sn * tn))
                r = max(-1.0, min(1.0, r))
            row.append(f"{r:>+12.4f}")
        print(" ".join(row))

    # Sanity: token count distribution
    unique_lengths = np.unique(token_counts)
    print(f"\nToken count support: {len(unique_lengths)} unique values. "
          f"First 10: {unique_lengths[:10].astype(int).tolist()}")


if __name__ == "__main__":
    main()
