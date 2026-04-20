"""Partial Llama-2 family geometry — 3 variants (no chat, no 13B/70B).

Produces the first real change-geometry cosine matrix on 7B-scale models.
Outputs both terminal rendering and JSON for README/paper.

Run:
    mamba run -n lmdiff python scripts/run_family_geometry_partial.py
"""
from __future__ import annotations

import time
from pathlib import Path

from lmdiff import ChangeGeometry, Config, ProbeSet
from lmdiff.report.json_report import write_json
from lmdiff.report.terminal import print_geometry


V01_PATH = Path(__file__).parent.parent / "lmdiff" / "probes" / "v01.json"
OUTPUT_DIR = Path(__file__).parent.parent / "examples"
OUTPUT_DIR.mkdir(exist_ok=True)

BASE = "meta-llama/Llama-2-7b-hf"
VARIANTS = {
    "yarn": "NousResearch/Yarn-Llama-2-7b-128k",
    "long": "togethercomputer/LLaMA-2-7B-32K",
    "code": "codellama/CodeLlama-7b-hf",
}
MAX_NEW_TOKENS = 16


def main() -> None:
    print(f"Loading probes from {V01_PATH}")
    probes = ProbeSet.from_json(V01_PATH)
    print(f"  {len(probes)} probes across domains: {probes.domains}")

    base_config = Config(model=BASE)
    variant_configs = {
        name: Config(model=model_id, name=name)
        for name, model_id in VARIANTS.items()
    }

    print(f"\nBase:     {BASE}")
    print("Variants:")
    for name, model_id in VARIANTS.items():
        print(f"  {name:6s} = {model_id}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}\n")

    cg = ChangeGeometry(
        base=base_config,
        variants=variant_configs,
        prompts=probes,
    )

    t0 = time.time()
    result = cg.analyze(max_new_tokens=MAX_NEW_TOKENS)
    elapsed = time.time() - t0
    print(f"\nAnalysis finished in {elapsed/60:.1f} min")

    # Terminal rendering
    print_geometry(result)

    # JSON export
    out_path = OUTPUT_DIR / "family_geometry_partial.json"
    write_json(result, out_path)
    print(f"\nJSON written to {out_path}")

    # Raw summary for quick scan
    print("\n=== Raw summary ===")
    print(f"n_probes={result.n_probes}, n_skipped={result.metadata.get('n_skipped', 0)}")
    print(f"bpb_normalized: {result.metadata.get('bpb_normalized', {})}")
    print("\nMagnitudes (sorted):")
    for name in sorted(result.magnitudes, key=result.magnitudes.get, reverse=True):
        print(f"  {name:6s} = {result.magnitudes[name]:.4f}")
    print("\nCosine matrix:")
    names = result.variant_names
    header = "        " + "  ".join(f"{n:>7s}" for n in names)
    print(header)
    for n1 in names:
        row = [f"{n1:6s}"]
        for n2 in names:
            v = result.cosine_matrix[n1][n2]
            row.append(f"{v:+.4f}" if v == v else "   n/a ")  # NaN check
        print("  ".join(row))


if __name__ == "__main__":
    main()
