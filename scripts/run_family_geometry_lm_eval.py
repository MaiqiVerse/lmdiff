#!/usr/bin/env python
"""Family geometry + accuracy radar using lm-eval probe sets (script wrapper).

Thin argparse wrapper around ``lmdiff.experiments.family.run_family_experiment``.
Preserved for backward compatibility with the original comma-separated
``--variants`` flag; new callers should prefer either:

  - the library API: ``from lmdiff.experiments import run_family_experiment``
  - the CLI:         ``lmdiff family-experiment --variant name=model_id ...``

Requires: pip install lmdiff-kit[lm-eval,viz]

Usage:
    mamba run -n lmdiff python scripts/run_family_geometry_lm_eval.py \\
        --base meta-llama/Llama-2-7b-hf \\
        --variants yarn=NousResearch/Yarn-Llama-2-7b-128k,code=codellama/CodeLlama-7b-hf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lmdiff.experiments.family import DEFAULT_TASKS, run_family_experiment


def _parse_variant_spec(specs: str) -> dict[str, str]:
    """Parse 'name1=path1,name2=path2' -> dict. Raises on malformed input."""
    out: dict[str, str] = {}
    for piece in specs.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"variant {piece!r} must be in 'name=model_id' format")
        name, mid = piece.split("=", 1)
        name, mid = name.strip(), mid.strip()
        if not name or not mid:
            raise ValueError(f"variant {piece!r} has empty name or model id")
        if name in out:
            raise ValueError(f"duplicate variant name: {name}")
        out[name] = mid
    if not out:
        raise ValueError("no variants specified")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Family geometry + accuracy radar using lm-eval probe sets.",
    )
    parser.add_argument("--base", required=True, help="HF id or path for base model")
    parser.add_argument(
        "--variants", required=True,
        help="Comma-separated name=model_id list "
             "(e.g., yarn=NousResearch/Yarn-Llama-2-7b-128k)",
    )
    parser.add_argument(
        "--tasks", default=",".join(DEFAULT_TASKS),
        help=f"Comma-separated lm-eval task names (default: {','.join(DEFAULT_TASKS)})",
    )
    parser.add_argument("--limit-per-task", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "examples",
    )
    parser.add_argument("--output-prefix", default="family_geometry_lm_eval")
    parser.add_argument(
        "--skip-accuracy", action="store_true",
        help="Only compute delta-magnitude radar (skip accuracy runs).",
    )
    parser.add_argument(
        "--dtype", default=None,
        choices=[None, "bfloat16", "float16", "float32"],
    )
    args = parser.parse_args(argv)

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    variant_specs = _parse_variant_spec(args.variants)

    run_family_experiment(
        base=args.base,
        variants=variant_specs,
        tasks=task_names,
        limit_per_task=args.limit_per_task,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        dtype=args.dtype,
        skip_accuracy=args.skip_accuracy,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        write_outputs=True,
        render_radars=True,
        progress=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
