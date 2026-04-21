#!/usr/bin/env python
"""Family geometry + accuracy radar using lm-eval probe sets.

Loads default 5 tasks (hellaswag, arc_challenge, gsm8k,
mmlu_college_computer_science, longbench_2wikimqa) via from_lm_eval,
runs ChangeGeometry once over the concatenated probe set, and runs
per-task accuracy (loglikelihood for MC, generate+evaluator for
generate_until). Emits JSON + delta-magnitude radar PNG + accuracy
radar PNG.

Requires: pip install lmdiff-kit[lm-eval,viz]

Usage:
    # Default: 5 tasks, 100 probes each. Specify variants:
    mamba run -n lmdiff python scripts/run_family_geometry_lm_eval.py \\
        --base meta-llama/Llama-2-7b-hf \\
        --variants yarn=NousResearch/Yarn-Llama-2-7b-128k,code=codellama/CodeLlama-7b-hf

    # Smaller smoke run for debugging:
    mamba run -n lmdiff python scripts/run_family_geometry_lm_eval.py \\
        --base gpt2 --variants v1=distilgpt2 \\
        --tasks hellaswag,arc_challenge --limit-per-task 5

Notes:
  - delta-magnitude radar uses raw |delta| in nats (or bpb if base and
    variant have different tokenizers).
  - Accuracy radar is 0-1 (acc for MC via loglikelihood, acc for
    generate via Gsm8kNumberMatch / F1 / ContainsAnswer).
  - HumanEval/MBPP excluded from default — use --tasks humaneval for
    delta-only run (accuracy radar will show 0 for those axes).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from lmdiff import (
    ChangeGeometry,
    Config,
    F1,
    Gsm8kNumberMatch,
    ProbeSet,
    Task,
    loglikelihood_accuracy,
)
from lmdiff.engine import release_cuda_cache
from lmdiff.probes.adapters import KNOWN_TASK_DOMAINS, from_lm_eval
from lmdiff.probes.loader import Probe
from lmdiff.report.terminal import print_geometry


DEFAULT_TASKS = [
    "hellaswag",
    "arc_challenge",
    "gsm8k",
    "mmlu_college_computer_science",
    "longbench_2wikimqa",
]

# Evaluator class for each generate_until task whose accuracy we score.
GENERATE_EVALUATORS = {
    "gsm8k": Gsm8kNumberMatch,
    "longbench_2wikimqa": F1,
    "longbench_hotpotqa": F1,
    "longbench_narrativeqa": F1,
    "longbench_qasper": F1,
    "squadv2": F1,
    "triviaqa": F1,
    "nq_open": F1,
}


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


def _load_concatenated_probes(
    task_names: list[str], limit_per_task: int, seed: int,
) -> tuple[ProbeSet, dict[str, list[Probe]]]:
    per_task: dict[str, list[Probe]] = {}
    all_probes: list[Probe] = []
    for task in task_names:
        ps = from_lm_eval(task, limit=limit_per_task, seed=seed)
        per_task[task] = list(ps)
        all_probes.extend(ps)
    mega = ProbeSet(
        all_probes,
        name=f"lm_eval:{'+'.join(task_names)}",
        version="lm-eval-harness",
    )
    return mega, per_task


def _accuracy_for_task(task_name: str, probes: ProbeSet, engine: Any) -> float:
    """Run the correct evaluator for this task on one engine. Returns accuracy in [0,1] or NaN."""
    info = KNOWN_TASK_DOMAINS.get(task_name)
    if info is None:
        from lmdiff.tasks.evaluators import ContainsAnswer
        task = Task(task_name, probes, ContainsAnswer(), max_new_tokens=32)
        return task.run(engine).accuracy

    if info.output_type == "multiple_choice":
        return loglikelihood_accuracy(probes, engine, task_name=task_name).accuracy

    if info.output_type == "generate_until":
        if info.requires_execution:
            # HumanEval / MBPP: pass@k would need sandbox. Skip accuracy.
            return float("nan")
        evaluator_cls = GENERATE_EVALUATORS.get(task_name)
        if evaluator_cls is None:
            from lmdiff.tasks.evaluators import ContainsAnswer
            evaluator_cls = ContainsAnswer
        task = Task(task_name, probes, evaluator_cls(), max_new_tokens=64)
        return task.run(engine).accuracy

    # loglikelihood / loglikelihood_rolling: not currently wired for accuracy.
    return float("nan")


def _task_probeset(
    per_task_probes: dict[str, list[Probe]], task: str, mega: ProbeSet,
) -> ProbeSet:
    return ProbeSet(
        per_task_probes[task], name=f"lm_eval:{task}", version=mega.version,
    )


def _partition_delta_by_task(
    per_probe: dict[str, dict[str, float]],
    per_task_probes: dict[str, list[Probe]],
) -> dict[str, dict[str, list[float]]]:
    """Split per_probe[variant] (keyed by probe text) into per-task lists."""
    task_by_text: dict[str, str] = {}
    for task, probes in per_task_probes.items():
        for p in probes:
            task_by_text[p.text] = task

    out: dict[str, dict[str, list[float]]] = {}
    for variant, probe_deltas in per_probe.items():
        out[variant] = {t: [] for t in per_task_probes}
        for text, delta in probe_deltas.items():
            task = task_by_text.get(text)
            if task is not None:
                out[variant][task].append(delta)
    return out


def _l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Family geometry + accuracy radar using lm-eval probe sets."
        ),
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

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== lm-eval family geometry experiment ===")
    print(f"Base: {args.base}")
    print(f"Variants ({len(variant_specs)}): {variant_specs}")
    print(f"Tasks ({len(task_names)}): {task_names}")
    print(f"Probes per task: {args.limit_per_task}")
    print(f"Max new tokens: {args.max_new_tokens}")

    t0 = time.time()
    mega, per_task_probes = _load_concatenated_probes(
        task_names, args.limit_per_task, args.seed,
    )
    t_load = time.time() - t0
    print(f"\nLoaded {len(mega)} probes across {len(task_names)} tasks in {t_load:.1f}s")
    for t in task_names:
        print(f"  {t}: {len(per_task_probes[t])} probes")

    # --- Phase A: delta-magnitude via ChangeGeometry ---
    print(f"\n=== ChangeGeometry ({len(variant_specs)} variants) ===")
    t0 = time.time()
    base_cfg = Config(model=args.base, dtype=args.dtype)
    variant_cfgs = {
        name: Config(model=mid, name=name, dtype=args.dtype)
        for name, mid in variant_specs.items()
    }
    cg = ChangeGeometry(base=base_cfg, variants=variant_cfgs, prompts=mega)
    geo = cg.analyze(max_new_tokens=args.max_new_tokens)
    t_geo = time.time() - t0
    print(f"ChangeGeometry done in {t_geo/60:.1f} min")
    print_geometry(geo)

    per_task_delta = _partition_delta_by_task(geo.per_probe, per_task_probes)
    delta_mag_by_variant: dict[str, dict[str, float]] = {}
    for variant, task_deltas in per_task_delta.items():
        delta_mag_by_variant[variant] = {
            t: _l2_norm(d) for t, d in task_deltas.items()
        }

    del cg
    gc.collect()
    release_cuda_cache()

    # --- Phase B: accuracy per task per variant ---
    accuracy_by_variant: dict[str, dict[str, float]] = {}
    if args.skip_accuracy:
        print("\nSkipping accuracy phase (--skip-accuracy).")
    else:
        print(f"\n=== Accuracy ({len(variant_specs)} variants x {len(task_names)} tasks) ===")
        t0 = time.time()
        from lmdiff.engine import InferenceEngine
        for vname, vcfg in variant_cfgs.items():
            print(f"  loading {vname} ...")
            engine = InferenceEngine(vcfg)
            try:
                accuracy_by_variant[vname] = {}
                for task in task_names:
                    ps_task = _task_probeset(per_task_probes, task, mega)
                    acc = _accuracy_for_task(task, ps_task, engine)
                    accuracy_by_variant[vname][task] = acc
                    if acc == acc:
                        print(f"    {task}: acc={acc:.3f}")
                    else:
                        print(f"    {task}: n/a (requires_execution or unsupported)")
            finally:
                del engine
                gc.collect()
                release_cuda_cache()
        t_acc = time.time() - t0
        print(f"Accuracy done in {t_acc/60:.1f} min")

    # --- JSON report ---
    report = {
        "base": args.base,
        "variants": variant_specs,
        "tasks": task_names,
        "limit_per_task": args.limit_per_task,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "delta_magnitude_by_variant": delta_mag_by_variant,
        "accuracy_by_variant": accuracy_by_variant,
        "geometry_metadata": geo.metadata,
        "magnitudes_total": dict(geo.magnitudes),
    }
    json_path = args.output_dir / f"{args.output_prefix}.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(f"\nJSON report: {json_path}")

    # --- Radars ---
    try:
        from lmdiff.viz.radar import plot_radar
    except ImportError as exc:
        print(f"[WARN] matplotlib not available; skipping radar plots: {exc}")
        return 0

    delta_png = args.output_dir / f"{args.output_prefix}_delta_radar.png"
    plot_radar(
        delta_mag_by_variant,
        axes=task_names,
        title=f"delta-magnitude vs {args.base}",
        out_path=delta_png,
    )
    print(f"delta-magnitude radar: {delta_png}")

    if accuracy_by_variant:
        # Replace NaN with 0 for radar rendering.
        acc_for_radar = {
            v: {t: (a if a == a else 0.0) for t, a in d.items()}
            for v, d in accuracy_by_variant.items()
        }
        acc_png = args.output_dir / f"{args.output_prefix}_accuracy_radar.png"
        plot_radar(
            acc_for_radar,
            axes=task_names,
            title="Accuracy per task",
            out_path=acc_png,
            value_range=(0.0, 1.0),
        )
        print(f"Accuracy radar: {acc_png}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
