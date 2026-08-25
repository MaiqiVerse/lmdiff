"""v0.4.5 GPU verification — the deprecated path's accuracy, for reference.

`run_family_experiment` is the only other thing in the tree that
measures accuracy, and **v0.5.0 deletes it**. So this is the last
release in which "does the ported path agree with the path it replaces"
can be asked mechanically at all.

The comparison is only exact for the three multiple-choice tasks, where
both paths score by log-likelihood over the same stored choices:

    hellaswag                       -> commonsense    exact, ~1e-6
    arc_challenge                   -> reasoning      exact, ~1e-6
    mmlu_college_computer_science   -> code           exact, ~1e-6
    gsm8k                           -> math           plausibility only
    longbench_2wikimqa              -> long-context   n/a (out of window)

`gsm8k` differs by design: the deprecated path generates fresh at
`TASK_MAX_NEW_TOKENS["gsm8k"]`, while v0.4.5 scores the δ generations,
which use the run's `max_new_tokens`.

    mamba run -n lmdiff python scripts/_v045_legacy_accuracy.py

Pair with `_v045_accuracy_report.py`; pass criteria are in
docs/internal/v045_engine_port_notes.md.
"""
from __future__ import annotations

import pathlib
import sys
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# task -> the domain the same probes land in on the new path.
TASK_TO_DOMAIN = {
    "hellaswag": "commonsense",
    "arc_challenge": "reasoning",
    "gsm8k": "math",
    "mmlu_college_computer_science": "code",
    "longbench_2wikimqa": "long-context",
}
EXACT = {"hellaswag", "arc_challenge", "mmlu_college_computer_science"}


def main() -> int:
    from tests.integration._v041_4variant_spec import build_run_kwargs

    kw = build_run_kwargs()
    probes = kw["probes"]
    tasks = probes.split(":", 1)[1].split("+") if isinstance(probes, str) else []
    variants = {
        n: (v if isinstance(v, str) else v.model) for n, v in kw["variants"].items()
    }

    print("Running the DEPRECATED path (run_family_experiment) for a reference.")
    print(f"  base     : {kw['base']}")
    print(f"  variants : {list(variants)}")
    print(f"  tasks    : {tasks}")
    print(f"  n_probes : {kw.get('n_probes')}   seed: {kw.get('seed')}\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from lmdiff.experiments.family import run_family_experiment

        result = run_family_experiment(
            base=kw["base"],
            variants=variants,
            task_names=tasks,
            limit_per_task=kw.get("n_probes", 100),
            seed=kw.get("seed", 42),
            max_new_tokens=kw.get("max_new_tokens", 16),
            skip_accuracy=False,
            write_outputs=False,
            progress=True,
        )

    acc = result.accuracy_by_variant
    print("\n=== Deprecated-path accuracy, keyed by lm-eval task ===")
    variant_names = list(acc)
    head = f"  {'task':<32}{'domain':<15}" + "".join(f"{v:>9}" for v in variant_names)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for task in tasks:
        domain = TASK_TO_DOMAIN.get(task, "?")
        mark = "  <- exact" if task in EXACT else ""
        cells = "".join(
            f"{acc[v].get(task, float('nan')):>9.3f}" for v in variant_names
        )
        print(f"  {task:<32}{domain:<15}{cells}{mark}")

    print("\nCompare the rows marked `exact` against the same domain column")
    print("in _v045_accuracy_report.py's table. Agreement to 1e-6 is the")
    print("pass criterion; a disagreement means the port changed a")
    print("measurement rather than relocating one.")
    print("\nThe deprecated path has no base column — it never scored the")
    print("base engine, which is why `base_accuracy` has never existed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
