"""v0.4.5 Run 3 — the deprecated path as a mechanical reference. Trimmed.

The full ``run_family_experiment`` form this script previously ran is
gone, deliberately. It would have re-run the whole geometry phase
(~1 h) and then scored the two generative tasks, and neither generative
comparison is mechanical: ``gsm8k`` generates at
``TASK_MAX_NEW_TOKENS["gsm8k"]`` on the legacy path against the δ
budget on the new one (different by design), and ``longbench`` through
the validity-unaware legacy engine pushes 9k-token prompts into
4k-window models. Box time spent on numbers that cannot agree.

What *is* mechanical is the three multiple-choice tasks. Both paths
score them by log-likelihood over the same stored choices from the same
``from_lm_eval`` probes; the scoring function is shared (it was ported
in place), so the comparison isolates exactly the layer the port
changed — ``InferenceEngine``'s score path against ``HFEngine``'s.
Agreement to 1e-6 means the restored accuracy equals what the
deprecated path produced; disagreement means the port changed a
measurement rather than relocating one.

    hellaswag                     -> commonsense
    arc_challenge                 -> reasoning
    mmlu_college_computer_science -> code

The base engine is scored too. The deprecated path never did — which is
why ``base_accuracy`` never existed — but nothing stops its scorer
being *run* on the base config, and that gives the new base column a
reference as well: 15 cells, all mechanical.

    mamba run -n lmdiff python scripts/_v045_legacy_accuracy.py \\
        --payload v045/run2_payload.json

Exit 0 iff every cell agrees within 1e-6. v0.5.0 deletes
``InferenceEngine``, so this is the last release in which this
comparison can be run at all.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASK_TO_DOMAIN = {
    "hellaswag": "commonsense",
    "arc_challenge": "reasoning",
    "mmlu_college_computer_science": "code",
}
TOLERANCE = 1e-6


def _new_path_cells(payload: dict) -> dict[str, dict[str, float]]:
    """{engine_label: {domain: accuracy}} from the Run 2 payload."""
    meta = payload.get("metadata") or {}
    out: dict[str, dict[str, float]] = {}
    base = meta.get("base_accuracy") or {}
    if base:
        out["__base__"] = {d: base.get(d) for d in TASK_TO_DOMAIN.values()}
    for v, row in (meta.get("accuracy_by_variant") or {}).items():
        out[v] = {d: row.get(d) for d in TASK_TO_DOMAIN.values()}
    return out


def compare(new_cells: dict, legacy_cells: dict) -> int:
    """Print the side-by-side and return the count of cells outside
    tolerance. Pure function of two dicts — CPU-tested."""
    print("\n=== Legacy (InferenceEngine) vs new path (HFEngine), 1e-6 ===")
    print(f"  {'engine':<12}{'task':<32}{'legacy':>10}{'new':>10}{'|Δ|':>12}")
    failures = 0
    for label in legacy_cells:
        for task, domain in TASK_TO_DOMAIN.items():
            lv = legacy_cells[label].get(task)
            nv = (new_cells.get(label) or {}).get(domain)
            if lv is None or nv is None:
                print(f"  {label:<12}{task:<32}{'?':>10}{'?':>10}"
                      f"{'MISSING':>12}  ***")
                failures += 1
                continue
            delta = abs(lv - nv)
            ok = delta <= TOLERANCE
            failures += not ok
            print(f"  {label:<12}{task:<32}{lv:>10.6f}{nv:>10.6f}"
                  f"{delta:>12.3e}{'' if ok else '  *** MOVED ***'}")
    verdict = "PASS" if failures == 0 else f"{failures} CELL(S) DISAGREE"
    print(f"\nRESULT: {verdict}")
    if failures:
        print("A disagreement here means the port changed a measurement "
              "rather than relocating one. Release blocker.")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", type=pathlib.Path, required=True,
                    help="run2_payload.json from _v045_accuracy_report.py")
    args = ap.parse_args()

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    new_cells = _new_path_cells(payload)
    if "__base__" not in new_cells:
        print("payload carries no base_accuracy — nothing to compare "
              "the base column against; run 2 should have failed already.")
        return 2

    from tests.integration._v041_4variant_spec import build_run_kwargs

    kw = build_run_kwargs()
    engines_spec: dict[str, str] = {"__base__": kw["base"]}
    for name, v in kw["variants"].items():
        engines_spec[name] = v if isinstance(v, str) else v.model

    n_probes = kw.get("n_probes", 100)
    print("Scoring the three multiple-choice tasks through the DEPRECATED "
          "engine.")
    print(f"  engines : {list(engines_spec)}")
    print(f"  tasks   : {list(TASK_TO_DOMAIN)}   n_probes={n_probes} per task")
    print("  (DeprecationWarnings from lmdiff.config / lmdiff.engine are "
          "expected here — this script exists to drive the deprecated path.)")

    # Probes once, engines many: same builder and args the new path's
    # _resolve_probes used, so both paths score identical Probe objects.
    from lmdiff.probes.adapters import from_lm_eval

    probesets = {t: from_lm_eval(t, limit=n_probes) for t in TASK_TO_DOMAIN}
    for t, ps in probesets.items():
        print(f"  {t}: {len(ps)} probes")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from lmdiff.config import Config as LegacyConfig
        from lmdiff.engine import InferenceEngine, release_cuda_cache
        from lmdiff.experiments.family import _accuracy_for_task

        legacy_cells: dict[str, dict[str, float]] = {}
        for label, model_id in engines_spec.items():
            print(f"\nloading {label} ({model_id}) ...")
            engine = InferenceEngine(LegacyConfig(model=model_id))
            try:
                legacy_cells[label] = {}
                for task in TASK_TO_DOMAIN:
                    acc = _accuracy_for_task(task, probesets[task], engine)
                    legacy_cells[label][task] = acc
                    print(f"  {task}: {acc:.6f}")
            finally:
                del engine
                import gc

                gc.collect()
                release_cuda_cache()

    return 1 if compare(new_cells, legacy_cells) else 0


if __name__ == "__main__":
    raise SystemExit(main())
