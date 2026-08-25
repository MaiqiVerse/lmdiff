"""v0.4.5 GPU verification — geometry unchanged, accuracy plausible.

One ``family()`` call over the canonical Llama-2 4-variant spec. Prints:

  1. field-by-field comparison against the committed v0.4.1 baseline
     (the release gate — every existing quantity must be unmoved), and
  2. the accuracy tables, which have no reference and are judged by eye.

    mamba run -n lmdiff python scripts/_v045_accuracy_report.py
    mamba run -n lmdiff python scripts/_v045_accuracy_report.py --skip-geometry

Pass criteria and what to look for: docs/internal/v045_engine_port_notes.md.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASELINE = ROOT / "tests" / "fixtures" / "calibration_v041_4variant_baseline.json"
TOLERANCE = 1e-6


def _cmp_geometry(payload: dict, baseline: dict) -> int:
    """Return the number of fields that moved. Zero is the gate."""
    failures = 0

    def _report(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"  {'OK  ' if ok else 'MOVED'}  {name}{'  ' + detail if detail else ''}")
        if not ok:
            failures += 1

    for field in ("magnitudes", "magnitudes_normalized"):
        b, c = baseline.get(field) or {}, payload.get(field) or {}
        worst = max(
            (abs(b[k] - c.get(k, float("nan"))) for k in b), default=0.0,
        )
        _report(field, worst < TOLERANCE, f"max|Δ|={worst:.3e}")

    b, c = baseline.get("change_vectors") or {}, payload.get("change_vectors") or {}
    worst = 0.0
    for v, vec in b.items():
        for i, x in enumerate(vec):
            worst = max(worst, abs(x - c.get(v, [])[i]))
    _report("change_vectors", worst < TOLERANCE, f"max|Δ|={worst:.3e}")

    for field in ("cosine_matrix", "selective_cosine_matrix"):
        b, c = baseline.get(field) or {}, payload.get(field) or {}
        worst = 0.0
        for v, row in b.items():
            for w, x in row.items():
                y = (c.get(v) or {}).get(w)
                if x is None or y is None:
                    continue
                worst = max(worst, abs(x - y))
        _report(field, worst < TOLERANCE, f"max|Δ|={worst:.3e}")

    for field in ("share_per_domain", "magnitudes_per_domain_normalized"):
        b, c = baseline.get(field) or {}, payload.get(field) or {}
        worst, none_mismatch = 0.0, 0
        for v, row in b.items():
            for d, x in row.items():
                y = (c.get(v) or {}).get(d)
                if (x is None) != (y is None):
                    none_mismatch += 1
                elif x is not None:
                    worst = max(worst, abs(x - y))
        _report(
            field, worst < TOLERANCE and none_mismatch == 0,
            f"max|Δ|={worst:.3e} none-mismatches={none_mismatch}",
        )

    _report(
        "domain_status",
        (baseline.get("domain_status") or {}) == (payload.get("domain_status") or {}),
    )
    return failures


def _acc_table(title: str, base_acc: dict, by_variant: dict) -> None:
    print(f"\n=== {title} ===")
    variants = list(by_variant)
    domains = sorted({d for row in by_variant.values() for d in row} | set(base_acc))
    if not domains:
        print("  (empty — accuracy was not computed)")
        return
    head = f"  {'domain':<16}{'base':>8}" + "".join(f"{v:>9}" for v in variants)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for d in domains:
        cells = [base_acc.get(d)] + [by_variant[v].get(d) for v in variants]
        row = f"  {d:<16}" + "".join(
            f"{'—':>9}" if c is None else f"{c:>9.3f}" for c in cells
        )
        print(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-geometry", action="store_true")
    ap.add_argument("--save", type=pathlib.Path, default=None)
    args = ap.parse_args()

    import lmdiff
    from lmdiff.report.json_report import to_json_dict
    from tests.integration._v041_4variant_spec import build_run_kwargs

    print(f"lmdiff {lmdiff.__version__} — running the 4-variant spec ...")
    result = lmdiff.family(**build_run_kwargs())
    payload = to_json_dict(result)
    payload.pop("generated_at", None)

    if args.save:
        args.save.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved payload -> {args.save}")

    failures = 0
    if not args.skip_geometry:
        print("\n=== Geometry vs the committed v0.4.1 baseline (gate) ===")
        if not BASELINE.exists():
            print(f"  BASELINE MISSING at {BASELINE} — this run verified nothing.")
            return 2
        with BASELINE.open(encoding="utf-8") as f:
            failures = _cmp_geometry(payload, json.load(f))

    meta = result.metadata or {}
    _acc_table(
        "Accuracy by domain (no reference — judge by eye)",
        meta.get("base_accuracy") or {},
        meta.get("accuracy_by_variant") or {},
    )

    print("\n=== Findings that should or should not be present ===")
    from lmdiff._findings import (
        AccuracyArtifactFinding,
        BaseAccuracyMissingFinding,
        extract_findings,
    )

    findings = extract_findings(result)
    base_missing = [f for f in findings if isinstance(f, BaseAccuracyMissingFinding)]
    artifacts = [f for f in findings if isinstance(f, AccuracyArtifactFinding)]
    print(f"  BaseAccuracyMissingFinding : {len(base_missing)}  (expected 0)")
    for f in artifacts:
        print(f"  AccuracyArtifactFinding    : {f.details.get('task')}  "
              f"max_new_tokens={f.details.get('max_new_tokens')}")
    if base_missing:
        failures += 1

    print(f"\nRESULT: {'PASS' if failures == 0 else f'{failures} FIELD(S) MOVED'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
