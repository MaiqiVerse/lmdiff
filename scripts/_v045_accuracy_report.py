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


def _findings_of(result) -> list[tuple[str, str]]:
    return sorted((type(f).__name__, f.summary) for f in result.findings)


def _findings_diff(result) -> int:
    """Cutover findings against the baseline, element by element.

    `test_findings_match` failed in Run 1 with its full diff truncated,
    so the two questions that left open get answered here rather than by
    hand:

      1. **Are the shared findings byte-identical?** A finding summary
         formats a derived number, so a summary that differs means a
         number moved and the 1e-6 field tolerance is coarser than the
         formatting. That would be a real regression, and none of the 50
         numeric assertions would have caught it.
      2. **Is `AccuracyArtifactFinding` the only addition?** A
         `BaseAccuracyMissingFinding` appearing means base scoring
         failed — a different problem, and the one v0.4.5 exists to
         prevent.

    An `AccuracyArtifactFinding` addition on its own is **not** counted
    as a failure: the baseline fixture is schema v6 and carries no
    accuracy at all, so that finding cannot exist on its side.
    """
    import lmdiff

    cut = _findings_of(result)
    base = _findings_of(lmdiff.load_result(str(BASELINE))) if BASELINE.exists() else []

    print("\n=== Cutover findings (full list) ===")
    for t, sm in cut:
        print(f"  {t:<28}{sm}")

    if not base:
        print("\n  (no baseline to diff against)")
        return 0

    print("\n=== Baseline findings (full list) ===")
    for t, sm in base:
        print(f"  {t:<28}{sm}")

    added = sorted(set(cut) - set(base))
    removed = sorted(set(base) - set(cut))

    # A moved number shows up as one added + one removed sharing a type.
    by_type: dict[str, tuple[list[str], list[str]]] = {}
    for t, sm in base:
        by_type.setdefault(t, ([], []))[0].append(sm)
    for t, sm in cut:
        by_type.setdefault(t, ([], []))[1].append(sm)
    changed = [
        (t, b, c) for t, (b, c) in sorted(by_type.items())
        if b and c and b != c
    ]

    print("\n=== Diff ===")
    failures = 0

    if changed:
        print("  *** SUMMARY CHANGED — a number moved ***")
        for t, b, c in changed:
            print(f"    {t}")
            for x in b:
                print(f"      baseline: {x}")
            for x in c:
                print(f"      cutover : {x}")
        failures += len(changed)
    else:
        print("  shared findings: IDENTICAL — no summary differs")

    # A changed summary surfaces as one added + one removed sharing a
    # type. Those are two halves of one difference, already counted
    # above — listing them again would double-count and, worse, report a
    # moved number as a spurious "unexpected new finding".
    changed_types = {c[0] for c in changed}
    added = [(t, sm) for t, sm in added if t not in changed_types]
    removed = [(t, sm) for t, sm in removed if t not in changed_types]

    expected_additions = {"AccuracyArtifactFinding"}
    added_types = {t for t, _ in added}
    if added:
        for t, sm in added:
            flag = "expected" if t in expected_additions else "*** UNEXPECTED ***"
            print(f"  + {t:<28}{sm}   [{flag}]")
    else:
        print("  no additions")
    failures += len(added_types - expected_additions)

    for t, sm in removed:
        print(f"  - {t:<28}{sm}   *** REMOVED ***")
        failures += 1

    if "BaseAccuracyMissingFinding" in added_types:
        print("\n  *** BaseAccuracyMissingFinding present: base scoring FAILED.")
        print("      This is the caveat that would otherwise fire on every")
        print("      report forever; scoring the base engine is what v0.4.5")
        print("      added to prevent it. Check the `base` column above —")
        print("      it will be empty or all-None.")

    return failures


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

    failures += _findings_diff(result)

    print(f"\nRESULT: {'PASS' if failures == 0 else f'{failures} FIELD(S) MOVED'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
