"""Regenerate ``tests/fixtures/calibration_v040_7variant_summary.json``.

Runs the *exact* ``family()`` call that
``tests/integration/test_calibration_regression_7variant.py`` runs as
its ``cutover_result`` fixture, using the shared spec module
``tests/integration/_v040_7variant_spec.py``. Single source of truth:
neither the script nor the test inlines the kwargs.

When to re-run:
  - After any change to ``lmdiff._pipeline``, ``lmdiff._engine``, or
    other engine-touching code that the calibration test would catch.
  - After bumping the canonical seed, n_probes, or variant set in the
    shared spec.
  - When ``test_calibration_regression_7variant.py`` skips because
    the fixture file is missing on a fresh checkout.

Usage (GPU box):
    cd <repo-root>
    mamba run -n lmdiff python scripts/_regenerate_v040_7variant_fixture.py

Optional flags:
    --output PATH    Override the fixture path (default: spec value)
    --no-progress    Suppress per-probe progress bars
    --dry-run        Print the call that would run; exit without loading models

Runtime: ~1.5 hours on RTX 5090 (Blackwell, sm_120) for 7 variants ×
500 probes × 2 score phases. Watch the per-variant phase markers in
the log for sanity (each variant should take roughly equal time
except the loaded-once-then-released sequence).

After successful run:
    git add tests/fixtures/calibration_v040_7variant_summary.json
    git diff --stat tests/fixtures/        # confirm only the JSON changed
    git commit -m 'fixture: regenerate v0.4.0 7-variant calibration baseline'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``tests.integration._v040_7variant_spec`` importable when the
# script is run from the repo root (``python scripts/...``). The
# integration test runs under pytest which already adds the repo root
# to sys.path; this script doesn't, so do it explicitly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.integration._v040_7variant_spec import (  # noqa: E402
    FIXTURE_PATH,
    build_run_kwargs,
)


def _format_kwargs(kwargs: dict) -> str:
    """Pretty-print the family() call for logs / dry-run."""
    lines = ["lmdiff.family("]
    for k, v in kwargs.items():
        if k == "variants":
            lines.append("    variants={")
            for vname, vcfg in v.items():
                lines.append(f"        {vname!r}: {vcfg!r},")
            lines.append("    },")
        else:
            lines.append(f"    {k}={v!r},")
    lines.append(")")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=FIXTURE_PATH,
        help=f"Output JSON path (default: {FIXTURE_PATH})",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Suppress per-probe progress bars",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the family() call and exit without running",
    )
    args = parser.parse_args()

    kwargs = build_run_kwargs()

    print("=" * 72)
    print(f"v0.4.0 7-variant calibration fixture regeneration")
    print("=" * 72)
    print(f"Output:  {args.output}")
    print(f"Repo:    {_REPO_ROOT}")
    print()
    print(_format_kwargs(kwargs))
    print()

    if args.dry_run:
        print("[dry-run] would run the call above; exiting.")
        return 0

    if not args.no_progress:
        kwargs.setdefault("progress", True)

    print("[regen] loading lmdiff and running family() — expect ~1-1.5 h on a single GPU")
    print()

    import lmdiff  # noqa: F401  (keep import scoped: heavy)
    from lmdiff.report.json_report import to_json_dict

    result = lmdiff.family(**kwargs)
    payload = to_json_dict(result)

    # Strip volatile field — the test doesn't compare it, and including
    # it forces a re-commit every run.
    payload.pop("generated_at", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print(f"[regen] wrote {args.output}")
    print(f"[regen] n_probes:        {payload.get('n_probes')}")
    print(f"[regen] variant_names:   {payload.get('variant_names')}")
    from collections import Counter
    pd = payload.get("probe_domains", [])
    print(f"[regen] probe_domains:   {dict(Counter(pd))}")
    print()
    print("Next steps:")
    # Resolve to absolute first so a CWD-relative ``--output regenerate.json``
    # still works; fall back to the user-supplied path string when the
    # output landed outside the repo (relative_to() raises ValueError).
    try:
        rel = args.output.resolve().relative_to(_REPO_ROOT)
        git_path = str(rel)
    except ValueError:
        git_path = str(args.output)
    print(f"  git add {git_path}")
    print("  git commit -m 'fixture: regenerate v0.4.0 7-variant calibration baseline'")
    print("  git push origin <branch>")
    print()
    print("Then re-run the calibration:")
    print("  mamba run -n lmdiff pytest tests/integration/test_calibration_regression_7variant.py -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
