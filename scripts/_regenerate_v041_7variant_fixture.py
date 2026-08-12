"""Regenerate ``tests/fixtures/calibration_v041_7variant_summary.json``.

Runs the *exact* ``family()`` call that
``tests/integration/test_calibration_regression_7variant.py`` runs as
its ``cutover_result`` fixture, using the shared spec module
``tests/integration/_v041_7variant_spec.py``. Single source of truth:
neither the script nor the test inlines the kwargs.

Companion to ``scripts/_regenerate_v041_4variant_fixture.py`` (the
4-variant baseline). Same probe set (5 lm-eval domains), more
variants — adds ``chat``, ``temp_1.5`` (sample decode), and
``system_prompt`` (runtime-only Config mod) on top of the 4 unique-
model variants.

When to re-run:
  - Same triggers as the 4-variant script — engine / pipeline /
    measurement code changes that the calibration test would catch.
  - When the test skips because the v0.4.1 fixture is missing.

Usage (GPU box):
    cd <repo-root>
    mamba run -n lmdiff python scripts/_regenerate_v041_7variant_fixture.py

Optional flags:
    --output PATH    Override the fixture path (default: spec value)
    --no-progress    Suppress per-probe progress bars
    --dry-run        Print the call that would run; exit without loading models

Runtime: ~1-1.5 hours on RTX 5090 (Blackwell, sm_120) for 7 variants
× 500 probes × 2 score phases under the v0.4.1 lazy-engine pipeline.
``temp_1.5`` and ``system_prompt`` reuse base via the Fix 4 anchor map
so peak resident is still 2; only the 5 unique-model variants
(yarn / long / code / math / chat) trigger fresh loads.

After successful run:
    git add tests/fixtures/calibration_v041_7variant_summary.json
    git diff --stat tests/fixtures/        # confirm only the JSON changed
    git commit -m 'fixture: regenerate v0.4.1 7-variant calibration baseline'

Then re-run the test on GPU:
    mamba run -n lmdiff pytest tests/integration/test_calibration_regression_7variant.py -v -m "slow and gpu"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``tests.integration._v041_7variant_spec`` importable when the
# script runs from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.integration._v041_7variant_spec import (  # noqa: E402
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


def _print_summary(payload: dict) -> None:
    """Per-variant share table + domain status table for sanity check."""
    from collections import Counter

    print()
    print("─" * 78)
    print(" Sanity check — share_per_domain per (variant, domain)")
    print("─" * 78)
    print(f"  n_probes:        {payload.get('n_probes')}")
    print(f"  variant_names:   {payload.get('variant_names')}")
    pd = payload.get("probe_domains") or []
    print(f"  probe_domains:   {dict(Counter(pd))}")
    print()
    share = payload.get("share_per_domain") or {}
    if not share:
        print("  (no share_per_domain in payload — unexpected)")
        return
    domains = sorted({
        d for row in share.values() for d in row
    })
    name_w = max(len(v) for v in share) if share else 7
    cell_w = max(8, max(len(d) for d in domains) + 1)
    print(f"  {'variant':<{name_w}} | " + " | ".join(d.ljust(cell_w) for d in domains))
    print(f"  {'-' * name_w}-+-" + "-+-".join("-" * cell_w for _ in domains))
    for v, row in share.items():
        cells = []
        for d in domains:
            val = row.get(d)
            if val is None:
                cells.append("  —    ".ljust(cell_w))
            else:
                cells.append(f"{val * 100:5.1f}%".ljust(cell_w))
        print(f"  {v:<{name_w}} | " + " | ".join(cells))
    print()
    status = payload.get("domain_status") or {}
    if status:
        print("─" * 78)
        print(" Domain status per (variant, domain) — full / partial / variant_only / out_of_range")
        print("─" * 78)
        for v, row in status.items():
            print(f"  {v:<{name_w}}: {dict(row)}")
        print()


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
    print(f"v0.4.1 7-variant calibration fixture regeneration")
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

    import lmdiff  # noqa: F401
    from lmdiff.report.json_report import to_json_dict

    result = lmdiff.family(**kwargs)
    payload = to_json_dict(result)

    payload.pop("generated_at", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print(f"[regen] wrote {args.output}")
    _print_summary(payload)

    print("Next steps:")
    try:
        rel = args.output.resolve().relative_to(_REPO_ROOT)
        git_path = str(rel)
    except ValueError:
        git_path = str(args.output)
    print(f"  git add {git_path}")
    print("  git commit -m 'fixture: regenerate v0.4.1 7-variant calibration baseline'")
    print("  git push origin <branch>")
    print()
    print("Then re-run the calibration:")
    print("  mamba run -n lmdiff pytest tests/integration/test_calibration_regression_7variant.py -v -m 'slow and gpu'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
