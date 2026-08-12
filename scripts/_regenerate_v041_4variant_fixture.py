"""Regenerate ``tests/fixtures/calibration_v041_4variant_baseline.json``.

Runs the *exact* ``family()`` call that
``tests/integration/test_calibration_regression.py`` runs as its
``cutover_result`` fixture, using the shared spec module
``tests/integration/_v041_4variant_spec.py``. Single source of truth:
neither the script nor the test inlines the kwargs.

When to re-run:
  - After any change to ``lmdiff._pipeline``, ``lmdiff._engine``,
    ``lmdiff._validity``, or other engine / measurement-touching code
    that the calibration test would catch.
  - After bumping the canonical seed, n_probes, or variant set in the
    shared spec.
  - When ``test_calibration_regression.py`` skips because the fixture
    file is missing on a fresh checkout.

Usage (GPU box):
    cd <repo-root>
    mamba run -n lmdiff python scripts/_regenerate_v041_4variant_fixture.py

Optional flags:
    --output PATH    Override the fixture path (default: spec value)
    --no-progress    Suppress per-probe progress bars
    --dry-run        Print the call that would run; exit without loading models

Runtime: ~30-45 minutes on RTX 5090 (Blackwell, sm_120) for 4 variants
× 500 probes × 2 score phases under the v0.4.1 lazy-engine pipeline.
Watch the ``[lmdiff lifecycle]`` lines for engine_load / engine_release
interleaving — peak resident should be 2 (base + active variant).

After successful run:
    git add tests/fixtures/calibration_v041_4variant_baseline.json
    git diff --stat tests/fixtures/        # confirm only the JSON changed
    git commit -m 'fixture: regenerate v0.4.1 4-variant calibration baseline'

Then re-run the test on GPU:
    mamba run -n lmdiff pytest tests/integration/test_calibration_regression.py -v -m "slow and gpu"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``tests.integration._v041_4variant_spec`` importable when the
# script runs from the repo root (``python scripts/...``). The
# integration test runs under pytest which already adds the repo root
# to sys.path; this script doesn't, so do it explicitly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.integration._v041_4variant_spec import (  # noqa: E402
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
    """Print a per-variant share table so the user can sanity-check
    before committing. Out_of_range / variant_only cells render as
    ``  —  `` so the regen output mirrors what users see in the
    docs and figures."""
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
    print(f"v0.4.1 4-variant calibration fixture regeneration")
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

    print("[regen] loading lmdiff and running family() — expect ~30-45 min on a single GPU")
    print()

    import lmdiff  # noqa: F401  (heavy import, scoped here)
    from lmdiff.report.json_report import to_json_dict

    result = lmdiff.family(**kwargs)
    payload = to_json_dict(result)

    # Strip volatile field — the test doesn't compare it.
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
    print("  git commit -m 'fixture: regenerate v0.4.1 4-variant calibration baseline'")
    print("  git push origin <branch>")
    print()
    print("Then re-run the calibration:")
    print("  mamba run -n lmdiff pytest tests/integration/test_calibration_regression.py -v -m 'slow and gpu'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
