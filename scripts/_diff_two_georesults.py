"""Compare two GeoResult JSON files field-by-field.

Used for the back-to-back same-seed reproducibility check after Fix 3:
run the v0.4.0 cutover twice with the same seed and compare the two
``family_geometry.json`` payloads (or ``calibration_v040_7variant_summary.json``-
shaped payloads from the regen script). If everything is byte-identical
within the float tolerance, Fix 3's seed plumbing landed correctly and
sample-decode variants are reproducible. If a field differs, the diff
points at where to investigate (probably the engine layer, or hardware
non-determinism on a specific reduction).

Usage:
    mamba run -n lmdiff python scripts/_diff_two_georesults.py PATH_A PATH_B
    mamba run -n lmdiff python scripts/_diff_two_georesults.py PATH_A PATH_B --tol 1e-9

Behaviour:
  - Walks both payloads recursively, comparing leaves.
  - Numeric leaves are compared with ``abs(a - b) <= tol`` (default 1e-12;
    matches what "byte-identical for our purposes" means in float math).
  - Non-numeric leaves (strings, ints, bools, None) are compared with ``==``.
  - Lists must match in length; element-wise comparison.
  - Dicts must match in key set; per-key recursion.
  - Reports the worst diff per field as ``max |Δ|``.
  - Volatile fields like ``generated_at`` are skipped automatically.

Exit codes:
  0  identical within tolerance — Fix 3 reproducibility holds
  1  one or more fields differ — investigate
  2  structural mismatch (different keys, list lengths, types) — bigger problem
  3  invocation error (file not found, malformed JSON, etc.)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Fields that are expected to differ across runs (timestamps, etc.) —
# walked but never reported as a diff. Add new entries sparingly; the
# whole point of this script is to surface unexpected differences.
_VOLATILE_KEYS: frozenset[str] = frozenset({
    "generated_at",
})


def _is_number(x: Any) -> bool:
    """True for actual numerics that participate in tolerance compare.

    Excludes ``bool`` (a subclass of ``int`` in Python — a ``True``
    field flipping to ``False`` is a structural change, not a small
    numeric drift, and should be reported even at zero ``tol``).
    """
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _compare(
    a: Any, b: Any, *, path: str, tol: float, diffs: list[tuple[str, str]],
) -> None:
    """Recursive walk. Appends one ``(path, message)`` per failure to diffs."""
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            diffs.append((path, f"type mismatch: {type(a).__name__} vs {type(b).__name__}"))
            return
        keys_a = set(a) - _VOLATILE_KEYS
        keys_b = set(b) - _VOLATILE_KEYS
        if keys_a != keys_b:
            only_a = sorted(keys_a - keys_b)
            only_b = sorted(keys_b - keys_a)
            diffs.append((path, f"key sets differ — only in A: {only_a}, only in B: {only_b}"))
            # Continue with intersection so we still surface inner diffs.
        for k in sorted(keys_a & keys_b):
            _compare(a[k], b[k], path=f"{path}.{k}" if path else k, tol=tol, diffs=diffs)
        return

    if isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)):
            diffs.append((path, f"type mismatch: {type(a).__name__} vs {type(b).__name__}"))
            return
        if len(a) != len(b):
            diffs.append((path, f"list length differs: {len(a)} vs {len(b)}"))
            return
        max_abs_diff = 0.0
        first_offender_idx: int | None = None
        for i, (av, bv) in enumerate(zip(a, b)):
            if _is_number(av) and _is_number(bv):
                # NaN-aware: NaN != NaN by IEEE, but for fixture
                # comparison we treat (NaN, NaN) as equal.
                if math.isnan(av) and math.isnan(bv):
                    continue
                d = abs(float(av) - float(bv))
                if d > tol and d > max_abs_diff:
                    max_abs_diff = d
                    first_offender_idx = i
            else:
                _compare(av, bv, path=f"{path}[{i}]", tol=tol, diffs=diffs)
        if first_offender_idx is not None:
            diffs.append((
                path,
                f"max |Δ| = {max_abs_diff:.3e} (first at [{first_offender_idx}]: "
                f"{a[first_offender_idx]} vs {b[first_offender_idx]})",
            ))
        return

    # Leaves.
    if _is_number(a) and _is_number(b):
        if math.isnan(a) and math.isnan(b):
            return
        d = abs(float(a) - float(b))
        if d > tol:
            diffs.append((path, f"|Δ| = {d:.3e} ({a} vs {b})"))
        return

    if a != b:
        diffs.append((path, f"value mismatch: {a!r} vs {b!r}"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path_a", type=Path, help="First GeoResult JSON")
    parser.add_argument("path_b", type=Path, help="Second GeoResult JSON")
    parser.add_argument(
        "--tol", type=float, default=1e-12,
        help="Numeric tolerance per leaf (default 1e-12, ~float64 noise floor)",
    )
    args = parser.parse_args()

    for p in (args.path_a, args.path_b):
        if not p.exists():
            print(f"[error] file not found: {p}", file=sys.stderr)
            return 3
    try:
        a = json.loads(args.path_a.read_text(encoding="utf-8"))
        b = json.loads(args.path_b.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[error] JSON decode failed: {e}", file=sys.stderr)
        return 3

    diffs: list[tuple[str, str]] = []
    _compare(a, b, path="", tol=args.tol, diffs=diffs)

    print(f"A: {args.path_a}")
    print(f"B: {args.path_b}")
    print(f"tol: {args.tol}")
    print()

    if not diffs:
        print(f"All fields identical within tolerance ({args.tol}).")
        return 0

    # Heuristic: structural diffs (key/length mismatches, type mismatches)
    # are exit 2; pure numeric drift is exit 1.
    structural_markers = (
        "type mismatch", "key sets differ", "list length differs",
        "value mismatch",
    )
    structural = any(
        any(m in msg for m in structural_markers) for _, msg in diffs
    )

    print(f"{len(diffs)} field(s) differ:")
    for path, msg in diffs:
        print(f"  {path or '<root>'}: {msg}")

    return 2 if structural else 1


if __name__ == "__main__":
    raise SystemExit(main())
