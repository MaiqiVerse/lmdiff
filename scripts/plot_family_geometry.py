#!/usr/bin/env python
"""Plot all geometry figures from a GeoResult JSON (script wrapper).

Thin argparse wrapper around ``lmdiff.experiments.family.plot_family_geometry``.
New callers should prefer either:

  - the library API: ``from lmdiff.experiments import plot_family_geometry``
  - the CLI:         ``lmdiff plot-geometry path/to/georesult.json --output-dir ...``

Usage:
    mamba run -n lmdiff python scripts/plot_family_geometry.py \\
        examples/family_geometry_extended.json \\
        --output-dir examples/family_geometry_extended_figures/

Requires: pip install lmdiff-kit[viz]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lmdiff.experiments.family import plot_family_geometry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot all geometry figures from a GeoResult JSON.",
    )
    parser.add_argument("input", type=Path, help="GeoResult JSON path")
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write PNGs + index.html (created if missing)",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: {args.input} does not exist", file=sys.stderr)
        return 2

    rendered = plot_family_geometry(args.input, args.output_dir)
    return 0 if rendered else 1


if __name__ == "__main__":
    sys.exit(main())
