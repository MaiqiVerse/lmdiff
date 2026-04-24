"""Internal: polymorphic GeoResult loader for the family-figure suite."""
from __future__ import annotations

import json
from pathlib import Path

from lmdiff.geometry import GeoResult


def _load_geo(source: GeoResult | dict | str | Path) -> GeoResult:
    """Accept a GeoResult, parsed dict, or path to GeoResult JSON."""
    if isinstance(source, GeoResult):
        return source
    if isinstance(source, dict):
        from lmdiff.report.json_report import geo_result_from_json_dict
        return geo_result_from_json_dict(source)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"GeoResult JSON not found: {path}")
    from lmdiff.report.json_report import geo_result_from_json_dict
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return geo_result_from_json_dict(payload)


__all__ = ["_load_geo"]
