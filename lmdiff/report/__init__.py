"""lmdiff.report — multi-channel rendering of GeoResult / DiffReport.

v0.3.0 commit 1.5 lays down the architecture: a :class:`Renderer`
Protocol (in :mod:`lmdiff.report._protocols`), a standard pipeline
(:func:`lmdiff.report.render`), and five renderer modules
(``terminal``, ``markdown``, ``html``, ``json_report``, ``figures``).
Concrete application-tier content lands in commits 1.7-1.10. Until
then, the renderers either delegate to their v0.2.x equivalents (where
they exist) or emit a minimal placeholder banner.

Imports are lazy: ``import lmdiff.report`` does NOT load matplotlib,
the html template engine, or anything heavy. First call to a specific
renderer module triggers its imports.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from lmdiff.report._pipeline import build_tables, render
    from lmdiff.report._protocols import Renderer


_LAZY_REPORT: dict[str, str] = {
    "Renderer": "lmdiff.report._protocols",
    "render": "lmdiff.report._pipeline",
    "build_tables": "lmdiff.report._pipeline",
}


def __getattr__(name: str):
    if name in _LAZY_REPORT:
        import importlib
        module = importlib.import_module(_LAZY_REPORT[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'lmdiff.report' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_REPORT.keys()))


__all__ = ["Renderer", "render", "build_tables"]
