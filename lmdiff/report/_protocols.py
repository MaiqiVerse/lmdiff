"""Protocol definitions for lmdiff result renderers (commit 1.5).

The :class:`Renderer` Protocol is :pep:`544`-style and
:func:`runtime_checkable`. Concrete renderers in
``lmdiff.report.{terminal,markdown,html,json_report,figures}`` conform via
duck typing, no inheritance required. v0.3.0 ships skeleton renderers
that delegate to their v0.2.x equivalents (where they exist) or print a
placeholder banner; commits 1.7-1.10 fill in the application-tier
content without restructuring this layer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lmdiff.geometry import GeoResult


@runtime_checkable
class Renderer(Protocol):
    """Render a :class:`~lmdiff.geometry.GeoResult` to some output channel.

    Return type varies by renderer:

      - ``terminal``: ``str`` (also writes to stdout when ``file`` kwarg given)
      - ``markdown``: ``str`` (or writes to ``path`` when given)
      - ``html``:     ``str`` (or writes to ``path`` when given)
      - ``json``:     ``dict`` (or writes to ``path`` when given)
      - ``figures``:  ``dict[str, Path]`` of ``{key: path}``
    """

    def render(self, result: "GeoResult", **kwargs: Any) -> Any:
        """Render ``result`` to this renderer's output channel."""
        ...


__all__ = ["Renderer"]
