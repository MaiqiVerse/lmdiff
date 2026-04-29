"""Tiny progress-bar helper used by per-probe loops in the engine and
by phase markers in ``ChangeGeometry`` / ``family``.

Goals:
  - Zero new dependencies — uses ``rich`` which is already core.
  - Off by default; opt in via ``progress=True`` or env var.
  - A no-op when stdout is not a tty (stays silent in pipelines / log
    redirection so ``run.py > run.log`` doesn't bloat the file with
    ANSI escapes).

The helper is dual-shaped:

    iterate(items, *, desc=..., enable=...)
        Wraps any iterable. Yields the items as they go. Renders a
        single-line progress bar with count + elapsed + ETA when
        ``enable`` is True and stdout is a tty.

    phase(desc, *, enable=...)
        Context manager for a chunky step that has no inner loop
        ("loading variant 3/7: math"). Prints ``[hh:mm:ss] desc`` on
        entry and ``[hh:mm:ss] desc done in 4m12s`` on exit.

Both fall back to plain ``print`` when ``rich`` is not installed (which
shouldn't happen since ``rich`` is a core dep — but the fallback keeps
the import safe in stripped-down environments).
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


_LMDIFF_PROGRESS_ENV = "LMDIFF_PROGRESS"


def _is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _resolve_enable(enable: bool | None) -> bool:
    """Decide whether progress should render.

    Precedence:
      1. Explicit ``enable=True`` / ``enable=False`` always wins.
      2. ``LMDIFF_PROGRESS=0`` / ``LMDIFF_PROGRESS=1`` overrides default.
      3. Default: render iff stdout is a tty.
    """
    if enable is not None:
        return bool(enable)
    env = os.environ.get(_LMDIFF_PROGRESS_ENV)
    if env is not None:
        return env.strip() not in ("", "0", "false", "False", "no")
    return _is_tty()


def iterate(
    items: Iterable[T],
    *,
    desc: str = "",
    total: int | None = None,
    enable: bool | None = None,
) -> Iterator[T]:
    """Yield from ``items`` while rendering an optional progress bar.

    The bar shows ``desc`` + count + elapsed + ETA. Updates are coalesced
    to ~10 Hz so a tight inner loop doesn't spam stdout.
    """
    if not _resolve_enable(enable):
        yield from items
        return

    # Resolve total: prefer explicit, else len() if available.
    if total is None:
        try:
            total = len(items)  # type: ignore[arg-type]
        except TypeError:
            total = None

    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
    except ImportError:
        # Cheap fallback: a periodic dot-tick.
        last = time.time()
        for i, item in enumerate(items, 1):
            yield item
            now = time.time()
            if now - last > 5.0:
                sys.stdout.write(f"  [{desc}] {i}{'/' + str(total) if total else ''}\n")
                sys.stdout.flush()
                last = now
        return

    columns = [
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
    ]
    with Progress(*columns, transient=False) as bar:
        task_id = bar.add_task(desc or "working", total=total)
        for item in items:
            yield item
            bar.advance(task_id)


@contextmanager
def phase(desc: str, *, enable: bool | None = None) -> Iterator[None]:
    """Wrap a chunky step (no inner loop) with start/stop timing prints.

    Uses ``[HH:MM:SS]`` timestamps so a long log can be scanned for the
    expensive phases. Off when stdout isn't a tty unless ``enable=True``.
    """
    if not _resolve_enable(enable):
        yield
        return

    t0 = time.time()
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {desc} ...", flush=True)
    try:
        yield
    finally:
        dt = time.time() - t0
        if dt < 60:
            took = f"{dt:.1f}s"
        elif dt < 3600:
            took = f"{int(dt // 60)}m{int(dt % 60)}s"
        else:
            took = f"{int(dt // 3600)}h{int((dt % 3600) // 60)}m"
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {desc} done in {took}", flush=True)


def device_map_summary(model) -> str | None:
    """Return a short one-line summary if the model is sharded across
    multiple devices. Returns None when everything sits on one device.

    Designed to surface the silent CPU-spillover failure mode where
    ``device_map="auto"`` puts some layers on CPU under VRAM pressure
    and forward passes then run partly on CPU at ~0% GPU util.
    """
    dm = getattr(model, "hf_device_map", None)
    if not dm:
        return None
    by_device: dict[str, int] = {}
    for layer, dev in dm.items():
        key = str(dev)
        by_device[key] = by_device.get(key, 0) + 1
    if len(by_device) <= 1:
        return None
    parts = ", ".join(f"{k}={v}" for k, v in sorted(by_device.items()))
    return f"hf_device_map sharded across devices: {parts}"


_LMDIFF_DEBUG_ENGINE_LIFECYCLE_ENV = "LMDIFF_DEBUG_ENGINE_LIFECYCLE"


def engine_lifecycle_enabled() -> bool:
    """True if engine-lifecycle debug logging has been opted into.

    Off by default. Turn on by exporting
    ``LMDIFF_DEBUG_ENGINE_LIFECYCLE=1`` for runs where you want to see
    every InferenceEngine init / cache hit / release event — useful for
    diagnosing OOMs and wasted-load patterns in multi-variant family runs.
    """
    val = os.environ.get(_LMDIFF_DEBUG_ENGINE_LIFECYCLE_ENV)
    if val is None:
        return False
    return val.strip() not in ("", "0", "false", "False", "no")


def lifecycle_log(event: str, **fields: object) -> None:
    """Emit one ``[lmdiff lifecycle]`` line when the env var is set.

    No-op otherwise. ``fields`` formats as ``key=value`` (repr) pairs.
    """
    if not engine_lifecycle_enabled():
        return
    parts = " ".join(f"{k}={v!r}" for k, v in fields.items())
    sys.stdout.write(f"[lmdiff lifecycle] {event} {parts}\n")
    sys.stdout.flush()


__all__ = [
    "iterate",
    "phase",
    "device_map_summary",
    "engine_lifecycle_enabled",
    "lifecycle_log",
]
