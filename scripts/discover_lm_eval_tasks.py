#!/usr/bin/env python
"""Discover lm-evaluation-harness tasks and diff against KNOWN_TASK_DOMAINS.

Read-only tool. Does not modify adapters.py. Produces a report identifying
tasks in lm-eval's registry that are not yet in our KNOWN_TASK_DOMAINS,
suggests domain classifications, and can optionally verify that
from_lm_eval() succeeds on each unregistered task.

Usage:
    # Quick scan: just enumerate and diff (fast, no task instantiation)
    mamba run -n lmdiff python scripts/discover_lm_eval_tasks.py

    # Filter by glob pattern on task name
    mamba run -n lmdiff python scripts/discover_lm_eval_tasks.py --filter 'mmlu_*'

    # Deep verification: run from_lm_eval(task, limit=1) on each unregistered task
    mamba run -n lmdiff python scripts/discover_lm_eval_tasks.py --deep

    # Dump everything as JSON for downstream analysis
    mamba run -n lmdiff python scripts/discover_lm_eval_tasks.py --output-json discovery.json

    # Generate Python snippets ready to paste into adapters.py
    mamba run -n lmdiff python scripts/discover_lm_eval_tasks.py --generate-entries

Requires: pip install lmdiff-kit[lm-eval]
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lmdiff.probes.adapters import KNOWN_TASK_DOMAINS


KEYWORD_DOMAIN_HINTS: dict[str, str] = {
    "math": "math",
    "code": "code",
    "coding": "code",
    "program": "code",
    "reasoning": "reasoning",
    "logic": "reasoning",
    "qa": "knowledge",
    "trivia": "knowledge",
    "safety": "safety",
    "toxic": "safety",
    "bias": "safety",
    "read": "reading",
    "squad": "reading",
    "long": "long-context",
}


def _print_err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _import_lm_eval():
    """Return (lm_eval_module, lm_eval.tasks submodule, version) or exit(2)."""
    try:
        import lm_eval  # type: ignore[import-not-found]
        import lm_eval.tasks as _tasks  # type: ignore[import-not-found]
    except ImportError:
        _print_err(
            "ERROR: lm-eval is not installed.\n"
            "Install with: pip install lmdiff-kit[lm-eval]"
        )
        sys.exit(2)
    version = getattr(lm_eval, "__version__", "unknown")
    return lm_eval, _tasks, version


def enumerate_lm_eval_tasks(tasks_module) -> tuple[list[str], Any]:
    """Return (sorted task names, TaskManager instance) from lm-eval."""
    TaskManager = getattr(tasks_module, "TaskManager", None)
    if TaskManager is None:
        raise RuntimeError(
            "lm_eval.tasks has no TaskManager attribute. "
            "Confirm lm-eval>=0.4.0 is installed."
        )
    tm = TaskManager()

    all_tasks = getattr(tm, "all_tasks", None)
    if all_tasks is not None:
        return sorted(str(t) for t in all_tasks), tm

    task_index = getattr(tm, "task_index", None)
    if isinstance(task_index, dict):
        return sorted(task_index.keys()), tm

    raise RuntimeError(
        "TaskManager exposes neither `all_tasks` nor `task_index`. "
        "Script does not know how to enumerate tasks in this lm-eval version."
    )


def read_task_config_shallow(tm, name: str) -> dict[str, Any]:
    """Best-effort shallow read of task config without instantiating data.

    Returns {"output_type": ..., "metric_list": ..., "ok": bool}.
    Never raises; failures set ok=False and leave fields as None.
    """
    result: dict[str, Any] = {
        "output_type": None,
        "metric_list": None,
        "ok": False,
    }
    getter = getattr(tm, "_get_config", None)
    if getter is None:
        return result
    try:
        cfg = getter(name)
    except BaseException:  # noqa: BLE001 - some lm-eval tasks sys.exit() on missing deps (e.g. SWI-Prolog)
        return result
    if cfg is None:
        return result

    # lm-eval configs may be dicts, attribute-bags, or TaskConfig dataclass instances.
    def _get(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    output_type = _get(cfg, "output_type")
    if isinstance(output_type, str):
        result["output_type"] = output_type
    metric_list = _get(cfg, "metric_list")
    if metric_list is not None:
        # metric_list can be list[dict] (with "metric" keys) or list[str].
        flat: list[str] = []
        try:
            for m in metric_list:
                if isinstance(m, dict) and "metric" in m:
                    flat.append(str(m["metric"]))
                else:
                    flat.append(str(m))
        except TypeError:
            flat = []
        result["metric_list"] = flat or None
    result["ok"] = True
    return result


def suggest_domain(task_name: str, registered: set[str]) -> tuple[str, str]:
    """Suggest a domain for an unregistered task.

    Order:
      1. prefix match: task_name.split("_", 1)[0] ∈ KNOWN_TASK_DOMAINS.
         (Mirrors adapters._resolve_domain's fallback.)
      2. keyword match: any KEYWORD_DOMAIN_HINTS key in task_name.lower().
      3. "unknown".

    Returns (domain, reason) where reason is one of:
      "prefix:<key>", "keyword:<hint>", "none".
    """
    if "_" in task_name:
        prefix = task_name.split("_", 1)[0]
        if prefix in KNOWN_TASK_DOMAINS:
            return KNOWN_TASK_DOMAINS[prefix].domain, f"prefix:{prefix}"

    lowered = task_name.lower()
    for hint, domain in KEYWORD_DOMAIN_HINTS.items():
        if hint in lowered:
            return domain, f"keyword:{hint}"

    return "unknown", "none"


def verify_task_deep(task_name: str) -> dict[str, Any]:
    """Instantiate via from_lm_eval(limit=1). Capture any failure."""
    from lmdiff.probes.adapters import from_lm_eval

    info: dict[str, Any] = {
        "verified": False,
        "error_type": None,
        "error_msg": None,
        "n_probes": 0,
        "has_expected": None,
        "text_len": None,
    }
    try:
        ps = from_lm_eval(task_name, limit=1)
    except Exception as exc:  # noqa: BLE001 - deep mode catches every failure
        info["error_type"] = type(exc).__name__
        info["error_msg"] = str(exc)[:200]
        return info

    info["verified"] = True
    info["n_probes"] = len(ps)
    if len(ps) > 0:
        probe = ps[0]
        info["has_expected"] = probe.expected is not None
        info["text_len"] = len(probe.text)
    return info


def build_report(
    *,
    filter_pattern: str,
    deep: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Do the whole discovery dance. No side effects on the codebase."""
    lm_eval, tasks_module, version = _import_lm_eval()
    all_tasks, tm = enumerate_lm_eval_tasks(tasks_module)

    filtered = [t for t in all_tasks if fnmatch.fnmatchcase(t, filter_pattern)]
    registered_names = set(KNOWN_TASK_DOMAINS.keys())
    lm_eval_names = set(all_tasks)

    registered_and_in_lm_eval = sorted(registered_names & lm_eval_names)
    missing_from_lm_eval = sorted(registered_names - lm_eval_names)

    unregistered_filtered = sorted(set(filtered) - registered_names)

    if deep:
        _print_err(
            f"[WARN] --deep will instantiate each of "
            f"{len(unregistered_filtered)} unregistered tasks and call "
            f"from_lm_eval(name, limit=1). This may download HF datasets "
            f"and take hours. Use --filter to narrow scope."
        )

    unregistered_entries: list[dict[str, Any]] = []
    for i, name in enumerate(unregistered_filtered, start=1):
        cfg = read_task_config_shallow(tm, name)
        suggested_domain, reason = suggest_domain(name, registered_names)
        entry: dict[str, Any] = {
            "task_name": name,
            "output_type": cfg["output_type"],
            "metric_list": cfg["metric_list"],
            "config_ok": cfg["ok"],
            "suggested_domain": suggested_domain,
            "suggestion_reason": reason,
        }
        if deep:
            if verbose:
                _print_err(f"[deep {i}/{len(unregistered_filtered)}] {name} ...")
            deep_info = verify_task_deep(name)
            entry.update({
                "verified": deep_info["verified"],
                "verify_error_type": deep_info["error_type"],
                "verify_error_msg": deep_info["error_msg"],
                "verify_n_probes": deep_info["n_probes"],
                "verify_has_expected": deep_info["has_expected"],
                "verify_text_len": deep_info["text_len"],
            })
        else:
            entry["verified"] = None
            entry["verify_error_type"] = None
            entry["verify_error_msg"] = None
        unregistered_entries.append(entry)

    report = {
        "lm_eval_version": version,
        "filter": filter_pattern,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "known_registry_size": len(KNOWN_TASK_DOMAINS),
        "lm_eval_registry_size": len(all_tasks),
        "filter_matched": len(filtered),
        "registered_and_in_lm_eval": registered_and_in_lm_eval,
        "missing_from_lm_eval": missing_from_lm_eval,
        "unregistered": unregistered_entries,
        "deep": deep,
    }
    return report


def render_stdout_report(report: dict[str, Any], max_rows: int) -> None:
    print("=== lm-eval task discovery ===")
    print(f"lm-eval version: {report['lm_eval_version']}")
    print(f"Registry size: {report['lm_eval_registry_size']} tasks")
    print(f"Filter: {report['filter']!r} (matched {report['filter_matched']})")
    print(f"KNOWN_TASK_DOMAINS size: {report['known_registry_size']}")
    print(f"  - registered & in lm-eval: {len(report['registered_and_in_lm_eval'])}")
    print(f"  - registered but MISSING from lm-eval: {len(report['missing_from_lm_eval'])}"
          + ("  ← warn" if report["missing_from_lm_eval"] else ""))
    print(f"  - in lm-eval but UNREGISTERED (after filter): {len(report['unregistered'])}")
    print()

    if report["missing_from_lm_eval"]:
        print("=== Missing from lm-eval registry (check for typos / renames) ===")
        for name in report["missing_from_lm_eval"]:
            print(f"  {name}")
        print()

    entries = report["unregistered"]
    if not entries:
        print("(no unregistered tasks to display)")
        return

    # Group by suggested_domain, then by reason
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        by_domain[e["suggested_domain"]].append(e)

    print(f"=== Unregistered tasks (first {max_rows}, grouped by suggested domain) ===")
    shown = 0
    truncated = False
    for domain in sorted(by_domain.keys()):
        group = by_domain[domain]
        reason_group = defaultdict(list)
        for e in group:
            reason_group[e["suggestion_reason"]].append(e)
        for reason in sorted(reason_group.keys()):
            subgroup = reason_group[reason]
            print(f"\n[{domain}] (suggested via {reason}, {len(subgroup)} tasks)")
            name_w = max((len(e["task_name"]) for e in subgroup[: max_rows - shown]), default=20)
            name_w = max(name_w, len("task name"))
            header = f"  {'task name':<{name_w}}  {'output_type':<20}  {'verified':<12}"
            print(header)
            for e in subgroup:
                if shown >= max_rows:
                    truncated = True
                    break
                verified = (
                    "(not in --deep)" if e["verified"] is None
                    else "ok" if e["verified"] else f"FAIL: {e['verify_error_type']}"
                )
                ot = e["output_type"] or "?"
                print(f"  {e['task_name']:<{name_w}}  {ot:<20}  {verified:<12}")
                shown += 1
            if truncated:
                break
        if truncated:
            break

    if truncated:
        hidden = len(entries) - shown
        print(f"\n  ... ({hidden} more; raise --max-unregistered or use --output-json)")


def render_generate_entries(entries: list[dict[str, Any]]) -> None:
    print()
    print("# --- Generated TaskInfo entries (review and paste into adapters.py) ---")
    print("# Suggested domain is a heuristic; verify before committing.")
    print("# requires_execution defaults to False — flip to True for humaneval/mbpp-style tasks.")
    print()
    for e in entries:
        domain = e["suggested_domain"]
        reason = e["suggestion_reason"]
        output_type = e["output_type"] or "generate_until"
        metrics = e.get("metric_list") or []
        native = metrics[0] if metrics else "acc"
        print(
            f'    "{e["task_name"]}": TaskInfo('
            f'"{domain}", "{native}", False, "{output_type}"),  '
            f"# suggested via {reason}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diff lm-eval task registry vs KNOWN_TASK_DOMAINS.",
    )
    parser.add_argument(
        "--filter", default="*",
        help="fnmatch-style glob on task names (default: '*')",
    )
    parser.add_argument(
        "--deep", action="store_true",
        help="Instantiate each unregistered task via from_lm_eval(limit=1); slow.",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Write full (non-truncated) discovery report to PATH.",
    )
    parser.add_argument(
        "--generate-entries", action="store_true",
        help="Emit Python TaskInfo(...) snippets for unregistered tasks.",
    )
    parser.add_argument(
        "--max-unregistered", type=int, default=50,
        help="Max rows to show in the stdout table (default: 50).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Per-task progress in --deep mode (to stderr).",
    )
    args = parser.parse_args(argv)

    report = build_report(
        filter_pattern=args.filter,
        deep=args.deep,
        verbose=args.verbose,
    )

    render_stdout_report(report, max_rows=args.max_unregistered)

    if args.output_json is not None:
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nJSON report written to {args.output_json}")

    if args.generate_entries:
        render_generate_entries(report["unregistered"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
