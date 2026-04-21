"""Adapters that load external task specs into lmdiff ProbeSets.

The lm-eval-harness integration is lazy: `lm_eval` is only imported
inside `from_lm_eval()` so importing `lmdiff.probes.adapters` on a
machine without `lm-eval` installed is always safe.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lmdiff.probes.loader import Probe, ProbeSet

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


__all__ = [
    "TaskInfo",
    "KNOWN_TASK_DOMAINS",
    "from_lm_eval",
    "from_hf_dataset",
]


_SUPPORTED_OUTPUT_TYPES = frozenset({
    "multiple_choice",
    "generate_until",
    "loglikelihood",
    "loglikelihood_rolling",
})


@dataclass(frozen=True)
class TaskInfo:
    """Metadata for a known lm-eval task."""
    domain: str
    native_metric: str
    requires_execution: bool
    output_type: str
    notes: str = ""


KNOWN_TASK_DOMAINS: dict[str, TaskInfo] = {
    # --- commonsense ---
    "hellaswag": TaskInfo("commonsense", "acc_norm", False, "multiple_choice"),
    "winogrande": TaskInfo("commonsense", "acc", False, "multiple_choice"),
    "piqa": TaskInfo("commonsense", "acc_norm", False, "multiple_choice"),
    "openbookqa": TaskInfo("commonsense", "acc_norm", False, "multiple_choice"),
    "commonsense_qa": TaskInfo("commonsense", "acc", False, "multiple_choice"),
    # --- reasoning ---
    "arc_challenge": TaskInfo("reasoning", "acc_norm", False, "multiple_choice"),
    "arc_easy": TaskInfo("reasoning", "acc_norm", False, "multiple_choice"),
    "logiqa": TaskInfo("reasoning", "acc_norm", False, "multiple_choice"),
    # --- math ---
    "gsm8k": TaskInfo(
        "math", "exact_match", False, "generate_until",
        notes="Use Gsm8kNumberMatch evaluator; extract number after '####' or last number.",
    ),
    "mathqa": TaskInfo("math", "acc_norm", False, "multiple_choice"),
    # --- knowledge (MMLU subsets) ---
    "mmlu": TaskInfo(
        "knowledge", "acc", False, "multiple_choice",
        notes="Aggregate task; prefer specific subsets.",
    ),
    "mmlu_college_computer_science": TaskInfo(
        "code", "acc", False, "multiple_choice",
        notes="Code-understanding MCQ, non-execution. Default code axis.",
    ),
    "mmlu_computer_security": TaskInfo(
        "code", "acc", False, "multiple_choice",
    ),
    "mmlu_machine_learning": TaskInfo(
        "code", "acc", False, "multiple_choice",
    ),
    "mmlu_high_school_mathematics": TaskInfo(
        "math", "acc", False, "multiple_choice",
    ),
    "mmlu_college_mathematics": TaskInfo(
        "math", "acc", False, "multiple_choice",
    ),
    # --- reading / QA ---
    "boolq": TaskInfo("reading", "acc", False, "multiple_choice"),
    "squadv2": TaskInfo("reading", "f1", False, "generate_until"),
    "triviaqa": TaskInfo("knowledge", "exact_match", False, "generate_until"),
    "naturalqs": TaskInfo("knowledge", "exact_match", False, "generate_until"),
    # --- language ---
    "lambada_openai": TaskInfo("language", "acc", False, "loglikelihood"),
    "wikitext": TaskInfo("language", "word_perplexity", False, "loglikelihood_rolling"),
    # --- long-context ---
    "longbench_2wikimqa": TaskInfo("long-context", "f1", False, "generate_until"),
    "longbench_hotpotqa": TaskInfo("long-context", "f1", False, "generate_until"),
    "longbench_narrativeqa": TaskInfo("long-context", "f1", False, "generate_until"),
    "longbench_qasper": TaskInfo("long-context", "f1", False, "generate_until"),
    # --- code (execution required; available for δ-only experiments) ---
    "humaneval": TaskInfo(
        "code", "pass@1", True, "generate_until",
        notes="Native metric requires sandbox. Use magnitude-only for δ experiments.",
    ),
    "mbpp": TaskInfo(
        "code", "pass@1", True, "generate_until",
        notes="Native metric requires sandbox. Use magnitude-only for δ experiments.",
    ),
    # --- safety ---
    "truthfulqa_mc1": TaskInfo("safety", "acc", False, "multiple_choice"),
    "truthfulqa_mc2": TaskInfo("safety", "acc", False, "multiple_choice"),
    "toxigen": TaskInfo("safety", "acc", False, "multiple_choice"),
}


def _resolve_domain(task_name: str, info: TaskInfo | None) -> str:
    """KNOWN_TASK_DOMAINS → prefix → 'unknown'."""
    if info is not None:
        return info.domain
    if "_" in task_name:
        prefix = task_name.split("_", 1)[0]
        prefix_info = KNOWN_TASK_DOMAINS.get(prefix)
        if prefix_info is not None:
            return prefix_info.domain
    return "unknown"


def _task_output_type(task: Any) -> str:
    """Pull output_type off the task, tolerant to lm-eval minor-version API drift."""
    # lm-eval 0.4.x: task.OUTPUT_TYPE or task._config.output_type
    for attr in ("OUTPUT_TYPE", "output_type"):
        val = getattr(task, attr, None)
        if isinstance(val, str):
            return val
    cfg = getattr(task, "_config", None) or getattr(task, "config", None)
    if cfg is not None:
        val = getattr(cfg, "output_type", None)
        if isinstance(val, str):
            return val
        if isinstance(cfg, dict) and isinstance(cfg.get("output_type"), str):
            return cfg["output_type"]
    raise NotImplementedError(
        "Could not determine lm-eval task output_type via OUTPUT_TYPE / output_type / _config.output_type."
    )


def _collect_docs(task: Any) -> list[Any]:
    """test_docs → validation_docs → training_docs, materialized to list."""
    for method_name in ("test_docs", "validation_docs", "training_docs"):
        method = getattr(task, method_name, None)
        if method is None:
            continue
        try:
            docs_iter = method()
        except Exception:  # noqa: BLE001 - some tasks raise if split absent
            continue
        if docs_iter is None:
            continue
        # Materialize now; some lm-eval iterators are single-pass.
        docs = list(docs_iter)
        if docs:
            return docs
    return []


def _render_prompt(task: Any, doc: Any, num_fewshot: int | None) -> str:
    """Build the prompt string, honoring fewshot when requested."""
    if num_fewshot is not None and num_fewshot > 0:
        # Try newer then older fewshot_context API shapes.
        for attr in ("fewshot_context", "build_context"):
            method = getattr(task, attr, None)
            if method is None:
                continue
            try:
                return str(method(doc=doc, num_fewshot=num_fewshot))
            except TypeError:
                try:
                    return str(method(doc, num_fewshot))
                except Exception:  # noqa: BLE001
                    continue
            except Exception:  # noqa: BLE001
                continue
    return str(task.doc_to_text(doc))


def _render_target(task: Any, doc: Any) -> tuple[str | None, list[str]]:
    """Return (primary_expected, aliases).

    Multi-answer lm-eval tasks return list[str]; multi-choice tasks
    return an int choice index. Fall back to str(target) when shape is
    unexpected.
    """
    target = task.doc_to_target(doc)

    if isinstance(target, list):
        if not target:
            return None, []
        primary = str(target[0]).strip()
        aliases = [str(t).strip() for t in target[1:]]
        return primary, aliases

    if isinstance(target, int):
        choices: list[str] | None = None
        choice_getter = getattr(task, "doc_to_choice", None)
        if choice_getter is not None:
            try:
                maybe_choices = choice_getter(doc)
                if maybe_choices is not None:
                    choices = [str(c) for c in maybe_choices]
            except Exception:  # noqa: BLE001
                choices = None
        if choices is None and isinstance(doc, dict):
            for key in ("choices", "options"):
                raw = doc.get(key)
                if isinstance(raw, list):
                    choices = [str(c) for c in raw]
                    break
                if isinstance(raw, dict) and "text" in raw and isinstance(raw["text"], list):
                    choices = [str(c) for c in raw["text"]]
                    break
        if choices is not None and 0 <= target < len(choices):
            return choices[target].strip(), []
        # Last-resort: hand back the raw index so callers can inspect.
        return str(target), []

    if target is None:
        return None, []
    return str(target).strip(), []


def from_lm_eval(
    task_name: str,
    limit: int | None = None,
    num_fewshot: int | None = None,
    seed: int = 42,
) -> ProbeSet:
    """Load an lm-evaluation-harness task and convert to a ProbeSet.

    Each doc becomes a Probe with:
        - id: f"{task_name}:{doc_idx}"
        - text: rendered prompt (doc_to_text, optionally with fewshot context)
        - domain: resolved via KNOWN_TASK_DOMAINS, then prefix fallback, then "unknown"
        - expected: primary doc_to_target, str (or None if absent)
        - metadata: {
              "task_name", "native_metric", "output_type",
              "requires_execution", "doc_idx",
              "aliases"?  (present only when doc_to_target returned a list of alternatives),
          }

    Args:
        task_name: lm-eval task name (must exist in lm-eval's task registry).
        limit: truncate to the first `limit` docs after a seeded shuffle.
        num_fewshot: override task's default fewshot (None = task default).
        seed: deterministic shuffle seed.

    Raises:
        ImportError: if lm-eval not installed (pip install lmdiff-kit[lm-eval]).
        KeyError: task_name not in lm-eval's registry.
        NotImplementedError: output_type not in the supported whitelist.
    """
    try:
        from lm_eval import tasks as _lm_eval_tasks  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - covered via no_lm_eval test fixture
        raise ImportError(
            "lm-eval not installed. Install with: pip install lmdiff-kit[lm-eval]"
        ) from exc

    try:
        task_dict = _lm_eval_tasks.get_task_dict([task_name])
    except KeyError:
        raise KeyError(
            f"Task '{task_name}' not found in lm-eval task registry."
        ) from None

    if not task_dict or task_name not in task_dict:
        raise KeyError(
            f"Task '{task_name}' not found in lm-eval task registry."
        )

    task = task_dict[task_name]
    # Newer lm-eval returns a (group, task) pair in nested contexts.
    if isinstance(task, tuple) and len(task) == 2:
        task = task[1]

    output_type = _task_output_type(task)
    if output_type not in _SUPPORTED_OUTPUT_TYPES:
        raise NotImplementedError(
            f"Task '{task_name}' has output_type '{output_type}' which is not "
            "yet supported (multi-turn / agent / tool-use tasks deferred to future work)."
        )

    docs = _collect_docs(task)
    rng = random.Random(seed)
    rng.shuffle(docs)
    if limit is not None:
        docs = docs[:limit]

    info = KNOWN_TASK_DOMAINS.get(task_name)
    resolved_domain = _resolve_domain(task_name, info)

    version = None
    raw_version = getattr(task, "VERSION", None)
    if raw_version is not None:
        version = str(raw_version)
    else:
        version = "lm-eval-harness"

    probes: list[Probe] = []
    for i, doc in enumerate(docs):
        text = _render_prompt(task, doc, num_fewshot=num_fewshot)
        primary, aliases = _render_target(task, doc)

        meta: dict[str, Any] = {
            "task_name": task_name,
            "native_metric": info.native_metric if info is not None else None,
            "output_type": output_type,
            "requires_execution": info.requires_execution if info is not None else False,
            "doc_idx": i,
        }
        if aliases:
            meta["aliases"] = aliases

        probes.append(Probe(
            id=f"{task_name}:{i}",
            text=text,
            domain=resolved_domain,
            expected=primary,
            metadata=meta,
        ))

    return ProbeSet(probes, name=f"lm_eval:{task_name}", version=version)


def from_hf_dataset(
    dataset_name: str,
    split: str = "test",
    text_field: str = "question",
    expected_field: str | None = None,
    domain: str | None = None,
    limit: int | None = None,
) -> ProbeSet:
    """Load probes from a HuggingFace dataset. Phase 2."""
    raise NotImplementedError("from_hf_dataset is Phase 2")
