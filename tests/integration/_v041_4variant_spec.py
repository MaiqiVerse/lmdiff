"""Shared spec for the v0.4.1 4-variant calibration baseline.

Mirrors ``_v041_7variant_spec.py`` for the 4-variant probe set
(``yarn / long / math / code`` against Llama-2-7b base, the historical
calibration set). Single source of truth: both
``test_calibration_regression.py`` and
``scripts/_regenerate_v041_4variant_fixture.py`` import from here.

The "4-variant" label refers to the 4 variants, not the probe set.
The probe set is the same 5-domain ``lm_eval:*`` mix used by the
7-variant calibration (commonsense / reasoning / math / code /
long-context). The 4-variant test was historically the byte-equivalence
gate for the v0.4.0 backend cutover; v0.4.1 must regenerate this
fixture too because the formula change shifts pdn values by factor
``√T̄_d`` everywhere — the v0.3.2 byte-equivalence is no longer the
contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "calibration_v041_4variant_baseline.json"


def build_run_kwargs() -> dict[str, Any]:
    """Build the exact ``family()`` kwargs for the 4-variant case.

    Matches the v0.4.0 4-variant calibration script — same probe set,
    same variant set, same seed. Difference under v0.4.1 is downstream:
    pdn formula and validity framework reshape every per-domain number,
    but the per-probe ``δ`` computation is identical (validity-aware
    skipping aside).
    """
    return {
        "base": "meta-llama/Llama-2-7b-hf",
        "variants": {
            "yarn": "NousResearch/Yarn-Llama-2-7b-128k",
            "long": "togethercomputer/LLaMA-2-7B-32K",
            "code": "codellama/CodeLlama-7b-hf",
            "math": "EleutherAI/llemma_7b",
        },
        "probes": (
            "lm_eval:hellaswag+arc_challenge+gsm8k"
            "+mmlu_college_computer_science+longbench_2wikimqa"
        ),
        "n_probes": 100,
        "max_new_tokens": 16,
        "task_overrides": {
            "gsm8k": {"max_new_tokens": 256},
            "longbench_2wikimqa": {"max_new_tokens": 128},
        },
        "seed": 42,
    }


# All 4 variant names in iteration order.
ALL_VARIANTS: tuple[str, ...] = ("yarn", "long", "code", "math")


__all__ = [
    "ALL_VARIANTS",
    "FIXTURE_PATH",
    "build_run_kwargs",
]
