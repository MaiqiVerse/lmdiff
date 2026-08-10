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


# ── Per-(variant, domain) validity status (v0.4.1) ──────────────────
#
# Transcribed from the regenerated fixture, which is the contract. Also
# asserted directly (``test_fixture_domain_status_matches_spec``) so a
# fixture regenerated with different validity behaviour fails loudly on
# CPU instead of silently reshaping the numeric assertions.
#
# Identical to the 7-variant spec's rows for these four variants — same
# probe set, same base model, so the same 9 of 100 long-context probes
# survive. The two regenerated fixtures agree to 0.0 on every shared
# change_vectors / share / pdn cell.
#
# long-context is the only non-``full`` column: 91 of 100
# ``longbench_2wikimqa`` probes exceed Llama-2-7B's 4096-token window,
# leaving 9 valid-for-both — below the ``min_valid_fraction`` floor of
# 0.5, so the domain reports no share. ``variant_only`` where the
# variant's own window covers ≥50 % of the domain (yarn 128K, long 32K,
# code 16K → 89/100), ``out_of_range`` for math (llemma is a 4K model).

DOMAINS: tuple[str, ...] = (
    "code", "commonsense", "long-context", "math", "reasoning",
)

EXPECTED_DOMAIN_STATUS: dict[str, dict[str, str]] = {
    "yarn": {
        "code": "full",
        "commonsense": "full",
        "long-context": "variant_only",
        "math": "full",
        "reasoning": "full",
    },
    "long": {
        "code": "full",
        "commonsense": "full",
        "long-context": "variant_only",
        "math": "full",
        "reasoning": "full",
    },
    "code": {
        "code": "full",
        "commonsense": "full",
        "long-context": "variant_only",
        "math": "full",
        "reasoning": "full",
    },
    "math": {
        "code": "full",
        "commonsense": "full",
        "long-context": "out_of_range",
        "math": "full",
        "reasoning": "full",
    },
}

#: ``full`` / ``partial`` cells — numeric share and pdn, asserted
#: against the fixture within 1e-6. 16 cells.
MEASURED_CELLS: tuple[tuple[str, str], ...] = tuple(
    (v, d)
    for v in ALL_VARIANTS
    for d in DOMAINS
    if EXPECTED_DOMAIN_STATUS[v][d] in ("full", "partial")
)

#: ``variant_only`` / ``out_of_range`` cells — share and pdn must both
#: be ``None``. 4 cells, all long-context.
UNMEASURED_CELLS: tuple[tuple[str, str], ...] = tuple(
    (v, d)
    for v in ALL_VARIANTS
    for d in DOMAINS
    if EXPECTED_DOMAIN_STATUS[v][d] in ("variant_only", "out_of_range")
)

# Every variant here is a distinct set of weights decoded greedily —
# no sample-decode variant in this set, so no looser tolerance tier is
# needed (contrast ``_v041_7variant_spec.SAMPLE_DECODE_VARIANTS``).


__all__ = [
    "ALL_VARIANTS",
    "DOMAINS",
    "EXPECTED_DOMAIN_STATUS",
    "FIXTURE_PATH",
    "MEASURED_CELLS",
    "UNMEASURED_CELLS",
    "build_run_kwargs",
]
