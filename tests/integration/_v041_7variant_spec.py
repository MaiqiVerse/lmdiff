"""Shared spec for the v0.4.0 7-variant calibration baseline.

Single source of truth for the family() call that
``test_calibration_regression_7variant.py`` runs and that
``scripts/_regenerate_v040_7variant_fixture.py`` regenerates against.
Importing the same constant from both places eliminates "did I run
the same call as the test?" risk.

Why a separate module instead of conftest:
  - ``conftest.py`` is pytest-only; the regen script lives under
    ``scripts/`` and runs as a plain Python program. It needs to
    ``import`` the spec without bringing in pytest fixtures.
  - Constants here are pure Python data; no fixtures, no pytest hooks.

Module is private (``_`` prefix) — not part of any public API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Resolve once at import time; both test and regen script can read it.
_REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "calibration_v041_7variant_summary.json"
# v0.4.1: pdn formula change + validity framework shifts every value;
# fixture regenerated on GPU via scripts/_regenerate_v041_7variant_fixture.py
# (commit 8 of v0.4.1 PR). Until then the calibration test auto-skips.


def build_run_kwargs() -> dict[str, Any]:
    """Build the exact ``family()`` kwargs.

    Returned as a *fresh* dict each call so callers can mutate without
    aliasing (the ``Config`` instances inside ``variants`` are
    re-created each call too — a Config is mutable in principle, even
    though the test never mutates it).

    Bound at call time rather than module-level so importing this
    module doesn't pull in ``lmdiff`` (which pulls in ``transformers``
    via the engine layer). Importable on a CPU-only box for spec
    inspection / dry-run.
    """
    from lmdiff import Config, DecodeSpec

    return {
        "base": "meta-llama/Llama-2-7b-hf",
        "variants": {
            "yarn":          "NousResearch/Yarn-Llama-2-7b-128k",
            "long":          "togethercomputer/LLaMA-2-7B-32K",
            "code":          "codellama/CodeLlama-7b-hf",
            "math":          "EleutherAI/llemma_7b",
            "chat":          "meta-llama/Llama-2-7b-chat-hf",
            "temp_1.5":      Config(
                model="meta-llama/Llama-2-7b-hf",
                decode=DecodeSpec(strategy="sample", temperature=1.5),
                name="temp_1.5",
            ),
            "system_prompt": Config(
                model="meta-llama/Llama-2-7b-hf",
                system_prompt="You are concise.",
                name="system_prompt",
            ),
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


# All 7 variant names in the order ``family()`` returns them — used
# to drive parametrize in the calibration test.
ALL_VARIANTS: tuple[str, ...] = (
    "yarn", "long", "code", "math", "chat", "temp_1.5", "system_prompt",
)

# Variants whose decode is sampling-based and whose byte-equivalence
# can NOT be asserted across runs. Empty under Fix 3 — ``temp_1.5`` is
# reproducible given a pinned seed (``family(seed=42)`` plus the
# Fix 3 once-per-variant ``manual_seed`` at probe 0). Kept as a
# named constant so future variants that genuinely can't be byte-
# checked (e.g. best_of_n with hardware-non-deterministic argmax ties)
# can be added without rewriting the test parametrize list.
SAMPLE_DECODE_VARIANTS_LEGACY: frozenset[str] = frozenset()


# Byte-checkable subset = full variant set minus the legacy unstable
# ones. Under Fix 3 this is all 7. The test asserts ``change_vectors``
# byte-equivalence within 1e-6 for every variant in this set against
# the fixture. The full ``ALL_VARIANTS`` set is used for the looser
# 2pp ``share_per_domain`` assertion.
BYTE_EQUIVALENT_VARIANTS: tuple[str, ...] = tuple(
    v for v in ALL_VARIANTS if v not in SAMPLE_DECODE_VARIANTS_LEGACY
)

# Variants using sample decoding. Seed-pinned and therefore reproducible
# (hence still in ``BYTE_EQUIVALENT_VARIANTS``), but their *derived*
# per-domain metrics keep the looser tolerances: share within 2pp rather
# than 1e-6, pdn within 1e-3 rather than 1e-6. Sampling amplifies any
# residual float jitter through the δ → pdn → share chain far more than
# it shows up in raw ``change_vectors``.
SAMPLE_DECODE_VARIANTS: frozenset[str] = frozenset({"temp_1.5"})


# ── Per-(variant, domain) validity status (v0.4.1) ──────────────────
#
# Transcribed from the regenerated fixture, which is the contract. Also
# asserted directly (``test_fixture_domain_status_matches_spec``) so a
# fixture regenerated with different validity behaviour fails loudly on
# CPU instead of silently reshaping the numeric assertions below.
#
# long-context is the only non-``full`` column: 91 of 100
# ``longbench_2wikimqa`` probes exceed Llama-2-7B's 4096-token window,
# leaving 9 valid-for-both — below the ``min_valid_fraction`` floor of
# 0.5, so the domain reports no share. It resolves to ``variant_only``
# where the variant's own window still covers ≥50 % of the domain
# (yarn 128K → 100/100, long 32K → 100/100, code 16K → 89/100) and to
# ``out_of_range`` otherwise (math / chat are 4K models; temp_1.5 and
# system_prompt are runtime-only variants of the 4K base).

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
    "chat": {
        "code": "full",
        "commonsense": "full",
        "long-context": "out_of_range",
        "math": "full",
        "reasoning": "full",
    },
    "temp_1.5": {
        "code": "full",
        "commonsense": "full",
        "long-context": "out_of_range",
        "math": "full",
        "reasoning": "full",
    },
    "system_prompt": {
        "code": "full",
        "commonsense": "full",
        "long-context": "out_of_range",
        "math": "full",
        "reasoning": "full",
    },
}

#: ``full`` / ``partial`` cells — numeric share and pdn, asserted
#: against the fixture within tolerance. 28 cells.
MEASURED_CELLS: tuple[tuple[str, str], ...] = tuple(
    (v, d)
    for v in ALL_VARIANTS
    for d in DOMAINS
    if EXPECTED_DOMAIN_STATUS[v][d] in ("full", "partial")
)

#: ``variant_only`` / ``out_of_range`` cells — share and pdn must both
#: be ``None`` (the "not measured" sentinel, distinct from "measured
#: zero drift"). 7 cells, all long-context.
UNMEASURED_CELLS: tuple[tuple[str, str], ...] = tuple(
    (v, d)
    for v in ALL_VARIANTS
    for d in DOMAINS
    if EXPECTED_DOMAIN_STATUS[v][d] in ("variant_only", "out_of_range")
)


__all__ = [
    "ALL_VARIANTS",
    "BYTE_EQUIVALENT_VARIANTS",
    "DOMAINS",
    "EXPECTED_DOMAIN_STATUS",
    "FIXTURE_PATH",
    "MEASURED_CELLS",
    "SAMPLE_DECODE_VARIANTS",
    "SAMPLE_DECODE_VARIANTS_LEGACY",
    "UNMEASURED_CELLS",
    "build_run_kwargs",
]
