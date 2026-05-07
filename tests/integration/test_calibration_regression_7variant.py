"""7-variant calibration regression — the wider gate for v0.4.0 cutover.

The 4-variant calibration (``test_calibration_regression.py``) only
covers variants without ``system_prompt`` and with greedy decoding —
the easy path. This test covers the two variants that exercise the
v0.4.0 features the 4-variant test missed:

  - ``system_prompt`` — runtime-only Config modification, exercises
    ``HFEngine.score(prefix_text=…)`` and the split-tokenize path
    that Fix 2 added (PR #15 fixup commits)
  - ``temp_1.5`` — sample decoding with explicit ``top_k=0``,
    exercises Fix 1's top_k passthrough

Plus the 5 unique-model variants (yarn/long/code/math/chat) for
spot-checking they still match v0.3.2.

Tolerance rationale:
  - ``change_vectors`` (per-probe δ): 1e-6 byte-equivalence for the
    6 deterministic variants; SKIPPED for ``temp_1.5`` because sample
    decoding consumes RNG state in ways that depend on prior call
    order and aren't reproducible across code-path changes
  - ``share_per_domain``: 2pp for ALL variants. This is the user-
    visible headline metric (the showcase percentages); a 2pp
    tolerance catches the 60→94% (system_prompt) and 34→5%
    (temp_1.5) regressions that the GPU 7-variant demo surfaced
  - ``magnitudes_per_domain_normalized``: 1e-3 (slightly looser than
    1e-6 because temp_1.5's pdn participates in the sum-of-squares
    normalization for share, and a single sampled-variant value
    shouldn't fail this metric for the 6 deterministic ones)

If any assertion fails, the cutover did not safely preserve v0.3.2
behavior on the broader set of variants. v0.4.0 doesn't ship.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

SUMMARY_PATH = Path(__file__).parent.parent / "fixtures" / "calibration_v032_7variant_summary.json"

# Tolerances per metric.
TOL_CHANGE_VECTORS = 1e-6
TOL_SHARE_PCT_POINTS = 2.0       # 2pp, user-spec
TOL_PDN = 1e-3
TOL_OVERALL_NORM = 1e-3

# Variants whose decode is sampling-based — these can't be byte-checked
# across runs because torch RNG state depends on full call history.
SAMPLE_DECODE_VARIANTS = {"temp_1.5"}


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not SUMMARY_PATH.exists():
        pytest.skip(
            f"7-variant summary not present at {SUMMARY_PATH}. "
            "Generate via _make_7variant_fixture.py from the "
            "demo_032_rerendered tarball and commit before running."
        )
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cutover_result() -> dict:
    """Run the 7-variant family() through the new HFEngine pipeline
    and return the to_json_dict payload for comparison."""
    import lmdiff
    from lmdiff import Config, DecodeSpec
    from lmdiff.report.json_report import to_json_dict

    result = lmdiff.family(
        base="meta-llama/Llama-2-7b-hf",
        variants={
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
        probes="lm_eval:hellaswag+arc_challenge+gsm8k+mmlu_college_computer_science+longbench_2wikimqa",
        n_probes=100,
        max_new_tokens=16,
        task_overrides={
            "gsm8k": {"max_new_tokens": 256},
            "longbench_2wikimqa": {"max_new_tokens": 128},
        },
        seed=42,
    )
    payload = to_json_dict(result)
    payload.pop("generated_at", None)
    return payload


# ── Structural / probe-distribution match ───────────────────────────


def test_variant_names_match(baseline, cutover_result):
    assert baseline["variant_names"] == cutover_result["variant_names"]


def test_n_probes_match(baseline, cutover_result):
    assert baseline["n_probes"] == cutover_result["n_probes"]


def test_probe_domain_distribution_match(baseline, cutover_result):
    """Per-domain probe count must be identical (per-task n_probes
    semantics from PR #8)."""
    from collections import Counter
    assert Counter(baseline["probe_domains"]) == Counter(
        cutover_result["probe_domains"],
    )


# ── Per-variant change_vectors (greedy variants only) ───────────────


@pytest.mark.parametrize(
    "variant",
    ["yarn", "long", "code", "math", "chat", "system_prompt"],
)
def test_change_vectors_match_for_deterministic_variants(
    baseline, cutover_result, variant,
):
    """Greedy + system_prompt variants must reproduce per-probe δ
    values exactly. system_prompt is the test that Fix 2's prefix_text
    threading actually reaches the engine — without Fix 2, the GPU
    demo showed 94% commonsense vs v0.3.2's 60%."""
    bvec = baseline["change_vectors"][variant]
    cvec = cutover_result["change_vectors"][variant]
    assert len(bvec) == len(cvec), variant
    for i, (b, c) in enumerate(zip(bvec, cvec)):
        assert abs(b - c) < TOL_CHANGE_VECTORS, (
            f"change_vectors[{variant}][{i}]: baseline={b}, cutover={c}, "
            f"diff={abs(b-c)}"
        )


# ── share_per_domain (ALL variants, 2pp tolerance) ──────────────────


@pytest.mark.parametrize(
    "variant", [
        "yarn", "long", "code", "math", "chat",
        "temp_1.5", "system_prompt",
    ],
)
def test_share_per_domain_within_2pp(baseline, cutover_result, variant):
    """The headline showcase metric. 2pp tolerance is per the spec —
    catches the 60→94% (system_prompt) and 34→5% (temp_1.5) regressions
    that Fix 1 + Fix 2 address. Tighter than any natural cross-run
    variation but loose enough to accommodate sampling jitter for
    temp_1.5."""
    b_row = baseline["share_per_domain"][variant]
    c_row = cutover_result["share_per_domain"][variant]
    assert set(b_row.keys()) == set(c_row.keys()), variant
    for d in b_row:
        b_pct = b_row[d] * 100.0
        c_pct = c_row[d] * 100.0
        diff_pp = abs(b_pct - c_pct)
        assert diff_pp <= TOL_SHARE_PCT_POINTS, (
            f"share_per_domain[{variant}][{d}]: baseline={b_pct:.2f}%, "
            f"cutover={c_pct:.2f}%, diff={diff_pp:.2f}pp "
            f"(tolerance: {TOL_SHARE_PCT_POINTS}pp)"
        )


# ── magnitudes_per_domain_normalized (deterministic variants) ───────


@pytest.mark.parametrize(
    "variant",
    ["yarn", "long", "code", "math", "chat", "system_prompt"],
)
def test_pdn_match_for_deterministic_variants(
    baseline, cutover_result, variant,
):
    b_pdn = baseline["magnitudes_per_domain_normalized"][variant]
    c_pdn = cutover_result["magnitudes_per_domain_normalized"][variant]
    assert set(b_pdn.keys()) == set(c_pdn.keys()), variant
    for d in b_pdn:
        b, c = b_pdn[d], c_pdn[d]
        assert abs(b - c) < TOL_PDN, (
            f"pdn[{variant}][{d}]: baseline={b}, cutover={c}, "
            f"diff={abs(b-c)} (tolerance: {TOL_PDN})"
        )


# ── magnitudes_normalized (overall, deterministic variants) ─────────


@pytest.mark.parametrize(
    "variant",
    ["yarn", "long", "code", "math", "chat", "system_prompt"],
)
def test_overall_normalized_for_deterministic_variants(
    baseline, cutover_result, variant,
):
    b = baseline["magnitudes_normalized"][variant]
    c = cutover_result["magnitudes_normalized"][variant]
    assert abs(b - c) < TOL_OVERALL_NORM, (
        f"magnitudes_normalized[{variant}]: baseline={b}, "
        f"cutover={c}, diff={abs(b-c)} (tolerance: {TOL_OVERALL_NORM})"
    )
