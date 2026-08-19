# CLAUDE.md

Orientation file. Read it fully — it is short on purpose. Its job is to
stop you operating on a stale mental model and point you at the two
documents that hold the detail:

- **`docs/internal/PHASE_PLAN_v6.md`** — design authority. Phase scope,
  release plan, and the Update sections (1–7) that record every design
  decision and why. If a design question is settled anywhere, it is
  settled there.
- **`LESSONS.md`** — L-001 through L-037. Accumulated engineering
  lessons, each written after something cost real time. Grep it before
  debugging anything that feels familiar.

## Development environment

- Python: **always** `mamba run -n lmdiff <cmd>`, never `mamba activate`.
  Activation is unreliable on Windows (shell hook vs PowerShell and
  non-interactive sessions).
- GPU: RTX 5090 (Blackwell, sm_120). PyTorch cu130.
- Git: push to origin. Branch, PR, wait for CI, merge.
- Inline `python -c "…"` frequently breaks on this shell's quoting.
  Write a script to the scratchpad and run that instead.

## Verification scope

**`pytest tests/` — the whole tree. Not a subdirectory.**

CI runs `pytest tests/`. Anything narrower is not a pass, and saying
"tests pass" on a narrower run is a false report. This cost a release
once: `pytest tests/unit/` was used as the gate seven times while half
the suite went uncollected (L-034). A pass count with no denominator
looks identical whether it covers 100 % or 51 %.

Current: **1075 pass, 8 skipped** (1288 collected including gpu/slow,
which are deselected by default via `addopts`).

GPU calibration tests need an explicit marker override:
`pytest tests/integration/test_calibration_regression.py -m ""`.

## What this is

`lmdiff` compares language model **configurations** — weights + context
+ decoding + adapter + scaffold — not just models. The novelty is not
the distance metric (that is established literature); it is change
geometry (behavioural change as vectors), the configuration
abstraction, and the engineering.

## Current state

- **v0.4.2 shipped** (2026-08-19, on PyPI as `lmdiff-kit`).
- **Phase 2 in progress.** Next: **commit 4.2 — run-config schema and
  emission** (PHASE_PLAN Update 7 AA). Design audit first.
- v0.5.0 is a cross-cutting release, not a phase: v0.2.x removals plus
  variant-only measurement. See PHASE_PLAN Z.4.

## Architecture

**The live path.** `lmdiff.compare()` / `lmdiff.family()` in `_api.py`
→ `_pipeline.run_family_pipeline` → `_engine.HFEngine`. Everything
user-facing goes through this.

| module | role |
|---|---|
| `_api.py` | public entry points |
| `_pipeline.py` | the family pipeline — per-probe loop, prompt assembly, engine lifetime |
| `_engine.py` | `Engine` Protocol + `HFEngine`. Lazy-imports torch/transformers |
| `_config.py` | `Config` and sub-specs (v0.3.0+) |
| `_validity.py` | per-probe validity, `compute_domain_status`, the shared `filter_measured_cells` predicate, and the `PDN_*` label vocabulary |
| `_findings.py` | narrative findings, one definition consumed by every renderer |
| `geometry.py` | **`GeoResult`** — the central result type. Also holds the deprecated `ChangeGeometry` |
| `report/`, `viz/` | renderers. Consume `GeoResult`, never engines |

**v0.2.x legacy, `DeprecationWarning` since v0.4.0, removed in v0.5.0:**
`ModelDiff` (`diff.py`), `InferenceEngine` (`engine.py`),
`ChangeGeometry` (`geometry.py`), `lmdiff.config.Config` (`config.py`),
`run_family_experiment` (`experiments/family.py`), and the
`normalization_effect` figure. Do not build on these. Note `geometry.py`
is **not** wholly legacy — `GeoResult` lives there and is current.

## Rules that do not bend

- **Nothing outside the engine layer imports `torch` or `transformers`.**
  Metrics, tasks, reports, and viz receive engine outputs, never model
  objects.
- **Metrics are zero-coupled.** No metric imports another.
- **`ProbeSet` is immutable after loading.**
- **One definition per user-facing quantity.** Duplicated formulas and
  labels have drifted three times now (L-035). If a predicate or a
  description exists in two places, that is the bug.
- **Excluded cells are removed, not nulled.** Consumers that aggregate
  must exclude them by construction — a display-time skip leaves the
  statistic wrong (the v0.4.2 specialization z-score).

## Working patterns that earned their place

**Design audit before implementation** — for any metric, schema, or
cross-cutting change. Write the audit, resolve the open questions
explicitly, get them approved, then implement. v0.4.1 and v0.4.2 both
worked this way. The failure it prevents: a formula that is
self-consistent with its own spec and measures nothing (L-033).

**Mutation-check the regression test** — after writing a test that
protects a fix, revert *that fix alone* (the predicate, the comparison,
the filter call — not the commit; commit-level reverts conflict in
stacked work) and confirm the test fails. If it still passes, the test
is not testing what you think. Mandatory when a path has more than one
guard, since that is exactly when a test goes green on the wrong one
— which happened in v0.4.2 and was invisible to any amount of running
the test.

**Stop and ask rather than expand scope.** If you hit a design decision
the plan does not cover, say so and stop. Prefix it `[QUESTION]`.
Do not resolve it by picking something reasonable — several settled
decisions in this project were re-litigated that way, and threshold
choices in particular need their reasoning written at the definition,
not inferred later.

**Verify, do not assert.** "Tests pass", "behaviour preserved", "the
figure is fine" are claims that need evidence. Render the figure and
look at it. Snapshot before and after and diff. The v0.4.1 HTML report
was broken on arrival because nothing in the release process rendered
one.

## Conventions

Python 3.10+, `X | Y` unions, type hints on public functions, f-strings,
`rich` for terminal colour. Explicit imports, no wildcards. Tests use
`pytest`; new tests go in `tests/unit/` or `tests/integration/`, never
at `tests/` top level.

Probes are **completion-style** for base models — `"The capital of
France is "`, not `"What is the capital of France? Answer in one word."`
Instruction-style probes belong in a separate versioned file (L-001).

Behavioural distance, change vectors, and the per-domain normalization
formula are derived in `docs/methodology/normalization.md`. Read that
before touching a formula.

## When to update LESSONS.md

After any debug session over ~15 minutes, any finding where the data
looked one way but was another, or any design decision future-you would
question. **Propose the entry; do not write it unasked** — the user
decides what is worth preserving.
