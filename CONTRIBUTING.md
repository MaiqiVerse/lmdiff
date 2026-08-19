# Contributing to lmdiff

## Environment

```bash
mamba create -n lmdiff python=3.12
mamba run -n lmdiff pip install -e ".[dev,viz]"
```

Always invoke through `mamba run -n lmdiff <cmd>`. `mamba activate` is
unreliable on Windows, where the shell hook interacts badly with
PowerShell and with non-interactive sessions.

## Running the tests

```bash
mamba run -n lmdiff pytest tests/
```

**The whole tree. Not a subdirectory.** This is what CI runs, and a
narrower run is not a pass. `pyproject.toml` sets
`addopts = "-m 'not slow and not gpu'"`, so model-loading and GPU tests
are deselected by default.

To run the GPU calibration regressions, override the marker filter
explicitly:

```bash
mamba run -n lmdiff pytest tests/integration/test_calibration_regression.py -m "" -v -s
```

New tests go in `tests/unit/` or `tests/integration/`. Nothing belongs at
`tests/` top level — that split once hid half the suite from a run that
reported green (LESSONS L-034).

## After writing a regression test, before trusting it

For each behavioural change the test is meant to protect, revert **that
change alone** — the predicate, the comparison, the filter call — and
confirm the test fails. If it still passes, the test is not testing what
you think it is.

**Revert the smallest edit, not the commit.** Commit-level reverts
conflict in stacked work and over-revert.

**Run the whole suite, not the file you just wrote.** Coverage often
lives elsewhere, and a single-file run reports "not caught" when the
assertion that would have caught it is in another module.

Mandatory when the path has more than one protective layer — that is
exactly when a test goes green on the wrong one.

Cost is minutes, and only for the fixes the new test claims to cover.

### Why this is a checklist step and not a tool

This was tested rather than assumed. The obvious automation —
`git revert --no-commit <sha>` per fix commit, then run the suite —
failed on three of five commits in the v0.4.2 PR:

| commit | result |
|---|---|
| shared validity helper | **conflicted** |
| z-score aggregation | **conflicted** |
| html crash + drift tables | caught |
| shared unit labels | caught |
| `change_size` predicate | **false negative** |

Two failure modes, both fatal to automation.

**Reverting an early commit in a stack conflicts, because later commits
touch the same lines — so the better a PR is sequenced, the worse
commit-level reverts work.** The two that conflicted were the two
prerequisites everything else built on. Good practice in one dimension
defeats the tooling in another; this is structural, not bad luck, and it
is the reason to stop looking for a `git`-level shortcut.

**The false negative was a scope error wearing a different hat** — the
same shape as L-034. `change_size`'s claim-gating assertions live in a
different test file from the one being run, so a single-suite run
reports "not caught" whenever the coverage lives elsewhere. Hence "run
the whole suite" above.

Generic mutation harnesses (`mutmut`, `cosmic-ray`) are worse here for a
different reason: operator-level mutants are mostly semantically
irrelevant to this codebase and cost orders of magnitude more time for a
weaker signal. The value came from choosing *meaningful* mutations,
which is a thinking step, not a tool.

See LESSONS L-040 for the incident this came from.

## Before opening a PR

- `pytest tests/` passes.
- For any metric, schema, or cross-cutting change: a design audit under
  `docs/internal/` first, with its open questions resolved explicitly.
  See `docs/internal/v041_validity_design.md` for the expected shape.
- If the change touches a formula, a threshold, or what a report says:
  render the affected output and read it. Every figure individually, not
  `figures()` as a unit. Three of the four defects in v0.4.2 produced
  output that was wrong rather than absent, and no assertion on a
  computed value could see them (L-038).
- Thresholds and formulas get their reasoning written at the definition,
  not inferred later from the value.

## Conventions

Python 3.10+, `X | Y` unions, type hints on public functions, f-strings,
`rich` for terminal colour, explicit imports. Match the density and idiom
of the surrounding code.

`CLAUDE.md` is the short orientation file; `docs/internal/PHASE_PLAN_v6.md`
is the design authority; `LESSONS.md` is the incident log. Grep the last
of those before debugging anything that feels familiar.
