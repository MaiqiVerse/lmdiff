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

**Choose the mutation per assertion, not per fix.** One fix often needs
several, because different assertions in the same file defend against
different failures. "Revert the fix and confirm the tests fail" is too
coarse wherever a test guards against *overshoot* rather than absence —
such a test passes under the revert and fails only under a mutation that
produces the overshoot.

Two worked examples, both from v0.4.3:

- `_values_equal` gaining a dataclass branch. Reverting the branch
  catches the tests asserting the fix is present, but **not** the
  type guard — without the branch the fallback `a == b` already returns
  `False` for mismatched types, so it passes either way. That guard
  defends against an *over-eager* branch, so it needs the mutation that
  produces one: removing the branch's type check.
- `min_valid_fraction` passthrough. Three mutations — dropping it,
  hardcoding the default, and `x or DEFAULT` coalescing. Only the
  `0.0` test catches the third, because `0.0` is a legitimate value that
  a falsy-coalesce silently replaces.

A test that keeps passing under the mutation you assumed relevant is
testing something other than what you assumed. Some tests are
mutation-invariant by design — pure regression guards for behaviour that
holds before *and* after — and those are fine, but say so rather than
leaving them looking like failures.

Mandatory when the path has more than one protective layer — that is
exactly when a test goes green on the wrong one.

Cost is minutes, and only for the fixes the new test claims to cover.

### The mutation check can fail silently, in the shape it exists to catch

Three ways a mutation run reports nothing wrong while having verified
nothing. All three were hit in one v0.4.5 session.

**Verify every anchor still matches, before running.** A mutation whose
`old` string no longer appears in the source **skips**, and a skip reads
as coverage at a glance — it sits in the same column as a pass. This
happens exactly when you are most exposed: you rewrote the function, so
the anchor moved, so the mutation aimed at your new code silently
stopped aiming at anything. Parse the mutation list and assert every
anchor matches before spending the twenty minutes.

**Check the mutation actually changed the behaviour, not just the
bytes.** `base_accuracy = {} or _accuracy_by_domain(...)` reads like a
disable and is a no-op: `{}` is falsy, so `or` returns the call. It
reported **NOT CAUGHT** against a test that was working perfectly. When
a mutation comes back not-caught, re-read the mutation before you
re-read the test — a bad mutation and a missing test are indistinguishable
from the summary line.

**Run on a quiescent tree.** The harness captures each file, mutates it,
and restores it in a `finally`. Editing the same files during a run
makes every result unreliable — the restore writes back a snapshot from
before your edit — and a run killed mid-mutation leaves the mutation
applied in the working tree. Both happened; the second one was
self-inflicted twice, once by editing during a run and once by a helper
script that did `from mutate_45 import MUTATIONS`, which *executes* the
module.

The leftover mutation was caught by the suite within one command, which
is the system working — but only because the two tests that pin it
existed. That is the argument for writing them, not a reason to relax
about the hygiene.

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
