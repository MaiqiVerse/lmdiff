# v0.4.5 verification record

The GPU runs that cleared the task-layer engine port, and the provenance
for the calibration fixture regenerated from them. Progress bars
stripped; nothing else altered.

Scope and reasoning: [`../v045_engine_port_notes.md`](../v045_engine_port_notes.md).

| file | what it is |
|---|---|
| `run2.log` | `scripts/_v045_accuracy_report.py` — geometry against the previous fixture, plus the accuracy tables and the finding diff |
| `run3.log` | `scripts/_v045_legacy_accuracy.py` — the 15 multiple-choice cells scored through the deprecated `InferenceEngine`, against the same cells from the ported path |

**Why these are committed rather than left in a chat log.** The fifteen
cells in `run3.log` have exactly one mechanical reference in the world,
and v0.5.0 deletes it: `run_family_experiment` and `InferenceEngine` are
on that release's removal list. After that, nothing can re-derive them.
A fixture whose justification exists only in a conversation is a fixture
nobody can defend when someone asks in six months where the numbers came
from.

---

## What each run established

**Geometry is bit-identical.** Every field in `run2.log`'s gate block
reports `max|Δ| = 0.000e+00` against the pre-port baseline — not "within
1e-6", identical. `change_vectors`, `cosine_matrix`,
`selective_cosine_matrix`, `magnitudes`, `magnitudes_normalized`,
`share_per_domain`, `magnitudes_per_domain_normalized`, `domain_status`.
The port moved nothing that already existed.

**Findings are byte-identical, plus one structural addition.** The seven
findings the previous fixture produced appear verbatim. The eighth,
`AccuracyArtifactFinding` on math, can only exist once accuracy exists —
`_extract_accuracy_findings` returns `[]` on an empty
`accuracy_by_variant`, and the previous fixture was schema v6 with no
accuracy at all. This is why `test_findings_match` failed on the port
branch: the gate compared a v8 result against a fixture that structurally
predated one of the inputs findings now read.

**Fifteen accuracy cells agree with the path being replaced, exactly.**
`run3.log` scores `hellaswag`, `arc_challenge` and
`mmlu_college_computer_science` for all four variants **and base**
through `InferenceEngine`, and every one of the 15 comes back
`|Δ| = 0.000e+00`. Both paths score multiple-choice by log-likelihood
over the same stored choices from the same `from_lm_eval` probes, and
the scoring function is shared — it was ported in place — so the
comparison isolates exactly the layer that changed.

---

## Fixture provenance: what is verified and what is a pin

`tests/fixtures/calibration_v041_4variant_baseline.json` is the payload
`run2.log` produced. Its cells are **not** all of one kind, and treating
them as if they were is the mistake this section exists to prevent.

### Verified against the deprecated path — 15 cells

| domain | task | engines |
|---|---|---|
| `commonsense` | `hellaswag` | base + 4 variants |
| `reasoning` | `arc_challenge` | base + 4 variants |
| `code` | `mmlu_college_computer_science` | base + 4 variants |

Cross-checked in `run3.log` at `|Δ| = 0.000e+00`. **This check cannot be
repeated after v0.5.0.**

### Regression pins with no external reference

- **`math` (gsm8k) — every cell reads 0.01–0.03.** Generated at the
  run's `max_new_tokens=16`, which is far too short for chain-of-thought
  arithmetic. `AccuracyArtifactFinding` fires on this domain, correctly,
  and that firing is part of what the fixture pins.

  **Do not read gsm8k 0.02 as a measured capability.** It is the number
  this configuration produces, pinned so a change to the pipeline shows
  up as a diff. It is not what Llama-2-7B scores on gsm8k. The
  deprecated path cannot arbitrate it either: it generated at
  `TASK_MAX_NEW_TOKENS["gsm8k"]`, a different budget, so the two paths
  are expected to disagree there and a comparison would prove nothing.

- **`long-context` (longbench_2wikimqa)** — three surviving cells
  (`yarn`, `long`, `code`) are full-coverage measurements on models
  whose windows fit the probes. Not cross-checked: the deprecated engine
  is validity-unaware and would push 9k-token prompts into 4k-window
  models, so running it there produces a number about truncation rather
  than about the model. `base` and `math` are suppressed by the validity
  floor (9 of 100 probes attempted).

- **All geometric fields** — `change_vectors` and everything derived
  from them. Bit-identical to the previous fixture, which is a stronger
  guarantee than a reference: they are the same numbers this project has
  been pinning since v0.4.1, unchanged by the port.

### The honest summary

15 accuracy cells are verified. 5 are pins whose value is that they
change when the code changes, not that they are true. The geometry
carries forward unchanged from v0.4.1. Anyone regenerating this fixture
should preserve that distinction rather than flatten it — the generative
cells being unverifiable is the actual state of affairs, and saying so
is what stops the next reader treating `gsm8k 0.02` as a measurement.
