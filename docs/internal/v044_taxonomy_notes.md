# v0.4.4 — probe taxonomy: investigation notes

Commit 4.3. Specification: PHASE_PLAN §5.1, in service of §5.2 and §5.4.

**Round 3.** Round 1 asked whether §5.1's eight capability names survive
contact with the code — they do not. Round 2 found scoring method is the
real second axis, already present in `Probe.metadata` for lm-eval probes
and absent for v01's. Round 3 looks at `CapabilityRadar`, and your
reading holds: **§5.1 and §5.4 were written to fix it**, the symptom is
"the radar cannot tell how a probe should be scored", and that is Round
2's finding with a name.

Everything below came from a script against shipped code. Scripts in §8.

---

## 0. Settled in earlier rounds

**Recount.** `lmdiff/probes/v01.json` — 90 probes, **three** domains,
30 each (`code`, `knowledge`, `math`), zero undomained. The nine was
`KNOWN_TASK_DOMAINS`, the lm-eval adapter table. `lmdiff/probes/builtin/`
does not exist; §5.2's tree shows a path that was never created.

**Scoring is a real second axis; capability is not.** Over the 31
curated tasks — 19 `multiple_choice`, 10 `generate_until`, 1
`loglikelihood`, 1 `loglikelihood_rolling` — domain predicts
`output_type` 81 % against a 61 % always-guess-MC baseline, and
`output_type` predicts domain 35 %. Five of nine domains split by
`output_type`, `math` and `code` among them. §5.1's eight types all
collapse onto domain.

**§5.2 comes apart three ways.** `ifeval` already serves
instruction-following. `truthfulqa_gen` already serves hallucination and
lmdiff registers only the MC variants. **Nothing in lm-eval's 14,069
tasks measures refusal**, so `safety_regression` is the only one of the
four that is genuinely a probe-writing job. Both writable ones are
instruction-style — the L-001 collision.

**Trap, recorded.** lm-eval's `TaskConfig` default `output_type` is
`generate_until`, not `multiple_choice`. A resolver that assumes MC
mislabels every task whose YAML omits the key, including the
calibration's `longbench_2wikimqa`.

---

## 1. What `CapabilityRadar` does

### 1.1 How it evaluates

```python
CapabilityRadar(probes, evaluator=None, max_new_tokens=16)
    self.evaluator = evaluator or ContainsAnswer()
```

**One evaluator instance, for the whole ProbeSet, chosen by the caller.**
`by_domain()` splits the probes, and every domain gets a `Task` built
with that same `self.evaluator`. Nothing inspects a probe.

Five evaluators exist:

| evaluator | rule | needs |
|---|---|---|
| `ExactMatch` | `output.strip() == expected.strip()` | `expected` |
| `ContainsAnswer` | `expected in output`, case-insensitive | `expected` |
| `MultipleChoice` | parse first `[A-Z]` or integer from output | `metadata["correct_index"]` |
| `F1` | SQuAD token-overlap, ≥ 0.5 | `expected`, optional `metadata["aliases"]` |
| `Gsm8kNumberMatch` | last number after `####`, `math.isclose` | `expected` |

**The CLI exposes three of the five.** `_EVALUATOR_MAP` in `cli.py` has
`exact_match`, `contains_answer`, `multiple_choice`; `F1` and
`Gsm8kNumberMatch` are unreachable from the CLI and appear only in
`experiments/family.py`'s per-task `GENERATE_EVALUATORS` dict. So the
evaluator set is enumerated twice, partially, in two modules — L-035's
shape.

**What the caller has to know:** the answer format of every probe in
every domain *simultaneously*, because there is one slot. For v01 that
means a single rule covering `"42"`, `"Paris"` and `"n"`.

### 1.2 It cannot be done correctly, and getting it wrong is silent

Measured against v01's actual `expected` values:

```
domain        n   len(expected) min/med/max   ≤2 chars
code         30              1 /  4 /  17     12  ['n','i','np','1','3','[','pd','f']
knowledge    30              2 /  5 /  11      5  ['Au','Fe','Na','32','Ag']
math         30              1 /  2 /   4     22  ['42','12','56','8','5','7','50','12']
```

- **`ContainsAnswer`, the default:** scores **3 of 30 code probes
  correct** for the output `"I don't know, but it is interesting to
  consider."` — single-character expecteds like `"n"` and `"i"` are
  substrings of almost anything.
- **`ExactMatch`:** marks **30/30 wrong in every domain** for a
  correct-but-verbose answer (`"42 is the answer."`).
- **`MultipleChoice`:** returns **0.0 for all 90**, because v01 probes
  carry no `metadata` at all.

Every one of those renders as a clean percentage. `Task.run` collapses
`per_probe` into `per_domain = {n, correct, accuracy}`; `CapabilityRadar`
then keeps only `n_probes` and `accuracy` in `DomainRadarResult`,
**discarding `TaskResult.per_probe` entirely**; `print_radar` shows
Domain / N / Acc. The `reason: "missing_mc_metadata"` recorded on all 90
probes reaches no output surface.

This is the L-039 shape one layer down: the pipeline gates a displayed
number on `correct`, and never on whether the evaluator could apply.

### 1.3 What it produces

`RadarResult` — per-domain accuracy for A, and for B when paired, plus
per-domain BD, `bd_healthy`, and degeneracy rates. `summary_table()`
flattens to rows. **Not radar coordinates**; the name is aspirational,
and the only renderer is a table.

It is plumbed into the report layer: `json_report.py` registers
`to_json_dict` for both `RadarResult` and `DomainRadarResult`, and
`terminal.py` has `print_radar`.

One thing it does right, worth preserving: `run_pair` generates once per
engine and passes `pre_generated` into both the evaluator and BD, so
accuracy and distance describe the same samples under sampling decode
(L-010).

### 1.4 Reachability

| | |
|---|---|
| exported from `lmdiff/__init__.py`? | **no** — `Task`, `TaskResult`, `BaseEvaluator`, `EvalResult`, all five evaluators and `loglikelihood_accuracy` **are**; the three radar classes are not |
| `DeprecationWarning`? | **none anywhere in `lmdiff/tasks/`** |
| in-tree callers | exactly one: `ModelDiff.capability_radar()` |
| on v0.5.0's removal list (Z.4 item 4)? | **no** |

`ModelDiff` **is** on the removal list. So as scoped, v0.5.0 deletes
`CapabilityRadar`'s only caller and keeps `CapabilityRadar`.

### 1.5 It cannot run on the live engine

Not "is on the deprecated path" — **structurally incompatible**, on both
engine methods:

| | v0.2.x `InferenceEngine` | live `Engine` / `HFEngine` |
|---|---|---|
| identity | `.model_name` | `.name` |
| generate | `generate(prompts: list[str], n_samples=1, max_new_tokens=64, …)` | `generate(prompt: str, *, max_new_tokens=16, temperature, top_p, top_k, seed)` |
| score | `score(prompts: list[str], continuations: list[str], …)` | `score(prompt: str, continuation: str, *, prefix_text="")` |

`Task.run` calls `engine.generate(self.probes.texts, n_samples=1, …)`
and `engine.model_name`. Driven against a stub with the live surface:

```
LIVE   (HFEngine surface)  Task.run  FAIL  TypeError: generate() got an
                                           unexpected keyword argument 'n_samples'
LEGACY (InferenceEngine)   Task.run  OK    acc=1.0
```

`loglikelihood_accuracy` is bound the same way through `score`.

**So the whole evaluation layer — `Task`, the five evaluators,
`loglikelihood_accuracy`, `CapabilityRadar`, and
`experiments/family.py::_accuracy_for_task` — speaks an engine API that
the v0.4.0 cutover replaced.** `lmdiff/__init__.py` exports most of it
as public API, and none of it runs against the engine
`lmdiff.family()` builds.

### 1.6 Is it the only thing computing evaluation metrics?

Effectively yes. `_accuracy_for_task` is the other, and it is a
*dispatcher over the same five evaluators* rather than an independent
implementation. The five evaluators plus `loglikelihood_accuracy` are
the entire stock, and they are pure functions of `(output, expected,
metadata)` — none touches an engine.

**That changes the cost of restoration, downward.** The scoring logic is
sound, tested and engine-free. What is broken is the ~15 lines that
*call* the engine, in `Task.run` and `loglikelihood_accuracy`. Porting
is an adapter, not a rewrite.

---

## 2. Your reading, checked

**It holds.** Four pieces of evidence:

1. **§5.4 is titled "Probe-aware metric registry"** and its content is
   "`metrics="default"` automatically selects appropriate
   task-type-specific metrics **based on probe set composition**."
   Selecting an evaluator from the probes is precisely the one thing
   `CapabilityRadar.__init__` does not do.
2. **§5.1's stated purpose is grouping**: "Metrics that care about task
   type group by it; metrics that care about domain group by domain."
   The radar groups by domain, once, and has no second grouping.
3. **The original commit order was label-then-dispatch** — §5.5's
   `2.1 ProbeSet supports task_type alongside domain` → `2.4 metric
   registry + auto selection`.
4. **The vintage fits.** §5.5 says that list was written when Phase 2
   was five commits ending at v0.4.0 — i.e. **before the backend
   cutover**, when `CapabilityRadar` and `Task` *were* the live
   evaluation path. The cutover moved the live path to
   `_pipeline`/`HFEngine` and left the task layer on the old API. §5 was
   never revised.

And your inference about §5.4's examples follows. "Hallucination Rate"
and "Safety Regression Rate" do not exist and were not being built.
Whoever wrote §5.4 had no list of available scoring methods to point at:
the five that exist were enumerated in two partial dicts in two modules,
named nowhere as a set, and attached to no probe. **Pointing at
metrics that do not exist is what you do when the ones that do exist are
not addressable.** §5.1's eight names are the same gesture — invented
labels standing in for a vocabulary that was already implicit in the
code and unwritten.

So the diagnosis chain is: *the radar cannot tell how a probe should be
scored* → §5.1 proposes labelling probes → §5.4 proposes dispatching on
the label → and the label it proposes is capability, which §0 shows is
nearly a function of domain and therefore cannot drive the dispatch.
Round 2's `output_type` + `scoring` is the same fix with the axis the
data supports.

---

## 3. What fixing the radar properly looks like

Four steps. They separate cleanly, and only the last is expensive.

### 3.1 Scoring becomes a probe field (the data)

Round 2 §5.3, unchanged:

```python
@dataclass(frozen=True)
class Probe:
    id: str
    text: str
    domain: str | None = None          # subject matter
    output_type: str | None = None     # how the model is queried
    scoring: str | None = None         # how the output is judged
    expected: str | None = None
    metadata: dict = field(default_factory=dict)
```

`output_type` takes lm-eval's four values verbatim. `scoring` names an
evaluator. Both default `None` — unlabelled is not a claim.

### 3.2 One evaluator registry (the lookup)

```python
EVALUATOR_REGISTRY = {
    "exact_match": ExactMatch, "contains_answer": ContainsAnswer,
    "multiple_choice": MultipleChoice, "f1": F1,
    "gsm8k_number_match": Gsm8kNumberMatch,
}
```

This **collapses two existing partial copies** — `cli.py::_EVALUATOR_MAP`
(three entries) and `experiments/family.py::GENERATE_EVALUATORS` (eight
task names → two classes). Each evaluator already carries a `name`
attribute matching these keys, so the registry is derivable rather than
hand-written. Net deletion.

### 3.3 `Task` selects per probe (the fix)

```python
ev = EVALUATOR_REGISTRY.get(probe.scoring) or self.evaluator
correct, score, meta = ev().evaluate(output, probe.expected, probe.metadata)
```

Falls back to the constructor's evaluator when `scoring` is `None`, so
every existing caller keeps its behaviour and old probe sets are
unaffected.

**This is the radar fix, and it lands on the engine the radar already
runs on.** No engine work. v01's math probes get number matching, code
gets something stricter than substring, knowledge gets containment — and
the caller stops having to pick one rule for all three.

One thing to add with it: `TaskResult` should carry the count of probes
whose evaluator reported it could not apply (`reason` present), and
`DomainRadarResult` should surface it. §1.2's silent 0.0 is only silent
because that count is thrown away.

### 3.4 Port `Task` to the `Engine` Protocol (the expensive one)

Loop single-prompt `generate` / `score` instead of calling the batch
API, and read `.name` instead of `.model_name`. Per §1.6 this touches
the call sites only — the evaluators are engine-free.

**This is the accuracy restoration, and it is on v0.5.0's critical
path.** v0.5.0 removes `InferenceEngine`. `Task`, the five evaluators
and `loglikelihood_accuracy` are **exported public API** and cannot run
without it. As scoped, v0.5.0 leaves exported classes that raise on the
only engine the library can build. Either the port lands in or before
v0.5.0, or those exports come out with it. **That is not currently in
Z.4, and it should be.**

---

## 4. Sequencing

| step | commit | why there |
|---|---|---|
| 3.1 fields | **4.3 / v0.4.4** | pure data, no behaviour change, no engine |
| 3.2 registry | **4.3 / v0.4.4** | net deletion of two partial copies; the fields are meaningless without something that reads `scoring` |
| 3.3 `Task` per-probe + unscorable count | **4.3 / v0.4.4** | the only consumer; fallback keeps every caller's behaviour; this is the radar fix and needs no engine work |
| 3.4 engine port | **its own commit, scheduled against v0.5.0** | different problem, different risk, and gated by the `InferenceEngine` removal rather than by the taxonomy |

**3.1–3.3 are one commit's worth of work and they are the same idea:**
put scoring on the probe, name the evaluators once, let the runner look
them up. Splitting them would ship a field nothing reads, then a
registry nothing calls.

**4.4 (§5.2 builtin sets) shrinks.** Two of its four are registry
entries rather than probe files; one needs a refusal set written from
scratch; and per §0 both writable ones are instruction-style, so they
need L-001's separate versioned file. Worth re-scoping before it starts.

**4.6 (§5.4) shrinks more.** Once 3.2 and 3.3 land, the "registry" it
specifies exists. What remains is `metrics="default"` reading probe-set
composition to choose which evaluators to run — genuinely small, and no
longer blocked on inventing task types. It *is* still blocked on 3.4:
until `Task` runs on the live engine, auto-selection has nothing to
select for.

### 4.1 Costing, folded

Round 2 left three costing questions open. All three resolve here:

- **`GeoResult` schema bump.** Yes, 7 → 8, carrying
  `probe_output_types` and `probe_scoring` instead of Round 1's
  `probe_task_types`. Same argument: the seven acceptance lists cost the
  same whenever they move, and unrecorded per-probe labels are
  unrecoverable without a GPU re-run. Two tuples, one bump, no extra
  gates — new nullable fields deserialize ungated via `.get()`, as
  `run_config_yaml` does.
- **Closed enum or open string.** `output_type` closed-with-warning
  against lm-eval's four; **`scoring` open**, because it names an
  evaluator and the registry is the vocabulary. An unknown `scoring`
  falls back to the caller's evaluator, which is the same total lookup
  that makes warn-rather-than-raise correct.
- **Default for unlabelled probes.** `None` for both, unchanged from
  Round 1 §3.3. v01 gets labelled explicitly — and now the label does
  something rather than being decoration.

### 4.2 [QUESTION] — one, and it is yours

**§5.1's `task_type` does not survive this.** Round 2 showed the axis is
nearly a function of domain; Round 3 shows the problem it was written
for is scoring, and that scoring has a better home. My recommendation is
to drop `task_type` from 4.3 entirely and record §5.1 as superseded by
`output_type` + `scoring`.

I am not doing that on my own judgement. §5.1 is the specification, and
three settled decisions in this project were re-litigated by someone
deciding a plan section was wrong. If you want it kept, the cheapest
honest form is a probe-**set** attribute — `ProbeSet.task_type`, author's
own words, no enforced vocabulary, nothing dispatching on it.

**Not a question, but needs a decision from you:** whether the §3.4 port
is v0.5.0 scope or its own release. It is the difference between v0.5.0
shipping a working exported task layer and shipping a dead one.

---

## 5. Revised Part 2

1. `Probe.output_type` / `Probe.scoring`, `str | None = None`, after
   `domain`.
2. `ProbeSet` mirrors — `output_types`, `scorings`, `filter(…)`,
   `by_output_type()`, both JSON sites with the existing
   write-only-if-truthy conditional.
3. `EVALUATOR_REGISTRY` derived from each evaluator's `name`;
   `cli.py::_EVALUATOR_MAP` and `GENERATE_EVALUATORS` deleted in favour
   of it. `KNOWN_OUTPUT_TYPES` closed-with-warning; `scoring` open.
4. `Task` selects per probe with fallback; `TaskResult` gains
   `n_unscorable`; `DomainRadarResult` surfaces it.
5. `from_lm_eval` promotes `output_type` / `native_metric` from
   `metadata` to the fields, leaving the metadata keys one release.
6. `v01.json` — 90 probes labelled `output_type: generate_until` plus a
   per-domain `scoring`. §1.2 says which: `gsm8k_number_match` or
   `exact_match` for math, `contains_answer` for knowledge, and code
   needs a look before choosing, since 12 of its 30 expecteds are ≤ 2
   characters.
7. `GeoResult.probe_output_types` / `probe_scoring`, `geo_schema` 7 → 8:
   eight acceptance lists, two ungated `.get()`s, **payload-level**
   regression test per Z.5.
8. Tests, mutation-checked per assertion. Four that matter: `to_json`
   dropping a new field; the default flipped from `None`; the 7 → 8 bump
   with one gate left at `("6","7")`; and — new, and the one this round
   exists for — `Task` reverted to the single-evaluator path, which must
   fail an assertion that a mixed-scoring ProbeSet produces different
   evaluators per domain.
9. CHANGELOG, `0.4.4` in `pyproject.toml` / `__init__.py` /
   `test_release.py`.
10. `docs/reference/probe-sets.md`, new.

Not in 4.3, carried forward: the engine port (§3.4); the v0.4.3
`ProbeSet`-instance run-config defect, where `reproducible: false` has
found its occupant and `docs/reference/run-config.md` says it has none.

---

## 6. Two things to record regardless of what 4.3 becomes

**Z.4 needs a sixth item.** v0.5.0 removes `InferenceEngine` while
`lmdiff/__init__.py` exports `Task`, `TaskResult`, `BaseEvaluator`,
`EvalResult`, five evaluators and `loglikelihood_accuracy`, all of which
require it. Either port or unexport; silently keeping them is the third
option and it is the bad one.

**`CapabilityRadar`'s absence from the removal list is an omission.**
Its only caller is on the list, it is not exported, it carries no
deprecation warning, and it cannot run on the live engine. Whatever is
decided — port it, remove it, or export it once ported — it should be
decided rather than left.

---

## 7. Scripts

| script | produced |
|---|---|
| `recount.py` | §0 — 90 probes, 3 domains, 30 each |
| `taskmeta.py`, `tc.py` | the `generate_until` default trap |
| `audit_known.py` | lmdiff's 31 vs lm-eval's YAML — 30/31 agree, `mmlu` is a group |
| `search_reg.py`, `targeted2.py` | `ifeval`, `truthfulqa_gen`, `realtoxicityprompts`, no refusal task |
| `crosstab.py` | 5/9 domains split; 81 % / 61 % / 35 % |
| `iface.py` | `HFEngine.name` vs `InferenceEngine.model_name` |
| `radar_live.py` | §1.5 — `Task.run` `TypeError` on the live surface, OK on legacy; §1.2's silent 0/90 |
| `v01_evalfit.py` | §1.2 — expected-length distribution; 3/30 false positives; 30/30 false negatives |

§1.2 and §1.5 are this round's run-it-don't-read-it findings. The
evaluator miscalibration is invisible in the source — it lives in the
interaction between a substring rule and a one-character answer — and
the engine incompatibility looks like a stale type hint until something
calls it.
