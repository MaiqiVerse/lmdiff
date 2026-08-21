# v0.4.4 — probe taxonomy: investigation notes

Commit 4.3. Specification: PHASE_PLAN §5.1, in service of §5.2 and §5.4.

Not a design audit — `task_type` is additive, with no numeric effect and
no new serialization path.

**Round 2.** Round 1 asked whether §5.1's eight capability-named types
survive contact with the code. They do not, and the reason Round 1 gave
was the wrong one. This round starts from scoring method and the
two-tier probe-set split, and lets the taxonomy come from the table.

Everything below came from a script against shipped code. Scripts in §7.

---

## 0. Recount

**v01 has three domains, 30 probes each, 90 total.** Counted straight
from the file:

```
lmdiff/probes/v01.json   name='v01'  version='0.2.1'  90 probes
  code         30    code_001 .. code_030
  knowledge    30    knowledge_001 .. knowledge_030
  math         30    math_001 .. math_030
probes with no domain: 0
```

The nine was `KNOWN_TASK_DOMAINS`, lmdiff's lm-eval adapter table — nine
distinct domains across 31 curated tasks. Round 1 §2.1 said "90 probes,
three domains, 30 each" and §2.4(a) then led with "There are nine
domains, not five", which reads as a claim about v01 three paragraphs
after the correct count. The numbers were right; putting the adapter's
count under a heading that looked like a v01 correction was not.

Two path notes while recounting. §5.2's tree shows
`lmdiff/probes/builtin/v01.json`; the real path is
`lmdiff/probes/v01.json` and **`builtin/` does not exist**. And v01's
probes carry no `metadata` at all — which matters in §5.

---

## 1. The table

Tier, domain, how the model is queried, how the output is judged, and
who decides. `output_type` is lm-eval's; `evaluator` is what lmdiff
actually runs.

### 1.1 Tier 1 — `v01`, the fast set

| domain | n | query | judged by | who decides |
|---|---|---|---|---|
| `code` | 30 | plain completion | one evaluator, caller's choice | `--evaluator`, default `contains_answer` |
| `knowledge` | 30 | plain completion | " | " |
| `math` | 30 | plain completion | " | " |

**One evaluator for all 90 probes.** `CapabilityRadar` takes a single
`evaluator=` and applies it to every domain; the CLI exposes three by
name. Nothing in v01 records how a probe should be scored, because the
answer is not a property of the probe here — it is an argument.

### 1.2 Tier 2 — the lm-eval calibration set

| task | domain | `output_type` | lm-eval metrics | lmdiff evaluator |
|---|---|---|---|---|
| `hellaswag` | commonsense | `multiple_choice` | `acc`, `acc_norm` | `loglikelihood_accuracy` |
| `arc_challenge` | reasoning | `multiple_choice` | `acc`, `acc_norm` | `loglikelihood_accuracy` |
| `gsm8k` | math | `generate_until` | `exact_match` | `Gsm8kNumberMatch` |
| `mmlu_college_computer_science` | code | `multiple_choice` | `acc` | `loglikelihood_accuracy` |
| `longbench_2wikimqa` | long-context | `generate_until` | `score`, `qa_f1_score` | `F1` |

### 1.3 Tier 2, widened to all 31 curated tasks

`output_type` distribution: **19 `multiple_choice`, 10 `generate_until`,
1 `loglikelihood`, 1 `loglikelihood_rolling`**.

| domain | tasks | `output_type` split |
|---|---|---|
| `code` | `mmlu_college_computer_science`, `mmlu_computer_security`, `mmlu_machine_learning`, `humaneval`, `mbpp` | 3 MC / 2 gen |
| `commonsense` | `hellaswag`, `piqa`, `winogrande`, `openbookqa`, `commonsense_qa` | 5 MC |
| `knowledge` | `mmlu`, `triviaqa`, `nq_open` | 1 MC / 2 gen |
| `language` | `lambada_openai`, `wikitext` | 1 ll / 1 ll-rolling |
| `long-context` | four `longbench_*` | 4 gen |
| `math` | `mathqa`, `mmlu_college_mathematics`, `mmlu_high_school_mathematics`, `gsm8k` | 3 MC / 1 gen |
| `reading` | `boolq`, `squadv2` | 1 MC / 1 gen |
| `reasoning` | `arc_challenge`, `arc_easy`, `logiqa` | 3 MC |
| `safety` | `truthfulqa_mc1`, `truthfulqa_mc2`, `toxigen` | 3 MC |

**lmdiff's table is accurate.** Resolved every entry against lm-eval's
own YAML: 30 of 31 agree exactly. The one disagreement is `mmlu`, which
is a registry **group**, not a leaf — it has no `output_type` of its
own, and lmdiff's note already says "prefer specific subsets".

> Caught a trap worth recording: **lm-eval's `TaskConfig` default
> `output_type` is `generate_until`, not `multiple_choice`.** My first
> resolver assumed MC and mislabelled every task whose YAML omits the
> key — all four `longbench_*` and `squadv2`. Verified directly:
> `TaskConfig(task='x').output_type == 'generate_until'`. Had I not
> checked, this document would have reported the calibration's
> long-context task as multiple-choice.

---

## 2. Q1 — what serves the default set's domains

Answered by §1.3. Three things it shows that the calibration set alone
does not:

**Every calibration domain has alternates at a different `output_type`
except two.** `math` can be served by `gsm8k` (generation, exact_match)
or by `mathqa` / `mmlu_*_mathematics` (loglikelihood over choices).
`code` likewise. `commonsense`, `reasoning` and `long-context` are
single-shape — the first two all-MC, the third all-generation.

**Two `output_type`s are registered but score nothing.** `lambada_openai`
and `wikitext` are the only `loglikelihood` / `loglikelihood_rolling`
entries, and `_accuracy_for_task` has no branch for either — both fall
through to `return float("nan")`. The same is true for `humaneval` and
`mbpp` via `requires_execution`. **Four of the 31 curated tasks cannot
produce an accuracy number today.**

**The calibration set is one task per domain, and that is a sampling
choice, not a constraint.** Nothing stops a run using three math tasks
at two `output_type`s — which is exactly the case §5 has to answer for.

---

## 3. Q2 — what lm-eval offers for the domains §5.2 wants

Searched the full 14,069-entry registry, not lmdiff's curated 31. The
three §5.2 domains come out **completely differently from each other**,
which is the finding.

### 3.1 `instruction_following` — already served

| task | `output_type` | metrics |
|---|---|---|
| `ifeval` | `generate_until` | `prompt_level_strict_acc`, `inst_level_strict_acc`, `prompt_level_loose_acc`, `inst_level_loose_acc` |
| `leaderboard_ifeval` | `generate_until` | same |

**§5.2 plans to write `instruction_following.json`. It does not need
to.** IFEval is the standard benchmark, it is registered, and it relates
to instruction-following exactly as `hellaswag` relates to commonsense.
The work is a `KNOWN_TASK_DOMAINS` entry and an evaluator, not a probe
file.

Two costs. Its metrics are programmatic constraint checks (*"write in
all caps"*, *"exactly three bullets"*) computed by lm-eval's own Python
— lmdiff must call or reimplement them, and they are unlike any of the
five evaluators it has. And IFEval prompts are instruction-style by
construction, which is the L-001 collision Round 1 flagged: unusable
against base models, which is what v01 was rewritten for.

### 3.2 `hallucination` — served, but not by what lmdiff registers

| task | `output_type` | metrics |
|---|---|---|
| `truthfulqa_gen` | `generate_until` | `bleu_max/acc/diff`, `rouge1/2/L_max/acc/diff` |
| `truthfulqa_mc1` / `mc2` | `multiple_choice` | `acc` |

**`truthfulqa_gen` exists and lmdiff registers only the MC variants.**
That is Round 1's finding restated precisely: the problem was never that
truthfulqa is filed under `safety`, it is that lmdiff picked the two
variants where the model chooses among supplied answers. Fabrication
needs generation, and the generative variant is one registry entry away.

Cost: bleu/rouge scoring, which needs `sacrebleu` and `rouge_score` —
new optional dependencies, and a reference-overlap metric unlike
anything lmdiff runs.

### 3.3 `safety` — split three ways, and the part §5.1 names is missing

| task | `output_type` | metrics | runnable offline? |
|---|---|---|---|
| `realtoxicityprompts` | `generate_until` | `perspective_api_toxicity_score` | **no — Google Perspective API key** |
| `toxigen` | `multiple_choice` | `acc`, `acc_norm` | yes |
| `crows_pairs_english` | `multiple_choice` | `likelihood_diff`, `pct_stereotype` | yes |
| `bbq` | `multiple_choice` | `acc` + 24 bias scores | yes |
| `bbq_generate` | `generate_until` | — | yes |
| `ethics_{cm,justice,virtue,utilitarianism,deontology}` | `multiple_choice` | `acc` | yes |

Three distinct things wear the word "safety": **toxic generation**
(generative, external API), **social bias** (MC, likelihood-difference),
and **refusal behaviour**.

**Nothing in lm-eval's 14,069 measures refusal or over-refusal.** That
is §5.1's `safety_regression` definition verbatim, and it is the one
§5.2 set that genuinely has to be written. It also cannot be written
completion-style — a refusal probe is a request, which is an
instruction — so it inherits the same L-001 collision as IFEval.

**Net for §5.2's four:** `general_capability` is v01 and the calibration
set. `instruction_following` is a registry entry, not a file.
`hallucination` is a registry entry plus two dependencies.
`safety_regression` is the only one that is actually a probe-writing
job — and its subject is the one §5.6 already flags for curation care.

---

## 4. Q3 — the two tiers side by side

| domain | v01 | lm-eval default | shared? |
|---|---|---|---|
| `code` | 30 completions | `mmlu_college_computer_science`, MC/`acc` | **both** |
| `math` | 30 completions | `gsm8k`, generation/`exact_match` | **both** |
| `knowledge` | 30 completions | — (`mmlu`/`triviaqa` registered, not in the default set) | v01 only |
| `commonsense` | — | `hellaswag`, MC/`acc_norm` | lm-eval only |
| `reasoning` | — | `arc_challenge`, MC/`acc_norm` | lm-eval only |
| `long-context` | — | `longbench_2wikimqa`, generation/`qa_f1_score` | lm-eval only |

Two of six shared, and **both shared domains are scored differently in
each tier.** v01's `math_001` is `"17 + 25 = "` judged by whichever
evaluator the caller passed; `gsm8k` is a five-shot word problem
generated to 256 tokens and matched after `####`. Same domain label, no
comparable measurement.

### 4.1 Would a scoring-shaped `task_type` split a shared domain?

**Yes — five of nine domains split, and `math` and `code` split inside
the calibration set's own domain list.** From §1.3: `code` 3 MC / 2 gen,
`knowledge` 1/2, `math` 3/1, `reading` 1/1, `language` 1 ll / 1 llr.

### 4.2 Is that a problem or the point?

**The point.** Splitting is what an independent axis looks like; a
second label that never splits the first is a relabelling of it.

Quantified over the 31 curated tasks:

```
guess output_type from domain alone      25/31 = 81%
guess "multiple_choice" every time       19/31 = 61%
      -> domain buys 6 correct answers out of 31

guess domain from output_type alone      11/31 = 35%
```

Domain carries real but partial information about scoring, and scoring
carries little about domain. They cross-cut. **§5.1's eight
capability-named types do not cross-cut domain at all** —
`safety_regression` and `hallucination_probe` both land on the `safety`
domain, `knowledge_drift` on `knowledge`, `general_capability` on
everything else. That axis is close to a function of the first one, and
a second axis that is a function of the first carries no information.

So: scoring method is a genuine second axis and capability is not. But
that is not the same as concluding `task_type` should be *named* after
scoring method, for the reason in §5.

---

## 5. Q4 — what falls out

### 5.1 Scoring method is already on the probe

`from_lm_eval` writes it into `Probe.metadata` for every lm-eval probe:

```python
meta = {"task_name": …, "native_metric": …, "output_type": …,
        "requires_execution": …, "doc_idx": …}
```

Confirmed on live output — `hellaswag:0` carries
`output_type='multiple_choice'`, `native_metric='acc_norm'`,
`requires_execution=False`, plus `choices` and `correct_index`.

So a `task_type` whose values are `multiple_choice` / `generate_until`
would be **a second name for `metadata["output_type"]`**, differing only
in which probes have it — and that is L-035's exact shape, the failure
this project has paid for three times.

**v01 probes carry no metadata at all.** That is the actual gap: scoring
method is recorded for tier 2 and absent for tier 1, in a dict rather
than a field, under no validation, with nothing keeping the two tiers'
vocabularies aligned.

### 5.2 There is nothing for a metric registry to dispatch

§5.4 says `metrics="default"` selects task-type-specific metrics. Tracing
where evaluation metrics are actually computed:

- `_accuracy_for_task` (`experiments/family.py`) **already dispatches on
  `output_type`** — `multiple_choice` → `loglikelihood_accuracy`,
  `generate_until` → `GENERATE_EVALUATORS[task]` or `ContainsAnswer`,
  `requires_execution` → `NaN`, unknown task → `ContainsAnswer`.
- It lives in `run_family_experiment`, **deprecated since v0.4.0 and
  removed in v0.5.0**, and uses the deprecated `InferenceEngine`.
- **The live path computes no accuracy at all.** `_api.py` and
  `_pipeline.py` contain zero occurrences of "accuracy". Reports read
  `result.metadata["accuracy_by_variant"]`, which only the deprecated
  path ever populates — so for any `lmdiff.family()` result it is `{}`.

Two consequences. The registry §5.4 describes **already exists**, and it
dispatches on scoring method, which is the answer to "what should
task_type distinguish" arriving from the code rather than from a plan.
And it is on the path being deleted — so before 4.6 can dispatch
anything, evaluation metrics have to exist on the live path at all. That
is a larger piece of work than 4.3, and it is not currently scheduled
anywhere.

### 5.3 Recommendation

**Neither of the two options as posed. Do not name `task_type` after
scoring method, and do not add scoring method as a third axis — promote
the scoring fields that already exist out of `metadata` into first-class
`Probe` fields, keep lm-eval's names, and drop §5.1's capability-named
`task_type` from 4.3.**

```python
@dataclass(frozen=True)
class Probe:
    id: str
    text: str
    domain: str | None = None          # unchanged — subject matter
    output_type: str | None = None     # NEW — how the model is queried
    scoring: str | None = None         # NEW — how output is judged
    expected: str | None = None
    metadata: dict = field(default_factory=dict)
```

- **`output_type`** — `multiple_choice` / `generate_until` /
  `loglikelihood` / `loglikelihood_rolling`. lm-eval's vocabulary
  verbatim, because inventing a synonym for a value copied from lm-eval
  is how two names for one quantity start. Populated from
  `metadata["output_type"]` for adapter probes; `generate_until` for
  v01, which is what a plain completion is.
- **`scoring`** — the evaluator identity: `exact_match`, `f1`,
  `contains_answer`, `acc`, `acc_norm`, `gsm8k_number_match`, `pass@1`.
  Today this is split between `metadata["native_metric"]` and the
  per-task `GENERATE_EVALUATORS` dict; one field on the probe subsumes
  both. This is what lets v01 stop taking a single caller-chosen
  evaluator for all three of its domains.
- **Both `None`-defaulting**, for Round 1 §3.3's reason unchanged:
  `None` means unlabelled, and a default is a claim nobody made.
- **`requires_execution` stays in `metadata`.** It is a property of the
  metric, and promoting it would be the third field with two homes.

Why this rather than `task_type`:

1. **It is the axis the data supports.** §4.2 — scoring cross-cuts
   domain; capability does not.
2. **It does not duplicate.** §5.1 — a scoring-valued `task_type` would
   be a second name for a field that exists.
3. **It matches the dispatch that already works.** §5.2 —
   `_accuracy_for_task` reads exactly these two things.
4. **It closes the tier gap**, which is the concrete defect: tier 1 has
   no scoring metadata, tier 2 has it in an unvalidated dict.
5. **It does not freeze eight names** that §3 shows are wrong in three
   different ways — one already served, one served by a variant lmdiff
   does not register, one not served at all.

`GeoResult.probe_output_types` / `probe_scoring` replace Round 1 §3.2's
`probe_task_types`, on the same argument and the same 7 → 8 bump: the
seven gates cost the same whenever they move, and unrecorded per-probe
labels are unrecoverable without a GPU re-run. Two tuples rather than
one, at no extra gate cost.

### 5.4 What this leaves open — your call

**§5.1's `task_type` does not disappear, it becomes unscheduled.** Five
of its eight names describe base-vs-variant measurements rather than
probes (Round 1 §2.4(c)), and the three that describe probes are
recoverable from `domain` + `output_type` + probe-set identity. My
reading is that the concept was standing in for "how is this scored",
and once that is a field the remainder does not earn a third axis — but
§5.1 is the specification and dropping a field from it is yours to
decide, not mine.

If you want it kept, the cheapest honest version is a probe-**set**
attribute rather than a per-probe one: `ProbeSet.task_type`, set by
whoever authored the set, with no vocabulary enforced and no metric
dispatching on it until 4.6 has something to dispatch.

**[QUESTION] — the one that blocks 4.6, not 4.3.** §5.2 says
`metrics="default"` auto-selects task-type-specific metrics, but §5.2
above shows the evaluation-metric layer exists only on the deprecated
path and the live path has none. Does restoring evaluation metrics to
`lmdiff.family()` belong in Phase 2, in the v0.5.0 removal work (which
is what deletes the current implementation), or somewhere else? 4.3
needs no answer — the fields are recorded either way — but it is on the
critical path for 4.6 and it is currently in no commit's scope.

---

## 6. Revised Part 2

Provisional. Fields and defaults per §5.3.

1. `Probe.output_type` / `Probe.scoring`, both `str | None = None`,
   after `domain`.
2. `ProbeSet` mirrors: `output_types`, `scorings`,
   `filter(output_type=…, scoring=…)`, `by_output_type()`, plus the two
   JSON sites with the existing write-only-if-truthy conditional.
3. `KNOWN_OUTPUT_TYPES` — lm-eval's four, warn-on-unknown (Round 1 §3.1's
   argument holds unchanged). `scoring` stays open: it names an evaluator,
   and 4.6 owns that registry.
4. `from_lm_eval` promotes `output_type` / `native_metric` from
   `metadata` to the fields, leaving the metadata keys in place for one
   release.
5. `v01.json` — 90 probes labelled `output_type: generate_until` plus a
   per-domain `scoring`, which is the first time v01 says how it should
   be judged rather than taking it as an argument.
6. `GeoResult.probe_output_types` / `probe_scoring`, `geo_schema` 7 → 8:
   eight acceptance lists, two ungated `.get()`s, **payload-level**
   regression test per Z.5.
7. Tests, mutation-checked per assertion. The three that matter: `to_json`
   dropping a new field; the default flipped from `None` to a value; and
   the 7 → 8 bump with one gate left at `("6","7")` — the v0.4.3 failure,
   which a version-string assertion passes under.
8. CHANGELOG, `0.4.4` in `pyproject.toml` / `__init__.py` /
   `test_release.py`.
9. `docs/reference/probe-sets.md`, new — `docs/reference/` holds only
   `run-config.md`, and nothing documents `ProbeSet` for users today.

Carried forward from Round 1, unchanged and still not 4.3 work: a
`ProbeSet` instance passed to `family()` makes the v0.4.3 run-config
emitter raise, so the artifact is dropped for the one input a reader
cannot reconstruct from an identifier. `reproducible: false` has found
its occupant and `docs/reference/run-config.md` says it has none.

---

## 7. Scripts

| script | produced |
|---|---|
| `recount.py` | §0 — 90 probes, 3 domains, 30 each, 0 undomained |
| `taskmeta.py` | YAML resolver: `include` chains, `python_task`, the `generate_until` default |
| `tc.py` | `TaskConfig(task='x').output_type == 'generate_until'` — the §1.3 trap |
| `audit_known.py` | lmdiff's 31 vs lm-eval's YAML: 30/31 agree, `mmlu` is a group |
| `search_reg.py`, `targeted2.py` | §3 — `ifeval`, `truthfulqa_gen`, `realtoxicityprompts`, `bbq_*`, `ethics_*`, no refusal task |
| `crosstab.py` | §4.2 — 5/9 domains split; 81 % vs 61 % vs 35 % |

Round 1's §2.2 finding stands (toxigen is classification, verified from
the live prompt). Round 2's equivalent is §1.3's default-`output_type`
trap and §5.2's dead accuracy path — neither visible without running
something.
