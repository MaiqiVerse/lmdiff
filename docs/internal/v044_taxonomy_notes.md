# v0.4.4 — probe taxonomy: investigation notes

Commit 4.3. Specification: PHASE_PLAN §5.1, in service of §5.2 and §5.4.

Not a design audit. `task_type` is an additive field with no numeric
effect and no new serialization path, so this is the short form: what
the code actually looks like, whether the eight types survive contact
with the probes that exist, and four decisions.

Written by running things. Every count below came from a script against
the shipped code, not from reading it — the scripts are listed in §6.

---

## 1. The shape `task_type` has to match

`Probe` is a frozen dataclass; `ProbeSet` is an immutable wrapper around
a tuple of them (`lmdiff/probes/loader.py`).

```python
@dataclass(frozen=True)
class Probe:
    id: str
    text: str
    domain: str | None = None
    expected: str | None = None
    metadata: dict = field(default_factory=dict)
```

`domain` is a **first-class optional field**, not a metadata key and not
a parallel array. It has four consumers in `ProbeSet`, and `task_type`
should acquire the mirror of each:

| domain | mirror |
|---|---|
| `Probe.domain: str \| None = None` | `Probe.task_type: str \| None = None` |
| `ProbeSet.domains` → sorted distinct, `None` dropped | `ProbeSet.task_types` |
| `ProbeSet.filter(domain=...)` | `filter(task_type=...)` |
| `ProbeSet.by_domain()` → `p.domain or "unknown"` | `by_task_type()` |

Plus the two serialization sites: `from_json` reads `p.get("domain")`,
`to_json` writes it **only when truthy** (`**({"domain": p.domain} if
p.domain else {})`). That conditional matters — it is what keeps
`task_type` out of every probe file that does not use it.

Every `Probe(...)` construction in `lmdiff/` and `tests/` is
keyword-based — 20 sites checked, zero positional — so field placement
is free. Put `task_type` directly after `domain`, since they are the two
labels.

Beyond the probe, `domain` is denormalized in exactly one other place:
`ProbeValidity.domain` (`_validity.py`), for group-by in
`compute_domain_status`. **`task_type` should not follow it there.**
Nothing computes validity per task type, and the field's own docstring
says it exists for a specific lookup. Denormalizing a second label with
no consumer is the duplication this project keeps paying for (L-035).

---

## 2. Do the eight types fit the probes that exist?

Short answer: **your prediction is right, and the two apparent
exceptions both fail on inspection.** Five of the eight types have zero
probes, one is a catch-all, and the remaining two are ambiguous rather
than populated.

### 2.1 The bundled set

`lmdiff/probes/v01.json` — 90 probes, three domains, 30 each:
`code`, `knowledge`, `math`. All completion-style, all with `expected`.

- `math` (30) → `general_capability`. Clean.
- `code` (30) → `general_capability`. Clean.
- `knowledge` (30) → **ambiguous.** `"The capital of France is "`,
  `"The largest planet in our solar system is "` — this is per-item
  factual recall, which is §5.1's definition of `knowledge_drift`
  verbatim. It is equally `general_capability`. Nothing in the probe
  distinguishes the two.

**30 of 90 probes in the bundled set fit two types.**

### 2.2 The `lm_eval:` adapter

`KNOWN_TASK_DOMAINS` holds 31 tasks. `from_lm_eval` assigns `domain`
per *task*, so every probe from one task shares a label — `task_type`
would resolve the same way, through the same table.

I ran the adapter on four tasks and read the actual prompts. Two
findings that reading the table would not have produced:

**`toxigen` is not a `safety_regression` probe.** Its prompt is:

```
Is the following statement hateful? Respond with either Yes or No.
Statement: '...'
```

That is hate-speech *classification*, scored `acc` over a 2-way choice.
§5.1 defines `safety_regression` as "harmful-content refusal /
over-refusal regression". Toxigen measures neither refusal nor
over-refusal — a model that answers "Yes" correctly has not refused
anything. It is `general_capability` on subject matter that sounds like
safety. The `domain: safety` label in `KNOWN_TASK_DOMAINS` is what
makes it look otherwise.

**`truthfulqa_mc1` cannot support the metric its natural type
promises.** By subject it is `hallucination_probe` — §5.1's "factual
fabrication rate". By format it is `output_type: multiple_choice`,
scored by loglikelihood over four options. **A model choosing among
supplied options never fabricates**, so a fabrication-rate metric cannot
run on it regardless of the label. All three `safety`-domain tasks are
`multiple_choice`; across the whole registry, 19 of 31 are
`multiple_choice` and 2 more are loglikelihood-only, leaving 10
`generate_until` tasks — none of which is a hallucination probe.

Same shape as `knowledge_drift`: `triviaqa` and `nq_open` are
`generate_until` factual recall, so they *could* support it, and they
are equally `general_capability`. Ambiguous, not populated.

### 2.3 The tally

| type | probes available today | note |
|---|---|---|
| `general_capability` | **all 90 v01 + all 31 lm_eval tasks** | catch-all |
| `knowledge_drift` | 0 unambiguous | v01 `knowledge`, `triviaqa`, `nq_open` fit it *and* `general_capability` |
| `hallucination_probe` | 0 usable | `truthfulqa_*` fits by subject, cannot support the metric in MC format |
| `safety_regression` | **0** | `toxigen` is classification, not refusal |
| `instruction_following` | **0** | nothing registered has multi-constraint instructions |
| `consistency_check` | **0** | needs paraphrase pairs; none exist |
| `style_drift` | **0** | needs open-ended generation + a style metric |
| `crosslingual_consistency` | **0** | nothing multilingual is registered |

**No probe fits *none* of the eight** — but only because
`general_capability` is a catch-all. A taxonomy with a catch-all cannot
report "unlabelable", so that column proves nothing.

### 2.4 Three things this turned up that §5.1 does not say

**(a) There are nine domains, not five.** §5.1 opens "Domain (existing
5): commonsense / reasoning / math / code / long-context". The real
vocabulary is `code`, `commonsense`, `knowledge`, `language`,
`long-context`, `math`, `reading`, `reasoning`, `safety` — nine in
`KNOWN_TASK_DOMAINS`, of which v01 uses three (`code`, `knowledge`,
`math`, and `knowledge` is not in §5.1's five at all). Nothing depends
on the number, but §5.1's "two orthogonal labels" framing was written
against a domain list that does not exist.

**(b) The domain axis already contains task semantics.** `safety` is a
*domain*. `knowledge` is a *domain*. Those are exactly the two axes
`safety_regression` and `knowledge_drift` are supposed to introduce as
the orthogonal second label. The axes are not orthogonal today because
the first one is already carrying some of the second. Adding `task_type`
does not fix that; it means the same probe can be labelled `safety` on
one axis and `general_capability` on the other, which reads as a
contradiction to anyone looking at a report.

**(c) Five of the eight names describe a measurement, not a probe.**

```
knowledge_drift            ← a measured change
safety_regression          ← a measured change
consistency_check          ← a measured change
style_drift                ← a measured change
crosslingual_consistency   ← a measured change
hallucination_probe        ← a kind of probe
instruction_following      ← a kind of probe
general_capability         ← a kind of probe
```

`drift`, `regression`, `check`, `consistency` are all base-vs-variant
comparisons — things you compute, not things a probe *is*. That is why
§2.1's ambiguity keeps appearing: a factual-recall probe is not "a
knowledge_drift probe", it is a probe on which you can compute knowledge
drift, and also accuracy, and also raw δ.

This matters operationally because of §5.4: "Hallucination Rate only on
`task_type="hallucination_probe"` probes". If the label names the
metric, the dispatch is a tautology — run metric X on probes labelled
"we want metric X" — and the label cannot be checked against the probe's
content, only trusted. `truthfulqa_mc1` is the demonstration: label it
`hallucination_probe` and the registry will dispatch a fabrication-rate
metric onto a multiple-choice task where fabrication is impossible.

**I am not proposing a rename here** — §5.1 settled these eight names
and re-litigating settled decisions is how this project has lost time
before. But the names are about to become a public API surface (a user's
YAML in commit 4.5 will contain these strings), and renaming after that
is a breaking change. Decision 3.1 below is where this lands.

---

## 3. Decisions

### 3.1 Closed enum or open string?

**Recommend: open `str | None`, validated against the known eight at
load time, warning — not raising — on anything else.**

Three reasons, in order of weight:

1. **The 4.6 dispatch never needs to raise.** Task-type-specific metric
   selection is a lookup: known type → its metrics, unknown type →
   nothing task-specific. That is total. There is no code path where an
   unrecognised string has to become an error, so a closed enum buys
   nothing at the point of use.
2. **Warn-on-unknown catches the typo where it happens.** Under a pure
   open string, `instruction_folowing` silently gets no metrics and the
   user finds out after a multi-hour run, if ever. Under a warning, they
   find out at load. This is the failure mode a closed enum is actually
   for, and a warning covers it without the cost.
3. **§2.4(c) says these names are not ready to be frozen.** A closed
   enum makes eight strings a hard API boundary in the same release that
   found five of them describe measurements rather than probes. Open +
   warn keeps the boundary soft for one more commit, which is exactly as
   long as it needs to stay soft — 4.5 is when users start writing them.

Concretely: `KNOWN_TASK_TYPES: frozenset[str]` as the vocabulary, a
`UserWarning` naming the probe id and the unknown value, and nothing in
the hot path — validate in `from_json` and in the `Probe` constructor's
call sites, not per-probe-per-metric.

### 3.2 Does `GeoResult` need `probe_task_types`?

**Recommend: yes, add it now, `geo_schema` 7 → 8.**

The argument that decides it is asymmetry of cost:

- **The cost of the bump is the same now and later.** The seven gates in
  `geo_result_from_json_dict` are `sv in ("5", "6", "7")` — membership
  lists of *accepted versions*, not of new fields. Any bump for any
  reason must edit all seven plus the guard at line 302. Deferring does
  not avoid that work, it just moves it to whichever release next
  touches the schema. (Z.4 item 5 / Z.5: this is the seven-gate trap,
  and the interim rule is edit all seven and add a **payload-level**
  test, not only a version-string one. The v0.4.3 bump failed exactly
  here — `domain_status` deserialized as `{}` while every version-pin
  test passed.)
- **The cost of not bumping is unrecoverable.** Every result saved
  between 4.3 and 4.6 would carry no task types, and 4.6's grouping
  could not be applied to it without re-running — a multi-hour GPU job,
  for results whose probe set may not even be reconstructible (see §4).
  `probe_domains` exists for precisely this reason.

The field is free when unused: `tuple[str | None, ...] = ()`, aligned
with `change_vectors` after the NaN filter, exactly like `probe_domains`
— populated at `_pipeline.py:704` from the same `valid_indices`.

Note one thing that makes this cheaper than it looks: **the new field
itself needs no gate.** `run_config_yaml` is deserialized at line 400
outside every gate, via plain `.get()`, because an additive-nullable
field reads correctly from any version. Only the version *acceptance*
lists move. So the diff is eight one-character edits and one new
`.get()`.

Honest counter, for the record: this ships a field nothing reads for
three commits, and "add it now in case" is a shape this project is
right to be suspicious of. What makes it different from speculative
generality is that the consumer is *scheduled* (4.6, §5.4) and the data
is *destroyed* by not capturing it — neither is true of a speculative
field.

### 3.3 Probes without a `task_type`

**Recommend: `None`. Do not default to `general_capability`.**

You named the reason yourself and it is the whole argument: `None` and
`"general_capability"` mean different things to a metric that groups by
type. Three consequences:

1. **`None` means unlabelled; `general_capability` is a claim.** A probe
   from a user's YAML with no `task_type` is a probe whose type nobody
   stated. Labelling it "generic capability" on the author's behalf
   asserts something the data does not say — the L-039 shape: the
   predicate *has no task_type* does not support the statement *is a
   general-capability probe*.
2. **It is one-way.** A metric that wants to treat unlabelled probes as
   general can do so at the point of use, visibly. Recovering
   "unlabelled" from a defaulted `general_capability` is impossible.
3. **It matches `domain`.** `domain` is `str | None` with `None` for
   unassigned, `probe_domains` is `tuple[str | None, ...]`, and
   `by_domain()` renders `None` as `"unknown"`. Defaulting `task_type`
   while `domain` does not default would be the second pattern the
   instruction says to avoid.

Old saves and old probe sets therefore load as `None`, which is
truthful: they were written before the concept existed.

**Separately — and this is not the same decision — v01.json's 90 probes
should be labelled `general_capability` explicitly in the file.** §2.1
says they are, and an explicit label in the artifact is a statement
someone made; a default is a statement nobody made. That is the whole
distinction, and it is why "default to `general_capability`" and "label
the existing probes `general_capability`" can both be right answers to
different questions.

Caveat carried forward from §2.1: the 30 `knowledge` probes get
`general_capability` under this, which is defensible but discards the
`knowledge_drift` reading. Since §2.4(c) may rename that type anyway,
labelling them the ambiguous-but-safe way now and revisiting in 4.4
costs nothing — a probe file is data, not schema.

---

## 4. Does the run config need changing?

**The §8 conclusion holds for the two paths that have an identifier, and
does not hold for the third — but the gap is not created by
`task_type`.**

v0.4.3 audit §8: *"`task_type` lives in the probe set, and the run
config points at it by identifier… Nothing to reserve now."*

Verified against the shipped emitter. `build_run_config` passes
`probes` through verbatim (`_runconfig.py:155`), and `_resolve_probes`
(`_api.py:100`) accepts three forms:

| form | emitted | task types recoverable? |
|---|---|---|
| `None` / `"v01"` | `probes: v01` | yes — bundled file, pinned by `provenance.lmdiff` |
| `"lm_eval:hellaswag+…"` | `probes: lm_eval:hellaswag+…` | yes — resolved from `KNOWN_TASK_DOMAINS`, pinned by `provenance.lmdiff` + `provenance.lm_eval` |
| `ProbeSet` instance | **nothing — the emitter raises** | n/a |

The first two are fine, and the reason is worth stating because it is
the same reason `domain` is fine: the identifier alone is not enough,
since resolution goes through a table *inside lmdiff*. What makes it
sound is `provenance.lmdiff`, which pins the table. `task_type`
inherits that guarantee unchanged. **Nothing to reserve, confirmed.**

The third is a v0.4.3 defect, found here rather than fixed here.

### 4.1 Defect — a `ProbeSet` instance drops the entire run config

```
probes = ProbeSet.from_json("my_probes.json")
result = lmdiff.family(base=…, variants=…, probes=probes)
# RuntimeWarning: could not emit run configuration:
#   RepresenterError: ('cannot represent an object', ProbeSet(name='v01', n=90, …))
# result.run_config_yaml is None
```

Reproduced end-to-end through `_attach_run_config`, not inferred:
`yaml.safe_dump` cannot represent a `ProbeSet`, the `except Exception`
at `_api.py:371` catches it, and the run passes with a warning and no
artifact.

Three things make this worth fixing before 4.5 rather than after:

- **It fails on the case provenance matters most for.** A named
  benchmark is recoverable from the identifier. A hand-built probe set
  is the one thing a reader cannot reconstruct — and it is the only path
  that produces no record at all.
- **`reproducible: false` has found its occupant.** v0.4.3 shipped
  `reproducible` / `non_serializable` per AA.5 with, in the audit's
  words, no current occupant, and `docs/reference/run-config.md`
  currently says *"Nothing in lmdiff currently triggers it."* That
  sentence is wrong. A nameless in-memory `ProbeSet` is precisely
  "something Python can hold and YAML cannot name" — the flag's stated
  criterion. The escape hatch exists and the one path that should set it
  raises instead.
- **It is the path 4.5 turns into the main one.** A YAML probe-set
  loader produces exactly this object.

**Recommended fix** (its own commit, or v0.4.5 — your call; it is not
4.3 work): give `probes` the `_variant_block` treatment. A `ProbeSet`
with a `name` emits `probes: <name>`, matching what the string form
would have produced. A `ProbeSet` without one sets `reproducible: false`
and adds `non_serializable: [{path: probes, reason: in-memory ProbeSet
with no name}]`. Note `ProbeSet.from_json` populates `name` from the
file, so the common case emits a usable identifier. The docs sentence
needs correcting either way.

---

## 5. What Part 2 looks like

Provisional, pending your review of §3.

1. `Probe.task_type: str | None = None`, after `domain`.
2. `ProbeSet.task_types`, `filter(task_type=…)`, `by_task_type()`,
   `from_json` / `to_json` — the §1 mirror, including the write-only-if-
   truthy conditional.
3. `KNOWN_TASK_TYPES` frozenset + warn-on-unknown (§3.1).
4. `v01.json`: 90 probes labelled `general_capability` (§3.3).
5. `KNOWN_TASK_DOMAINS`: `TaskInfo.task_type`, defaulting
   `general_capability`, so the adapter resolves it the way it resolves
   domain. Per §2.2 every one of the 31 is `general_capability` — which
   is the honest answer, not a placeholder.
6. `GeoResult.probe_task_types` + `geo_schema` 7 → 8: eight acceptance
   lists, one ungated `.get()`, and a **payload-level** regression test
   per Z.5.
7. Tests, mutation-checked per assertion. The mutations that matter:
   `to_json` dropping `task_type` (catches a lossy round-trip); the
   default flipped to `"general_capability"` (catches §3.3 — a test that
   only checks *a* value passes under this); and the 7→8 bump with one
   gate left at `("6", "7")` (catches §3.2 — the v0.4.3 failure, and a
   version-string assertion passes under it).
8. CHANGELOG, `0.4.4` in `pyproject.toml` / `__init__.py` /
   `test_release.py:_CURRENT_VERSION`.
9. Docs: **`docs/reference/probe-sets.md`, new.** `docs/reference/` holds
   only `run-config.md`, and nothing anywhere documents `ProbeSet`,
   `domain`, or the probe-spec forms for users — README covers the
   `probes=` argument in passing and no more. The taxonomy needs a home
   and there is no existing page to add a section to.

### 5.1 One thing 4.4 inherits

L-001: v01's math probes were once instruction-style, produced
degenerate output on base models, and were rewritten to completion
style; CLAUDE.md makes completion style the convention and says
instruction-style probes belong in a separate versioned file.

§5.2 plans `instruction_following.json` and `safety_regression.json` as
builtin sets. Both are instruction-style **by definition** — a
multi-constraint instruction cannot be phrased as a completion, and
neither can a refusal probe. So two of §5.2's four builtin sets are
unusable against base models, which is the audience v01 was rewritten
for.

Not a 4.3 problem — no code here touches it. Recorded because it is a
constraint on 4.4 that the taxonomy is what surfaces, and because "which
four" is already flagged for revisiting against lab feedback (§5.2's
first added constraint).

---

## 6. Scripts

Throwaway, in the session scratchpad, listed so the numbers above can be
re-derived:

| script | produced |
|---|---|
| `inspect_v01.py` | v01 shape: 90 probes, 3 domains, 30 each, keys `id/text/domain/expected` |
| `label_probe.py` | the nine-domain vocabulary, §2.4(a) |
| `try_adapter.py` | real prompts for `hellaswag`, `truthfulqa_mc1`, `toxigen`, `gsm8k` — §2.2's two findings |
| `output_types.py` | 19 `multiple_choice` / 10 `generate_until` / 2 loglikelihood; all three `safety` tasks MC |
| `rc_probeset.py` | `RepresenterError` on a `ProbeSet` in `build_run_config` |
| `rc_endtoend.py` | the same via `_attach_run_config`: `RuntimeWarning`, `run_config_yaml is None` |

§2.2 and §4.1 are the two findings that came only from running things.
Both are invisible in the source — `toxigen`'s prompt is in a dataset,
and the emitter's failure is a library type rule.
