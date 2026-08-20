# v0.4.3 design audit — run-configuration schema and emission

**Status:** design audit, no implementation. Phase 2 commit 4.2, per
PHASE_PLAN Update 7 AA.
**Scope:** emission only. Execution (a loader that runs a YAML file) is
deliberately later — AA.6.

---

## Table of contents

0. [Headline findings](#0-headline-findings)
1. [Config serialization surface](#1-config-serialization-surface)
2. [Call-level parameters](#2-call-level-parameters)
3. [Schema design and the worked example](#3-schema-design-and-the-worked-example)
4. [The provenance block](#4-the-provenance-block)
5. [The escape hatch](#5-the-escape-hatch)
6. [Emission — where, when, what triggers it](#6-emission)
7. [Round-trip guarantee, and the `steering` defect](#7-round-trip-guarantee)
8. [Forward compatibility with commits 4.3 and 4.5](#8-forward-compatibility)
9. [Migration and existing artifacts](#9-migration)
10. [Open questions with recommendations](#10-open-questions)

---

## 0. Headline findings

Eight things the investigation turned up that change the shape of the work.

**0.1 — `Config` already serializes.** `Config.to_dict()` and
`Config.from_dict()` exist at `_config.py:555` and `:566`, with a
recursive `_serialize_value` (`:199`), sub-spec reconstruction
(`_deserialize_subspec`, `:227`), and numpy wrapping (`_np_to_dict`,
`:141`). This commit does **not** need to design a serializer. It needs
to decide what wraps that dict, and to fix one defect in it (0.3).

This materially reduces the commit. AA.4's "one-directional, not a third
API" is not an aspiration to engineer toward — it is already the
structure, and the work is to keep it that way.

**0.2 — every real config round-trips through YAML today.** All eight
`Config` objects the 7-variant calibration builds survive
`Config → to_dict → yaml.safe_dump → yaml.safe_load → from_dict` to
field-by-field identity. Verified by `v043_roundtrip_check.py`, stage A.
YAML imposes no constraint that JSON did not.

**0.3 — `steering` fails the identity check, and the serializer is not at
fault.** The comparator is. `_values_equal` does not recurse into
dataclasses, which is a **general shape** — any future sub-spec holding an
array inherits it — and it reaches past serialization into
`is_runtime_only_modification_of`, causing redundant model loads.
Pre-existing, outside this commit. Diagnosis §7.2, generality §7.3,
remedy and sequencing §7.4. Not fixed here, per the audit constraints.

**0.4 — `min_valid_fraction` must be plumbed onto the public API.**
AA.2's settled example lists it as executable configuration; it is not a
parameter of `family()` or `compare()`. Raised as a collision between two
settled decisions and **resolved in favour of AA.3**: add the kwarg.
Omission fails to pin, and this is the parameter whose default most
recently changed meaning. §10.6, including the precedence rule and the
Z.5 bookkeeping.

**0.5 — moving arrays out of line makes the artifact a bundle.** §1.2
recommends `__ref__` rather than inlining megabytes of float literals.
§1.4 states what that costs: what must travel, what error a dangling
reference raises and when, and why a missing sidecar is **not**
`reproducible: false`. That last point corrected a defect in §5's
criterion. Vacuous in practice today — no shipped experiment carries an
array — but it changes what "attach" means, so §10.1 covers it too.

**0.6 — `metrics: default` does not satisfy AA.3.** It names something
that moves. The "provenance pins it" argument fails twice — `provenance`
is dropped on load, and the same argument would justify omitting
`min_valid_fraction`, which §10.6 rejects. Expand to the resolved
five-name list (§3.3). Auditing for other moving pointers found one more:
**`probes: lm_eval:…` resolves against an optional dependency whose
version was not in the provenance block**, so `lm_eval` is added to it
conditionally.

**0.7 — `GeoResult` must carry the emitted YAML, and `geo_schema` goes
6 → 7.** An earlier draft recommended emitting at `_api.family()` *and*
embedding in HTML, which cannot both hold: `to_html()` runs long after
the Configs leave scope, often on a reloaded result. Resolved in §6 —
`GeoResult` gains one `str` field holding the emitted text verbatim.
This revises §10.1's "not the JSON": the objection was to a structured
record competing with `metadata`, and an opaque string cannot disagree
with itself.

**0.8 — emission is not a one-line `safe_dump`.** The worked example
needs comments to be unambiguous, `yaml.safe_dump` cannot emit them, and
the annotated output must still round-trip or AA.2's "runnable" decision
breaks. §3.4 gives the four-step emitter and the self-check that guards
it.

---

## 1. Config serialization surface

Field-by-field, from `_config.py`. "Verdict" is the result of
`yaml.safe_dump` on the serialized value, measured by
`v043_roundtrip_check.py` stage C, not judged by eye.

### 1.1 `Config` fields

| field | type | serialized as | verdict |
|---|---|---|---|
| `model` | `str` | scalar | clean |
| `name` | `str \| None` | scalar / null | clean |
| `adapter` | `AdapterSpec \| None` | mapping | clean |
| `quantization` | `QuantSpec \| None` | mapping | clean |
| `pruning` | `PruneSpec \| None` | mapping | clean |
| `system_prompt` | `str \| None` | scalar | clean |
| `icl_examples` | `tuple[ICLExample] \| None` | list of mappings | clean |
| `context` | `tuple[Message] \| None` | list of mappings | clean |
| `soft_prompts` | `np.ndarray \| None` | `{__numpy__, data, dtype, shape}` | **awkward** |
| `kv_cache_compression` | `KVCacheSpec \| None` | mapping | clean |
| `decode` | `DecodeSpec` | mapping | clean |
| `steering` | `SteeringSpec \| None` | mapping w/ `__numpy_dict__` | **awkward**, and see §7.2 |
| `tokenizer_id_override` | `str \| None` | scalar | clean |
| `capabilities_required` | `frozenset[str]` | sorted list | clean |
| `training_recipe_summary` | `str \| None` | scalar | clean |

**Nothing is unserializable.** The `non_serializable` escape hatch (AA.5)
has no current occupant — which is the expected result, since the
motivating case (an in-memory model object) is post-1.0. It still ships,
per the settled decision, and §5 states the criterion for using it.

### 1.2 The two awkward fields

Both hold numpy. `soft_prompts` is a bare `ndarray`; `SteeringSpec.vectors`
is a `dict[str, ndarray]`. Serialized, a 2×2 float32 array becomes:

```yaml
soft_prompts:
  __numpy__: true
  data: [[0.10000000149011612, 0.20000000298023224], ...]
  dtype: float32
  shape: [2, 2]
```

Correct, YAML-safe, round-trips — and unreadable at any real size. A
steering vector set for a 32-layer model is megabytes of float literals
inside a file whose purpose is human inspection.

**Recommendation.** Do not inline arrays. Emit a reference and write the
payload beside the YAML:

```yaml
soft_prompts: {__ref__: soft_prompts.npy, dtype: float32, shape: [2, 2]}
```

This keeps the YAML readable, keeps the artifact runnable (the loader
resolves `__ref__` relative to the YAML), and matches how every other
tool in this space handles tensors. It does mean the artifact is a
directory rather than a file when arrays are present — stated as a
consequence, not hidden.

**Note the priority.** No shipped experiment uses `soft_prompts` or
`steering`; the calibration configs are all `model` + `decode` +
optionally `system_prompt`. This is a correctness-of-design question, not
a blocker, and `__ref__` can land in a later commit provided the schema
reserves the key now.

### 1.3 Sub-spec fields

All clean. Recorded because "expand defaults on emission" (AA.3) needs
each default to be writable, and every one is a scalar or `None`:

| spec | fields | defaults |
|---|---|---|
| `AdapterSpec` | `type, path, rank, target_modules` | `lora`, `None`, `None`, `None`; `path` required by `__post_init__` |
| `QuantSpec` | `method, bits, compute_dtype, config_path` | `int4`, `None`, `bf16`, `None` |
| `PruneSpec` | `type, sparsity, pattern, config_path` | `preloaded`, `None`, `None`, `None` |
| `ICLExample` | `user, assistant, metadata` | required, required, `None` |
| `Message` | `role, content, metadata` | required, required, `None` |
| `KVCacheSpec` | `method, keep_ratio, compute_dtype` | `none`, `None`, `bf16` |
| `DecodeSpec` | `strategy, temperature, top_p, top_k, num_samples, max_new_tokens, seed` | `greedy`, `1.0`, `1.0`, `0`, `1`, `16`, `None` |
| `SteeringSpec` | `vectors, scale, application, positions` | `None`, `1.0`, `add`, `all` |

`ICLExample.metadata` and `Message.metadata` are
`tuple[tuple[str, Any], ...]` — a hashable stand-in for a dict. They
serialize to a list of pairs, which is correct but reads poorly. Minor;
noted for completeness.

---

### 1.4 What `__ref__` costs, and the tension it creates

`__ref__` means an emitted config can be a YAML *plus sidecar array
files*. That sits against the "the artifact travels alone" argument used
in §3.4 to reject comment-free emission. The tension is real and needs
resolving rather than noting.

**The resolution is that the two move different things.** The comment-free
fallback moves *explanation* out of the artifact into report prose that
may not travel with it — and explanation is what makes the artifact
self-describing. `__ref__` moves *bulk payload* into a sibling file that
is part of the same artifact. A tensor is not explanation; no reader
learns anything from four hundred thousand float literals. So the
principle is narrower than "one file": **everything needed to understand
the run is in the YAML; everything needed to re-run it is in the
bundle.**

**The artifact is therefore a bundle, and the schema should say so.**

```
report.runconfig.yaml            # always
report.runconfig.d/              # only when a __ref__ exists
  soft_prompts.npy
  steering.l10.npy
```

For every experiment lmdiff has actually run, `report.runconfig.d/` does
not exist and the bundle is exactly one file. The tension is real in
principle and currently vacuous in practice — worth stating plainly so
nobody designs elaborate machinery for a case that has never occurred.

**Copy-edit-run: what has to be on disk.** The YAML and, if it names any
`__ref__`, the sibling directory. Copying the YAML alone is safe when it
contains no `__ref__` — which the reader can see by looking, since
`__ref__` is a visible key rather than an implicit dependency. That
visibility is the reason to prefer a key over, say, an out-of-band
convention: the file states its own completeness.

**A `__ref__` that no longer resolves: fail at load, before any weights.**
The loader resolves every `__ref__` during parse and validation, not
lazily at first use. Error names the YAML path, the filesystem path
tried, and that the run cannot proceed:

```
RunConfigError: soft_prompts references 'report.runconfig.d/soft_prompts.npy'
  (resolved to /abs/path/...), which does not exist.
  The run config was emitted as a bundle; the sidecar directory must
  travel with the YAML.
```

Lazy resolution would surface this after a 14 GB model load, which is the
difference between a two-second failure and a five-minute one.

**Does a missing reference mean `reproducible: false`? No — and §5's
criterion as written says otherwise, which is a defect in §5.** See the
sharpened criterion there. Briefly: `reproducible` is a property of the
*configuration*, decided at emission and permanent; a missing sidecar is a
property of *this copy on this disk*, discovered at load and fixed by
obtaining the file. Conflating them would require a loader to rewrite
`reproducible: true` → `false` in a file it is only reading, which is
incoherent.

## 2. Call-level parameters

Not everything belongs on `Config`. From `_api.py`:

```python
def family(base, variants, *, probes=None, n_probes=100, metrics="default",
           max_new_tokens=16, task_overrides=None, engine=None,
           seed=None, progress=None)
```

`compare()` is identical with `variant` singular.

The settled criterion (AA.3) is: **anything affecting a computed value is
executable configuration.**

| parameter | classification | reasoning |
|---|---|---|
| `base`, `variants` | executable | the comparison itself |
| `probes` | executable | selects the probe set — see §10.2 |
| `n_probes` | executable | changes which probes are drawn |
| `metrics` | executable | selects what is computed |
| `max_new_tokens` | executable | enters `T_i`, so it changes validity classification |
| `task_overrides` | executable | per-task `max_new_tokens`; same reasoning |
| `seed` | executable | pins sample-decode RNG (L-031) |
| `engine` | **neither** | an injected object, not a value. Test seam; has no YAML representation and should not acquire one |
| `progress` | runtime detail | affects display only, never a number |

**`min_valid_fraction` is the exception.** It affects computed values as
directly as anything on this list — it is what turns a domain's share to
`None` — but it is not currently a parameter of either public entry point.
Resolved in §10.6: it is added to `family()` and `compare()` in this
commit, with precedence *explicit call argument > default*.

---

## 3. Schema design and the worked example

### 3.1 Shape

```
lmdiff_schema: <int>          # schema version, see §10.5
base: <model-id | Config-block>
variants: {<name>: <model-id | Config-block>}
probes: <probe-set identifier>
n_probes: <int>
metrics: <str | list[str]>
max_new_tokens: <int>
task_overrides: {<task>: {<param>: <value>}}
seed: <int | null>
min_valid_fraction: <float>   # pending §10.6
reproducible: <bool>
non_serializable: [{path, reason}]   # present only when reproducible: false
provenance: {...}             # read-only, dropped on load
```

A variant is **either** a bare model-id string **or** a mapping that is
exactly `Config.to_dict()` minus its `None`-valued optional fields. The
string form is sugar for "this model, all defaults", and emission uses it
whenever the Config is all-default apart from `model`. That keeps the
common case readable without introducing a second representation — the
loader expands a string to `Config(model=s)` and there is one code path.

### 3.2 Worked example — the real 7-variant calibration

Generated from `tests/integration/_v041_7variant_spec.build_run_kwargs()`,
not hand-written:

```yaml
lmdiff_schema: 1
base: meta-llama/Llama-2-7b-hf
variants:
  yarn: NousResearch/Yarn-Llama-2-7b-128k
  long: togethercomputer/LLaMA-2-7B-32K
  code: codellama/CodeLlama-7b-hf
  math: EleutherAI/llemma_7b
  chat: meta-llama/Llama-2-7b-chat-hf
  temp_1.5:
    model: meta-llama/Llama-2-7b-hf
    decode:
      strategy: sample
      temperature: 1.5
      top_p: 1.0
      top_k: 0
      num_samples: 1
      max_new_tokens: 16
      seed: null
  system_prompt:
    model: meta-llama/Llama-2-7b-hf
    system_prompt: You are concise.
    decode:
      strategy: greedy
      temperature: 1.0
      top_p: 1.0
      top_k: 0
      num_samples: 1
      max_new_tokens: 16
      seed: null
probes: lm_eval:hellaswag+arc_challenge+gsm8k+mmlu_college_computer_science+longbench_2wikimqa
n_probes: 100
max_new_tokens: 16
task_overrides:
  gsm8k:
    max_new_tokens: 256
  longbench_2wikimqa:
    max_new_tokens: 128
seed: 42
metrics: [bd, drift, share, direction, specialization_zscore]
min_valid_fraction: 0.5
reproducible: true

# ---- provenance: read-only, ignored on load ----
provenance:
  lmdiff: 0.4.3
  python: 3.12.8
  torch: 2.6.0
  transformers: 4.51.0
  run_id: '2026-08-19T07:56:12Z'
  geo_schema: 6
```

**Three things writing this out revealed.**

*The five weight-mod variants collapse to one line each.* The string form
is not a nicety — without it, this document is five near-identical
seven-line `decode` blocks, and the two variants that actually differ
stop being visible. Readability here is a correctness property: this
artifact exists to be read.

*`decode.seed: null` next to top-level `seed: 42` reads like a
contradiction.* It is not — `DecodeSpec.seed` takes precedence per
variant, and `None` means "fall back to the family seed" (L-031). But a
reader copying this file to reproduce a sampling run will see `null` and
reasonably conclude the run was unseeded. **Recommend a comment on
emission**: `seed: null   # inherits family seed: 42`. PyYAML cannot emit
comments via `safe_dump`, so this needs templated emission rather than a
straight dump — a real constraint on the implementation, discovered by
writing the example.

*`task_overrides` duplicates `max_new_tokens` at two levels* and the
precedence is not visible in the file. Same remedy.

### 3.3 Moving pointers — does `metrics: default` satisfy AA.3?

**No. It should be expanded on emission, for the same reason
`min_valid_fraction` must be written.**

The tempting argument is that `default` is pinned by
`provenance.lmdiff`, since the metric set is a function of the version.
That argument fails twice:

- **`provenance` is dropped on load** (AA.2). The loader never sees it
  when resolving `metrics: default`, so it pins nothing operationally.
  And version mismatch is a *warning*, not a refusal (AA.3) — the run
  proceeds with the new metric set.
- **The argument is symmetric with the one already rejected.** "Its
  default is determined by the version, and provenance captures the
  version" is exactly the case for omitting `min_valid_fraction`, and
  §10.6 rejects it. Accepting it here would be inconsistent.

The operative test AA.3 sets is: *can a reader determine from the
artifact what was actually used?* `min_valid_fraction: 0.5` passes.
`metrics: default` does not.

Expansion is cheap. `_resolve_metrics` (`_api.py:152`) maps `"default"`
to a hardcoded five-element list:

```yaml
metrics: [bd, drift, share, direction, specialization_zscore]
```

Five short names, and the loader already accepts an explicit list, so
the emitted artifact stays runnable. The worked example in §3.2 should
read this rather than `default`.

**Other moving pointers.** Auditing the rest of the schema for the same
shape — a value that names something version-dependent rather than
stating it:

| key | moving? | treatment |
|---|---|---|
| `metrics: default` | **yes** | expand to the resolved list, above |
| `probes: lm_eval:…` | **yes** | see below — cannot be expanded cheaply |
| `n_probes`, `max_new_tokens`, `seed`, `task_overrides` | no | literal values |
| `min_valid_fraction` | no, once §10.6 lands | literal value |
| every `Config` / sub-spec field | no | literals; defaults expanded per AA.3 |

**`probes` is the one that cannot be fixed by expansion.** An
`lm_eval:hellaswag+…` identifier resolves against the installed lm-eval,
which is an *optional* dependency (`lm-eval>=0.4.0`, pyproject extras) —
so probe text, splits, and ordering can all change under a config that
looks identical. Expanding it inline means inlining 500 probes, which
§10.2 rejects on readability grounds and which would dwarf the rest of
the file.

The remedy is provenance, not expansion: **add `lm_eval` to the
provenance block whenever the probe spec is an `lm_eval:` identifier.**
§4 lists six fields; this makes it seven, conditionally. Without it the
artifact pins the metric set, the seed, and every decode parameter, and
leaves the actual probe text unpinned — which would be the largest
remaining hole.

This does not make the `lm_eval:` form reproducible in the strong sense
(§5) — it makes the mismatch *detectable*, which is the same standard
§10.2 applies to user-supplied probe sets via content hash.

### 3.4 Templated emission, and what it costs

`yaml.safe_dump` cannot emit comments. The two clarifications §3.2 calls
for — annotating `decode.seed: null` with the family seed it inherits,
and marking which `max_new_tokens` wins — are therefore not reachable
from a straight dump. Emission has to be templated, or dump-then-annotate.

**The constraint that makes this non-trivial: the annotated output must
still be valid YAML that the loader reads back unchanged.** A
comment-bearing artifact that no longer round-trips would break AA.2's
settled "the emitted YAML is runnable" decision — which is the whole
premise of one schema rather than two. Comments are safe by construction
(YAML ignores them), but hand-assembled or templated output is not
automatically well-formed: key ordering, quoting of strings that look
like numbers or booleans, multi-line strings in `system_prompt`, and
unicode all become the emitter's problem rather than PyYAML's.

**Consequence for the implementation estimate: emission is not a one-line
`safe_dump`.** It is:

1. build the ordered mapping (§3.1),
2. dump it,
3. inject comments at known keys,
4. **re-parse the result and assert it equals the pre-dump mapping.**

Step 4 is the guard, and it is cheap — one `safe_load` and one `==`. It
should be an assertion in the emitter, not only a test, because the
failure mode is silent: an artifact that looks right and no longer loads.

A defensible alternative is to emit comment-free YAML and put the
precedence explanation in the report prose beside it, which keeps
emission to a single `safe_dump`. That trades a self-explanatory artifact
for a simpler emitter. I would not take it — the artifact's value is that
it travels alone — but it is the fallback if step 3 proves fiddly.

---

## 4. The provenance block

The criterion I applied: **record what you cannot recover from the
GeoResult, and nothing else.** The block is not the report.

| field | include | why |
|---|---|---|
| `lmdiff` | **yes** | the answer to "why do I get different numbers" (AA.1). `share_per_domain` changed meaning twice in two minor versions |
| `torch`, `transformers` | **yes** | not recoverable, and both change numerics |
| `python` | **yes** | one line, occasionally the answer |
| `run_id` (ISO timestamp) | **yes** | the only handle for "which run was this" |
| `geo_schema` | **yes** | disambiguates which loader path produced the companion JSON |
| `lm_eval` | **conditionally** | only when `probes` is an `lm_eval:` identifier. Probe text, splits and ordering come from that package, so without it the artifact pins everything except the probes themselves (§3.3) |
| `duration_s` | no | diagnostics, not provenance; varies with hardware and tells you nothing about the numbers |
| `devices` | no | ditto. A reader wanting this has the run log |
| `probes_excluded` | **no** | `GeoResult.metadata["n_skipped"]` already holds it |
| `domain_status_summary` | **no** | `GeoResult.domain_status` holds it exactly, per cell |

The last two are the ones AA.8 question 3 flags, and they are the clearest
cuts: both are already in the artifact this file ships beside.
Duplicating them creates two copies free to disagree — the failure this
project has hit three times (L-035).

This trims AA.2's illustrative block from nine fields to six, plus
`lm_eval` when the probe spec needs it.

---

## 5. The escape hatch

Ships as specified in AA.5, with no current occupant (§1.1).

```yaml
reproducible: false
non_serializable:
  - path: variants.custom_ft
    reason: in-memory model object, not a resolvable identifier
```

**Criterion for setting it:** a value that **no possible file could
express**. Not "awkward to serialize" — `soft_prompts` is awkward and
fully reproducible. Not "the payload is missing" either — see below. The
temptation will be to reach for `reproducible: false` whenever something
is inconvenient, and that would make the flag meaningless.

An earlier draft of this section said "cannot be reconstructed from the
file alone", which is wrong and worth recording as a correction: under
that wording a `__ref__` config would be non-reproducible, since the YAML
alone is indeed not enough. §1.4 is what surfaced it.

**`reproducible` is a property of the configuration; a missing sidecar is
a property of the copy.** They differ in three ways that matter:

| | `reproducible: false` | missing `__ref__` target |
|---|---|---|
| decided | at **emission**, by the emitter | at **load**, by the loader |
| lifetime | permanent — no file will ever express an in-memory object | contingent — fixed by obtaining the sidecar |
| remedy | none; re-run from Python | copy the bundle |

Conflating them would require a loader to rewrite `reproducible: true` →
`false` in a file it is only reading, which is incoherent. A missing
sidecar raises `RunConfigError` (§1.4); it does not mutate a field.

`reproducible` lives in the executable section, not `provenance`, because
it determines whether the file can run at all. A loader seeing `false`
raises and names the paths rather than pretending.

---

## 6. Emission

**When.** Alongside every report — wherever `to_html`, `to_markdown`, or
`save` writes. Not on `family()` return, since the object is in memory
and the caller has the Config.

**Trigger.** Automatic, not opt-in. An artifact you must remember to
request is one you will not have when you need it.

**What it needs.** The emitter requires the `Config` set and the
call-level parameters. `GeoResult` currently holds neither — it has
`base_name` (a display string) and `variant_names`, not the objects.

**The ordering problem.** An earlier draft recommended emitting at
`_api.family()` and left it there. That does not survive §10.1's HTML
embed: `to_html()` is a `GeoResult` method, called arbitrarily later —
often on a result reloaded from JSON in a different process — and by then
the `Config` objects are long out of scope. Recommending both without
reconciling them was a gap.

**Resolution: `GeoResult` gains `run_config_yaml: str | None`, and it is
serialized.**

The emitted YAML text, verbatim, as a single string. Not the `Config`
objects, and not a structured mirror of them:

- **One `str` field.** No import dependency on `_config`, no new
  serialization path, nothing for a future field to drift out of sync
  with — it is the same bytes the sidecar contains, by construction.
- **`_api.family()` still does the emitting**, where the inputs are in
  scope. It hands the finished text to `GeoResult`. The producer stays
  where the data is; the artifact travels with the result.
- **`to_html()` embeds `self.run_config_yaml`** with no knowledge of
  `Config` at all.

**This revises §10.1's "not the GeoResult JSON."** That objection was to
a *structured* configuration record competing with `metadata` field by
field — two representations of the same values, free to disagree. An
opaque verbatim string is not a competing representation and cannot
disagree with itself. And carrying it is the only way `to_html()` on a
**reloaded** result can embed the config; without it, re-rendering a
report from a saved JSON silently drops the provenance this commit exists
to add. That failure mode — an operation that looks like it worked and
quietly lost something — is the one this project keeps meeting.

**Cost: `geo_schema` 6 → 7.** The loader constructs by explicit key, so
old readers ignore the field and new readers see `None` on old files;
no preserving-loader path is needed, unlike 5 → 6. Bumping is still
right, because the project's convention has been to bump on field
additions (v3 `probe_domains`, v4 `avg_tokens_per_probe`), and a reader
should be able to tell from the version whether to expect the field.

---

## 7. Round-trip guarantee, and the `steering` defect

### 7.1 The test

AA.4 mandates `Config → YAML → Config` identity, and it belongs in the
first commit. `v043_roundtrip_check.py` is its prototype: stage A over
every calibration Config, stage B over a maximal Config exercising every
field. Promote it to `tests/unit/` parametrized over both sets.

**Per L-040, the test needs a mutation check before it is trusted.** The
obvious failure is a comparator too lenient to notice a lossy field —
which is precisely what stage B found, in reverse.

### 7.2 The `steering` finding

Stage B reports `steering` as failing identity. **The serializer is
correct and the comparator is wrong.**

Demonstrated:

```
_serialize_value(SteeringSpec)  -> {'vectors': {'__numpy_dict__': True, ...}}
_deserialize_subspec(...)       -> vectors: {'l10': array([0.5, 0.6])}
_values_equal(orig.vectors, restored.vectors)  -> True     # data is fine
_values_equal(orig_spec, restored_spec)        -> False    # spec compares unequal
```

`_values_equal` (`_config.py:162`) recurses into `dict`, `list` and
`tuple`, but **not into dataclasses**. Two `SteeringSpec` instances fall
through to `a == b`, which is dataclass equality, which compares the
`vectors` dicts with plain `==`, which hits ambiguous numpy truth and
raises — swallowed by the trailing
`except (ValueError, TypeError): return False`.

**This reaches beyond serialization.** `steering` is weight-affecting, so
`Config.is_runtime_only_modification_of` compares it via `_values_equal`
(`:546`). Two value-identical Configs carrying steering vectors are
therefore judged *not* runtime-compatible:

```
two independently-built, value-identical Configs with steering:
  _values_equal(a.steering, b.steering) : False
  a.is_runtime_only_modification_of(b)  : False
  reflexive a vs a                      : True      # `a is b` short-circuit
```

Consequence: a redundant model load. It errs in the safe direction — the
module's stated "default-to-strict" policy — so this is a performance
defect, not a correctness one, and no shipped experiment uses steering.

`soft_prompts` is unaffected: a bare `ndarray` hits the numpy branch
directly.

### 7.3 This is a general shape, not a `SteeringSpec` quirk

`_values_equal` not recursing into dataclasses is a gap in the
*comparator*, and `SteeringSpec` is merely the only sub-spec that
currently holds an array. **Any future sub-spec with an array field
inherits the bug on the day it is added**, silently, and its symptom will
be a redundant model load that nobody attributes to a comparator.

The classification comment at `_config.py:44` already anticipates future
fields ("revisit when adding new fields") and applies a default-to-strict
policy for the *reuse* question. It does not anticipate this: a field can
be correctly classified as weight-affecting and still be compared wrongly.

The `soft_prompts` case shows the boundary — a bare `ndarray` hits
`_values_equal`'s numpy branch directly and works. The gap is specifically
**numpy nested inside a dataclass**.

### 7.4 Which remedy, and where it lands

**Remedy: add a dataclass branch to `_values_equal`**, recursing
field-by-field, rather than defining `__eq__` on each spec.

Two reasons. First, one edit covers every present and future sub-spec,
where per-spec `__eq__` is a list that must be maintained — and an
unmaintained list is how this project has repeatedly acquired drift
(L-035). Second, `_values_equal` is already the project's designated
"compare things that may contain arrays" helper, with the numpy, dict and
sequence branches sitting right there; a dataclass branch belongs beside
them. Overriding `__eq__` on frozen dataclasses would also change
hashing semantics, which is a larger blast radius than the problem
warrants.

Cost: roughly

```python
if is_dataclass(a) and is_dataclass(b):
    if type(a) is not type(b):
        return False
    return all(_values_equal(getattr(a, f.name), getattr(b, f.name))
               for f in fields(a))
```

placed before the trailing `try: return a == b`, plus a unit test for the
equal-but-distinct steering case and one asserting
`is_runtime_only_modification_of` is `True` for it.

**Where it lands: its own small PR, before the implementation PR.**

It is a pre-existing defect in `_config.py` with a consequence
(`is_runtime_only_modification_of`) entirely outside run-config
serialization. Bundling it into the implementation PR would put a fix for
engine-reuse behaviour inside a commit whose subject is a YAML schema,
where a reviewer looking at schema design is least likely to scrutinise
it. It also has its own regression test, which wants its own mutation
check (L-040), and that is cleaner to do in isolation.

The ordering constraint is the load-bearing part either way: **it must
land before the round-trip identity test.** In the other order, the
identity test fails on a correct serializer, and the natural reading of
that failure is "the serializer loses `steering`" — inviting a fix to the
wrong component.

---

## 8. Forward compatibility

**Commit 4.3 — `task_type`.** A probe-set-level attribute. If §10.2's
recommendation holds and probe sets are referenced by identifier, the
schema needs no change: `task_type` lives in the probe set, and the run
config points at it. If probe sets are ever inlined, it appears there.
Nothing to reserve now.

**Commit 4.5 — YAML probe-set loader.** PHASE_PLAN §5.3 carries an
explicit instruction not to invent a parallel dialect (L-035 applied to
file formats). Concretely, the two should share: `lmdiff_schema` as the
version key, the `provenance` block convention, and `__ref__` for
out-of-line payloads (§1.2). Recommend the probe-set schema reuse all
three verbatim rather than defining its own.

---

## 9. Migration

Emission is additive in behaviour — no existing artifact changes meaning
and nothing reloads differently — but it is **not** schema-neutral.
`GeoResult` gains `run_config_yaml` (§6), so `geo_schema` moves 6 → 7.

The upgrade is the cheap kind. The loader constructs by explicit key, so
a v6 file simply yields `run_config_yaml=None` and a v6 reader ignores
the new key in a v7 file. No preserving-loader path is needed, unlike
5 → 6, where the *meaning* of existing fields changed (Q9.8).

The one interaction is `GeoResult.metadata`, which today holds
`n_total_probes`, `n_skipped`, `bpb_normalized`, `max_new_tokens`,
`base_max_context`, `probe_set_name`, `probe_set_version`, plus `tasks`,
`domain_order`, `accuracy_by_variant`, `name_a`/`name_b` added by other
paths. Two of these — `max_new_tokens` and the probe-set identifiers —
are executable configuration and would appear in the run config too.
§10.1 resolves the overlap.

---

## 10. Open questions with recommendations

### 10.1 Where does the artifact attach?

**Recommend: sidecar file, plus an embedded copy in HTML only.**

`report.md` → `report.runconfig.yaml` beside it; `to_html` additionally
inlines it in a `<details>` block, because HTML's whole purpose is to
survive being emailed as one file, and a sidecar defeats that.

**The JSON carries the YAML text, but not a structured config record.**
The distinction matters and §6 works through it: an opaque verbatim
string cannot disagree with `metadata`, whereas a parallel structured
record would — that was the real objection. **Resolve the field-level
overlap by direction of truth**: the run config is the *input* record,
`metadata` is the *output* record. Where they overlap (`max_new_tokens`,
probe-set identity) the run config is authoritative and `metadata` keeps
its copy for existing readers.

Carrying it is also what makes `to_html()` work on a reloaded result
(§6). Without it, re-rendering a report from saved JSON silently drops
the provenance this commit exists to add.

A user holding only the HTML has everything. A user holding only the
JSON has the numbers and is told, in `provenance`, which version produced
them — which is the question the JSON alone needs to answer.

**What "attach" means when the artifact is a bundle.** §1.4 establishes
that a config carrying `__ref__` is a YAML plus a sidecar directory. That
changes each option differently, and the differences favour the
recommendation:

| option | with a bundle |
|---|---|
| sidecar YAML | already a sibling file; the `.d/` directory joins it. No change in kind |
| embedded in HTML | **cannot embed a `.npy`.** The HTML carries the YAML text, and any `__ref__` it names is unresolvable from the HTML alone |
| field in the JSON | same problem, plus the overlap objection above |

So the HTML embed is *self-describing but not always self-executing* —
a reader can always see what the run was, and can re-run it only when the
config carries no `__ref__`. For every experiment lmdiff has run to date
that is every config, so the practical answer today is "always".

The honest way to express that is in the embed itself: when a `__ref__`
is present, the HTML block carries a line saying the config is part of a
bundle and names the directory. A reader then knows to go find it rather
than discovering it from a load error. That costs one conditional line
and removes the only case where the embedded copy would silently mislead.

**What is a `__ref__` relative to, inside an embed?** Nothing — and the
loader must say so rather than guess. §1.4 defines `__ref__` as relative
to the YAML *file*; an embed is text inside a document and has no such
anchor. Resolving it against the HTML's own directory would be a guess
that silently succeeds when an unrelated file of the right name happens
to sit there, which is worse than failing.

So: **YAML extracted from an HTML embed is readable but not runnable
whenever it carries a `__ref__`**, and a loader handed such text refuses
with the same `RunConfigError` shape as a dangling reference, naming the
bundle directory recorded at emission. The user's remedy is to fetch the
sidecar bundle and load the YAML from disk, where the anchor exists. The
directory name in the embed is a *hint for a human*, not a resolvable
path — recording it as an absolute path at emission would be worse still,
since it names a filesystem the reader is probably not on.

### 10.2 What identifies a probe set?

**Recommend: identifier string, plus a content hash when the probe set is
user-supplied.**

```yaml
probes: lm_eval:hellaswag+arc_challenge      # built-in: identifier alone
probes:                                       # user-supplied (commit 4.5)
  path: ./probes/my_set.yaml
  sha256: 3f2a...
```

Built-ins are reproducible from the identifier plus the pinned lmdiff and
lm-eval versions — that is what `provenance` is for. Path alone breaks
across machines; hash alone cannot reconstruct; inlining bloats a
100-probe set beyond readability and defeats §3.2's readability argument.
Path-plus-hash lets a loader say *"this is not the probe set that
produced the report"* precisely, which is the failure worth catching.

Emission writes the identifier for built-ins and both keys for
user-supplied, once 4.5 exists. Until then, only the identifier form is
reachable.

### 10.3 How much provenance?

**Recommend six fields**, per §4: `lmdiff`, `python`, `torch`,
`transformers`, `run_id`, `geo_schema`. Drop `duration_s`, `devices`,
`probes_excluded`, `domain_status_summary` — the last two because the
GeoResult already holds them exactly, and duplication is the recurring
failure here.

### 10.4 CLI surface

**Recommend `--config` on the existing subcommands**, not a new
subcommand.

`lmdiff family --config run.yaml` and
`lmdiff family --base X --variant y=Y` produce the same thing by
different routes; they are one command with two input forms. A separate
`lmdiff run` would need every reporting flag duplicated onto it, and the
two would drift — the same argument as §8.

Where both are given, explicit flags override the file, with a warning
naming each overridden key. That is the "copy it and change one thing"
path from AA.2, done without editing the file.

### 10.5 Schema versioning — are three numbers one too many?

Three exist: `lmdiff_schema` (this file), `geo_schema` (currently 6), and
the package version.

**Recommend keeping them separate. They cannot be collapsed, and the
reason is asymmetry.**

`geo_schema` versions an *output* the loader must interpret; it has moved
six times, driven by fields being added to results. `lmdiff_schema`
versions an *input* a loader must accept; it will move when the run-config
grammar changes. These change for unrelated reasons and at unrelated
rates — v0.4.1 alone moved `geo_schema` 5 → 6 while the run-config
grammar did not exist yet. Tying them means every result-shape change
invalidates every stored run config, which is exactly backwards for a
file whose value is that old ones still run.

Collapsing either into the package version fails for the reason AA.3
gives about defaults: the package version moves constantly and for
reasons unrelated to either format.

**What each governs, and what a consumer does on disagreement:**

| number | governs | changes when |
|---|---|---|
| `lmdiff_schema` | run-config grammar — which keys exist and what they mean | a key is added, removed, or re-interpreted |
| `geo_schema` | GeoResult field shape | a result field is added or its meaning changes (6 times so far) |
| `provenance.lmdiff` | numeric semantics | every release |

**What a consumer does when they disagree.** `lmdiff_schema` is the only
one that is asymmetric, and that asymmetry is the point:

| condition | behaviour | why |
|---|---|---|
| `lmdiff_schema` **>** loader's | **refuse**, naming the file's version and the loader's | an unknown key may be load-bearing. Silently ignoring it can change results, which is the failure AA.3 exists to prevent |
| `lmdiff_schema` **<** loader's | **accept and upgrade**, warning once if any default had to be supplied | old configs continuing to run is the artifact's entire value. A newer loader knows what the older grammar meant |
| `lmdiff_schema` **==** loader's | proceed silently | — |
| `geo_schema` mismatch | existing v1–v6 upgrade paths, unchanged | out of scope for this commit; the run config does not read the GeoResult |
| `provenance.lmdiff` differs | **warn and proceed** (AA.3) | re-running an old config on a new version is legitimate and common; doing it silently is not |

The distinction between the first row and the last is what someone will
need: **`lmdiff_schema` too high means *I cannot read this file*;
`provenance.lmdiff` differing means *I can read it and the numbers may
differ.*** One is a parse error, the other is a caveat. Conflating them
either blocks a legitimate re-run or silently accepts a file it does not
understand.

Note that `provenance` is dropped on load (AA.2), so the version warning
must be emitted *while* parsing, before the block is discarded — a small
ordering constraint on the loader that is easy to get wrong.

### 10.6 `min_valid_fraction` — resolved: plumb it onto the public API

**The collision.** AA.2's settled example lists `min_valid_fraction: 0.5`
as executable configuration, and AA.3 requires every parameter affecting
a computed value to be written explicitly. It unambiguously affects
computed values — it is what turns a domain's share to `None`. But it is
not a parameter of `family()` or `compare()`; it stops at
`run_family_pipeline` (v0.4.1 commit 1), and PHASE_PLAN Z.5 deferred
exposing it publicly to v0.5.0+ on the grounds that no user had asked.

**Resolution: add `min_valid_fraction` to `family()` and `compare()` in
v0.4.3.**

AA.3 exists because omission fails to pin. `min_valid_fraction` is the
parameter whose default most recently changed meaning in this project —
from an implicit `1/n` to an explicit `0.5` — and that change is what
decides whether long-context reads 27.6 % or `—`. A config artifact that
cannot pin it fails at exactly the case it was built for.

Of the three options considered, it is the only one that does not
knowingly weaken a settled guarantee. Emitting it under `provenance`
would misfile an input as a runtime fact and let a loader silently ignore
it — precisely the failure AA.3 prevents. Omitting it leaves the artifact
under-specified for the one parameter with a documented history of
changing meaning.

**Not really scope expansion.** Z.5's deferral rested on "no user has
asked". The schema is now asking. The cost is one kwarg on two functions
plus a passthrough to `run_family_pipeline`, which already accepts it.

**Precedence follows the seed pattern** (L-031): explicit call argument >
default. There is no per-variant tier here — unlike `DecodeSpec.seed`,
`min_valid_fraction` is a property of the run, not of a variant — so the
chain is two levels, not three:

```
family(min_valid_fraction=0.6)   ->  0.6
family()                         ->  DEFAULT_MIN_VALID_FRACTION (0.5)
```

Emission always writes the effective value, defaulted or not, per AA.3.

**Plan bookkeeping.** PHASE_PLAN Z.5's row *"`min_valid_fraction` on the
public API → v0.5.0+"* is superseded by this commit. Its stated reason no
longer holds, and the row should say so rather than being deleted — the
deferral was correct when made.

---

## Appendix — reproducing this audit

```bash
mamba run -n lmdiff python docs/internal/v043_roundtrip_check.py
```

Stage A: eight calibration Configs, YAML round-trip.
Stage B: maximal Config; surfaces the `steering` comparator defect.
Stage C: per-field YAML classification (the §1.1 table).
Stage D: what a fully-defaulted `Config` emits — 15 keys, 12 of them
`None` — which is the concrete form of AA.3's "emitted YAML is verbose
and that is correct".
