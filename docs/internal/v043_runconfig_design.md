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
7. [Round-trip guarantee and how it is tested](#7-round-trip-guarantee)
8. [Forward compatibility with commits 4.3 and 4.5](#8-forward-compatibility)
9. [Migration and existing artifacts](#9-migration)
10. [Open questions with recommendations](#10-open-questions)

---

## 0. Headline findings

Four things the investigation turned up that change the shape of the work.

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
fault.** This is the audit's substantive finding, and it is a
**pre-existing bug outside this commit's scope**. Detail in §7.2. Not
fixed here, per the audit constraints; recommendation is there.

**0.4 — [QUESTION] AA.2's worked example contains a parameter the public
API cannot accept.** `min_valid_fraction` appears in the settled example
YAML as executable configuration. It is not a parameter of `family()` or
`compare()` — it stops at `run_family_pipeline`, and Z.5 defers exposing
it to v0.5.0+. Two settled decisions collide. Detail and options in
§10.6; flagging rather than designing around it, per the instruction.

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

**`min_valid_fraction` is the exception, and it is a problem.** It
affects computed values as directly as anything on this list — it is what
turns a domain's share to `None` — but it is not a parameter of either
public entry point. See §10.6.

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
metrics: default
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
| `duration_s` | no | diagnostics, not provenance; varies with hardware and tells you nothing about the numbers |
| `devices` | no | ditto. A reader wanting this has the run log |
| `probes_excluded` | **no** | `GeoResult.metadata["n_skipped"]` already holds it |
| `domain_status_summary` | **no** | `GeoResult.domain_status` holds it exactly, per cell |

The last two are the ones AA.8 question 3 flags, and they are the clearest
cuts: both are already in the artifact this file ships beside.
Duplicating them creates two copies free to disagree — the failure this
project has hit three times (L-035).

This trims AA.2's illustrative block from nine fields to six.

---

## 5. The escape hatch

Ships as specified in AA.5, with no current occupant (§1.1).

```yaml
reproducible: false
non_serializable:
  - path: variants.custom_ft
    reason: in-memory model object, not a resolvable identifier
```

**Criterion for setting it:** a value that cannot be reconstructed from
the file alone. Not "awkward to serialize" — `soft_prompts` is awkward and
is still fully reproducible. The distinction matters, because the
temptation will be to use `reproducible: false` for anything inconvenient,
and that would make the flag meaningless.

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
**This is the one structural addition the commit requires**: either
`GeoResult` gains a `run_config` field, or emission happens at the
`_api.family()` level where the inputs are still in scope.

Recommend the latter for v0.4.3 — `_api` already has everything, and it
avoids widening `GeoResult` before §10.1 settles where the artifact
attaches. If §10.1 chooses the JSON-embedded option, this changes.

---

## 7. Round-trip guarantee

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

**Recommendation, not applied here** (audit constraint): add a
dataclass branch to `_values_equal`, recursing field-by-field, and a unit
test for the equal-but-distinct steering case. Two lines and a test. It
should land **before** the round-trip identity test, or that test will
fail on a correct serializer and invite someone to "fix" the serializer.

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

Emission is purely additive (AA.6): no existing artifact changes, no
schema version moves, nothing reloads differently. `GeoResult` schema
stays at 6.

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

Not the GeoResult JSON. The JSON is consumed programmatically and
already carries `metadata`; adding a second configuration record there
creates exactly the two-copies-free-to-disagree problem. **Resolve the
overlap by direction of truth**: the run config is the *input* record,
`metadata` is the *output* record. Where they overlap
(`max_new_tokens`, probe-set identity) the run config is authoritative
and `metadata` keeps its copy for existing readers.

A user holding only the HTML has everything. A user holding only the
JSON has the numbers and is told, in `provenance`, which version produced
them — which is the question the JSON alone needs to answer.

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

| number | governs | on mismatch |
|---|---|---|
| `lmdiff_schema` | run-config grammar | **refuse** if higher than the loader knows — an unknown key may be load-bearing |
| `geo_schema` | GeoResult field shape | existing v1–v6 upgrade paths, unchanged |
| `provenance.lmdiff` | numeric semantics | **warn and proceed** (AA.3) — re-running old configs on new versions is the point |

The distinction between the first and third is the useful part:
`lmdiff_schema` mismatch means *I cannot read this file*;
`provenance.lmdiff` mismatch means *I can read it and the numbers may
differ.* One is a parse error, the other is a caveat.

### 10.6 [QUESTION] `min_valid_fraction` is in the settled example but not on the public API

AA.2's example lists `min_valid_fraction: 0.5` as executable
configuration, and AA.3 requires every parameter affecting a computed
value to be written explicitly. `min_valid_fraction` unambiguously
affects computed values — it is what turns a domain's share to `None`.

But it is not a parameter of `family()` or `compare()`. It stops at
`run_family_pipeline` (v0.4.1 commit 1), and PHASE_PLAN Z.5 defers
exposing it publicly to v0.5.0+ with the note *"no user has asked"*.

So the schema would emit a key that no public entry point can consume,
and a loader would have to either drop it — breaking AA.3's guarantee for
the one parameter whose default most recently changed meaning — or reach
past the public API into `_pipeline`.

Three options, none of which I should pick unilaterally:

1. **Plumb `min_valid_fraction` onto `family()`/`compare()` in v0.4.3.**
   Small (one kwarg, two call sites, passthrough), makes the schema
   honest, and pulls a v0.5.0 item forward. Scope expansion.
2. **Emit it under `provenance`** rather than as executable config.
   Honest about the current API, but it is not provenance — it is an
   input, and a loader ignoring it silently changes results if the
   default moves. This is precisely the failure AA.3 exists to prevent.
3. **Omit it entirely** until v0.5.0 exposes it. Simplest, and leaves the
   artifact under-specified for exactly the parameter with the most
   recent history of changing meaning.

I lean to (1) — it is the only option that does not knowingly weaken a
settled guarantee, and the cost is genuinely one kwarg. But it expands
scope, and the instruction is to stop rather than design around a
collision between settled decisions.

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
