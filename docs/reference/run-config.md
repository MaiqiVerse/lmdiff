# Run configuration

Every report lmdiff writes is accompanied by a **run configuration** — a
YAML file recording the call that produced it.

A report is otherwise a container for numbers. Reading
`CodeLlama → code 52.1%`, there is no way to recover which variants ran,
which probe set, what `n_probes`, which seed, or whether
`min_valid_fraction` was defaulted or overridden. The run config is that
record.

It also gives *"why do I get different numbers than you"* a mechanical
answer. Between v0.3.2 and v0.4.1 the meaning of `share_per_domain`
changed twice — once by a formula correction, once by the validity
framework. `lmdiff: 0.4.3` in the artifact is the answer.

> **Emission only, for now.** v0.4.3 *writes* these files. A loader that
> *runs* one is a later release. The schema is designed for both, because
> the emitted artifact has to be a valid input — see
> [Why it is runnable](#why-it-is-runnable).

## Where to find it

| you have | the config is |
|---|---|
| `report.md` | `report.runconfig.yaml`, beside it |
| `report.html` | beside it **and** inlined in the report, under "Run configuration" |
| only the JSON | in the `run_config_yaml` field |

The sidecar is named after the report's **stem**, not its format. Render
`report.md` and `report.html` into the same directory and you get one
`report.runconfig.yaml` — they describe the same run, so the content is
identical and the second write is a no-op. Give the reports different
stems if you want two files.

HTML additionally embeds a copy because HTML's purpose is surviving as a
single file — emailed, uploaded, pasted into a wiki — and a sidecar
defeats that. Markdown gets the sidecar only.

Writing is automatic. There is no flag to enable it: an artifact you have
to remember to request is one you will not have when you need it.

## What it looks like

```yaml
lmdiff_schema: 1
base: meta-llama/Llama-2-7b-hf
variants:
  yarn: NousResearch/Yarn-Llama-2-7b-128k
  temp_1.5:
    model: meta-llama/Llama-2-7b-hf
    decode:
      strategy: sample
      temperature: 1.5
      top_p: 1.0
      top_k: 0
      num_samples: 1
      max_new_tokens: 16
      seed: null  # inherits the family seed: 42
probes: lm_eval:hellaswag+arc_challenge
n_probes: 100
metrics:
- bd
- drift
- share
- direction
- specialization_zscore
max_new_tokens: 16
task_overrides:  # per-task values below override the top-level max_new_tokens
  gsm8k:
    max_new_tokens: 256
seed: 42
min_valid_fraction: 0.5
reproducible: true
provenance:
  lmdiff: 0.4.3
  python: 3.12.13
  geo_schema: '7'
  torch: 2.11.0+cu130
  transformers: 5.5.4
  lm_eval: 0.4.11
```

## The keys

### Executable configuration

Everything above `provenance` is the run itself.

| key | meaning |
|---|---|
| `lmdiff_schema` | grammar version of *this file* — see [Three version numbers](#three-version-numbers) |
| `base` | the base configuration |
| `variants` | `{name: configuration}` |
| `probes` | probe-set identifier |
| `n_probes` | probes drawn per task |
| `metrics` | the resolved metric list |
| `max_new_tokens` | generation cap |
| `task_overrides` | per-task overrides; these win over the top-level value |
| `seed` | top-level RNG seed |
| `min_valid_fraction` | floor below which a domain reports no share |
| `reproducible` | whether this file can be run at all |

**`base` and each variant** are either a bare model identifier — meaning
"this model, everything else default" — or a mapping expanding the full
`Config`. The short form exists because a seven-variant run written out
in full is five near-identical `decode` blocks burying the two variants
that actually differ.

**`metrics` is the resolved list, never the word `default`.** `default`
names something that moves between versions; the point of the artifact is
that a reader can tell what was used.

### `provenance` — read-only

Six fields, plus `lm_eval` when the probe set is an `lm_eval:`
identifier, because that package determines probe text, splits and
ordering and is an optional dependency.

**A loader ignores this block entirely.** It records how the run
happened, not what to do.

Deliberately absent: probe-exclusion counts and per-domain status
summaries. The `GeoResult` beside this file already holds both, exactly,
and two copies of one fact are two facts that can disagree.

## Why it is runnable

There is one schema, and the emitted file is a valid input. Someone
reading a report will copy the config, change two lines and run it — so
that path works, rather than there being a terse input format and a
separate verbose output snapshot that cannot be fed back in.

**Defaults are written out.** Every parameter affecting a computed value
appears explicitly, whether or not you passed it. Pinning by omission
fails the moment a default changes, and lmdiff has changed a numeric
default's effective meaning twice inside two minor versions. Emitted
files are verbose; hand-written ones may omit anything with a default.

**A version mismatch will warn, not block.** Re-running an old config on
a newer lmdiff is a legitimate thing to want. Doing it without being told
is not.

## `reproducible` and `non_serializable`

```yaml
reproducible: false
non_serializable:
  - path: variants.custom_ft
    reason: in-memory model object, not a resolvable identifier
```

`reproducible: false` means **no possible file could express this
configuration** — you passed something Python can hold and YAML cannot
name, such as a model object rather than an identifier. A loader refuses
and names the offending paths rather than pretending.

It does *not* mean "awkward to serialize", and it does not mean a
referenced file is missing. Nothing in lmdiff currently triggers it.

## Bundles

Large arrays — soft prompts, steering vectors — are not inlined. A
32-layer steering set is megabytes of float literals inside a file whose
purpose is being read. They are referenced instead:

```yaml
soft_prompts: {__ref__: soft_prompts.npy, dtype: float32, shape: [2, 2]}
```

A config carrying a `__ref__` is a **bundle**: the YAML plus a sibling
directory holding the payloads. Copy both. The rule is that everything
needed to *understand* the run is in the YAML, and everything needed to
*re-run* it is in the bundle.

Two consequences worth knowing:

- **A `__ref__` inside the HTML embed resolves against nothing.** An
  embed is text in a document; it has no directory. The embedded copy is
  for reading, and says so when a `__ref__` is present.
- **A missing payload is not `reproducible: false`.** That flag is a
  property of the configuration, decided when it was written. A missing
  file is a property of your copy, and is fixed by fetching the bundle.

No shipped lmdiff experiment uses arrays, so in practice every config is
a single file today.

## Three version numbers

| number | versions | when it moves |
|---|---|---|
| `lmdiff_schema` | the grammar of this file | a key is added, removed or re-interpreted |
| `provenance.geo_schema` | the `GeoResult` JSON shape | a result field is added or changes meaning |
| `provenance.lmdiff` | the numbers themselves | every release |

They are separate because they change for unrelated reasons at unrelated
rates. v0.4.1 moved `geo_schema` 5 → 6 while this file's grammar did not
yet exist. Tying them would mean every change to result shape invalidated
every stored run config — backwards for a file whose value is that old
ones still run.

The distinction that matters when they disagree: **`lmdiff_schema` too
high means the file cannot be read**; **`provenance.lmdiff` differing
means it can be read and the numbers may differ.** One is a parse error,
the other a caveat.

## In Python

```python
result = lmdiff.family(base="gpt2", variants={"v": "distilgpt2"})
print(result.run_config_yaml)      # the text, verbatim
result.to_markdown("report.md")    # writes report.runconfig.yaml too
```

`run_config_yaml` is `None` for results built any other way, and for any
result loaded from a save written before v0.4.3.
