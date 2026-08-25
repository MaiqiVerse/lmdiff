# Probe sets

A **probe set** is an ordered, immutable collection of prompts that
every configuration in a comparison sees. It is the shared input that
makes the comparison a comparison.

Each probe carries up to three labels, and they answer different
questions:

| label | question | used for |
|---|---|---|
| `domain` | what subject does this exercise? | grouping for normalization — the per-domain drift and share tables |
| `output_type` | how is the model queried? | free generation vs scoring supplied choices |
| `scoring` | how is the output judged? | which evaluator runs |

All three are optional. `None` means **unlabelled**, which is different
from any value: a probe with no `scoring` has not been assigned the
default evaluator, it has been left to whoever runs the task.

## Getting one

```python
import lmdiff
from lmdiff.probes.loader import ProbeSet

# the bundled quick set — 90 completion probes, three domains
result = lmdiff.family(base="gpt2", variants={"v": "distilgpt2"})   # default
result = lmdiff.family(..., probes="v01")                           # the same

# an lm-evaluation-harness task, or several
result = lmdiff.family(..., probes="lm_eval:hellaswag+arc_challenge", n_probes=100)

# your own
probes = ProbeSet.from_json("my_probes.json")
result = lmdiff.family(..., probes=probes)
```

`n_probes` applies **per task** for `lm_eval:` specs — `n_probes=100`
against a five-task string yields 500 probes, not 100.

> **Known limitation.** Passing a `ProbeSet` *instance* means the run
> configuration cannot be emitted, and the run warns and continues
> without one. Pass `"v01"` or an `lm_eval:` identifier if you need the
> provenance sidecar. Tracked for a future release.

## The file format

```json
{
  "name": "my_probes",
  "version": "1.0",
  "probes": [
    {
      "id": "math_001",
      "text": "17 + 25 = ",
      "domain": "math",
      "output_type": "generate_until",
      "scoring": "exact_match",
      "expected": "42"
    }
  ]
}
```

`id` and `text` are required. Everything else is optional and is omitted
from the file when unset, so a minimal probe is two keys.

`metadata` is a free dict. Multiple-choice probes use it for `choices`
and `correct_index`; the `lm_eval` adapter also stores `task_name`,
`native_metric` and `requires_execution` there.

## `output_type`

lm-evaluation-harness's vocabulary, unchanged:

| value | meaning |
|---|---|
| `generate_until` | the model generates freely; the text is judged |
| `multiple_choice` | each supplied choice is scored by log-likelihood; the best is the answer |
| `loglikelihood` | a single continuation is scored |
| `loglikelihood_rolling` | perplexity over a document |

The values are copied rather than translated on purpose. They come out
of lm-eval task configs, and a second set of names for the same four
things is how one quantity ends up with two vocabularies.

An unrecognised value **warns at load and is kept**. Nothing downstream
branches on membership, so refusing to load would cost more than it
protects.

## `scoring`

Which evaluator judges the output. Five ship:

| `scoring` | rule | needs |
|---|---|---|
| `exact_match` | output equals `expected`, whitespace stripped | `expected` |
| `contains_answer` | `expected` occurs anywhere in the output | `expected` |
| `multiple_choice` | parse a letter or integer from the output | `metadata["correct_index"]` |
| `f1` | SQuAD token-overlap ≥ 0.5 | `expected`, optionally `metadata["aliases"]` |
| `gsm8k_number_match` | last number after `####`, compared numerically | `expected` |

`scoring` is **open-ended**: the registry is the vocabulary, so a name
this version does not ship warns at load and falls back at run time to
whatever evaluator the caller passed. Nothing raises.

### Choosing one

Match the rule to the answer's shape, not to the subject:

- The answer is the whole output → `exact_match`.
- The answer is embedded in a sentence → `contains_answer`, but only if
  `expected` is long enough to be distinctive. A one- or two-character
  expected value is a substring of almost any output, and
  `contains_answer` will report it correct.
- The answer is a span to be compared loosely → `f1`.
- The answer is a number reached by working → `gsm8k_number_match`.

Mixed sets are fine and are the reason this is a per-probe field: one
probe set can hold `exact_match` and `f1` probes and each is judged by
its own rule.

## Accuracy and what it rests on

An evaluator can fail in two ways, and they mean opposite things:

- **The output was wrong.** The rule applied and the model missed. This
  counts against the model.
- **The rule could not apply.** The probe has no `expected`, or a
  multiple-choice rule met a probe with no `correct_index`. This says
  nothing about the model.

Probes in the second category are **unscorable**. They are excluded
from the accuracy denominator rather than counted wrong, because
counting them wrong penalises the model for a defect in the probe set.

Every surface that shows an accuracy also shows the count it rests on:

```
          Per-Domain Accuracy
┌───────────┬────┬─────────────────────┐
│ Domain    │  N │ Acc(A) (scorable/n) │
├───────────┼────┼─────────────────────┤
│ code      │ 30 │      43.33% (30/30) │
│ math      │ 30 │      50.00% (18/30) │
│ knowledge │ 30 │          n/a (0/30) │
└───────────┴────┴─────────────────────┘
42 probe-evaluations could not be judged and are excluded from the
accuracies above — the evaluator does not fit those probes (no
`expected`, or missing multiple-choice metadata). Set each probe's
`scoring` field, or pass a different fallback evaluator.
```

`n/a` is not zero. It means nothing in that domain could be judged, and
a `0.00%` there would be a claim about the model that the data does not
support.

In Python and in JSON the same holds: `accuracy` is `None` rather than
`0.0`, and `n_scorable` / `n_unscorable` sit beside it in the same
object.

```python
tr = task.run(engine)
tr.accuracy       # float | None
tr.n_scorable     # what the accuracy divides by
tr.n_unscorable   # probes this evaluator could not judge
```

## The bundled set: `v01`

90 probes, three domains of 30: `code`, `knowledge`, `math`. Fast, no
download, meant for checking that a pipeline runs and for catching
obvious anomalies — not for headline numbers.

Every probe is **completion-style**: the prompt ends mid-sentence and
the model continues it.

```
"17 + 25 = "                      → "42"
"The capital of France is "       → "Paris"
"import numpy as "                → "np"
```

This matters. Base models have no instruction tuning, and an
instruction-style probe (*"What is 17 + 25? Answer with just the
number."*) makes them echo the prompt or emit newlines — which measures
the distance between two degenerate output modes rather than any
difference in capability. v01's math probes were instruction-style once
and were rewritten for exactly this reason. Instruction-style probes
belong in their own versioned file.

`v01` declares `output_type: generate_until` and **deliberately leaves
`scoring` unset**, so the caller's evaluator applies to all 90. Its
probes are *prefix* completions — the answer is the immediate
continuation — and none of the five shipped rules expresses that: the
short expected values (`"n"`, `"i"`, `"["`) make `contains_answer`
match almost anything, and `exact_match` rejects every continuation that
runs on past the answer. Labelling them with a rule known to misfire
would be worse than leaving the choice explicit.

## Writing your own

```python
from lmdiff.probes.loader import Probe, ProbeSet

ps = ProbeSet(
    [
        Probe(id="q1", text="The tallest mountain is ", domain="geography",
              output_type="generate_until", scoring="contains_answer",
              expected="Everest"),
    ],
    name="my_probes", version="1.0",
)
ps.to_json("my_probes.json")
```

Domains are free strings — there is no fixed list, and one is not
imposed. The nine that appear in lmdiff's lm-eval mapping (`code`,
`commonsense`, `knowledge`, `language`, `long-context`, `math`,
`reading`, `reasoning`, `safety`) are a convention, not a schema.

Use at least two domains if you want the per-domain figures to say
anything; a single-domain set produces one bar.

### Inspecting a set

```python
ps.domains                    # ['code', 'knowledge', 'math']
ps.output_types               # ['generate_until']
ps.scorings                   # [] when nothing declares one
ps.by_domain()                # {'math': ProbeSet(...), ...}
ps.by_output_type()           # unlabelled probes bucket as 'unknown'
ps.filter(domain="math")
ps.filter(scoring="f1")
```

`ProbeSet` is immutable once loaded. `filter` and slicing return new
sets; nothing mutates in place.

## Where the labels end up

A `GeoResult` records the labels of every probe that survived, aligned
with `change_vectors`:

```python
result.probe_domains        # ('math', 'math', 'code', ...)
result.probe_output_types   # ('generate_until', ...)
result.probe_scoring        # ('exact_match', None, ...)
```

They are stored because a result outlives the probe set that produced
it — an in-memory set is gone at process exit, and an `lm_eval:`
identifier resolves through a table that changes between versions. Any
result saved before v0.4.4 loads with the two new tuples empty.
