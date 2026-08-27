# v0.4.5 — the task layer's engine port: scope

PHASE_PLAN Z.4 item 6, pulled out of v0.5.0 into its own release.

Two halves of one thing: porting `Task` to the `Engine` Protocol is what
puts evaluation metrics back on the live path, because the evaluators
are already engine-free.

Investigation only. Everything below came from a script or from reading
the two engine implementations side by side; scripts in §6.

---

## 1. The port itself

### 1.1 The multi-prompt signature expresses no capability

This was the thing worth checking, and the answer is clean: **both
`generate` implementations are per-prompt loops at batch size 1.**

```python
# InferenceEngine.generate(prompts: list[str], ...)   — engine.py:263
for probe in _progress_iter(prompts, ...):
    input_ids = torch.tensor([full_ids], device=self.device)
    outputs = self._model.generate(input_ids=input_ids, ...)

# HFEngine.generate(prompt: str, ...)                 — _engine.py:769
input_ids = torch.tensor([full_input_ids], device=device)
output = self._model.generate(input_ids, ...)
```

The list signature is a convenience wrapper around a loop, not batching.
Moving the loop from inside the engine to inside `Task.run` costs
nothing — no throughput, no numerics. **The Protocol can express
everything `Task` asks of it.**

Three things the list signature does carry that the single-prompt one
does not, and none of them survives contact with the callers:

- **`n_samples`** → `num_return_sequences`. Every caller in `tasks/` and
  in `metrics/` passes `n_samples=1`, and `ChangeGeometry` hard-codes it
  with a comment saying why. Only `ModelDiff` and the CLI's `compare`
  default to 5, and `ModelDiff` is on the v0.5.0 removal list. **The
  task layer needs no equivalent.**
- **The progress bar.** `_progress_iter` wraps the loop inside the
  engine. Move the loop and the bar moves with it — `lmdiff._progress`
  is engine-free, so this is a relocation, not a loss.
- **`system_prompt` / `context` / `decode` runtime overrides.** This one
  is not cosmetic. §1.2.

### 1.2 The trap: a naive port silently drops the configuration

`InferenceEngine` resolves scaffold and decode **from its stored
config** when the caller passes nothing:

```python
# engine.py:198 — _encode_for_model
prefix_text = self._prefix_text(system_prompt=system_prompt, context=context)
#   "system_prompt / context default to this engine's stored config"
# engine.py:251 — _decode_params(decode) -> self.config.decode
```

`HFEngine` is **stateless** about both. The caller supplies
`prefix_text=` and explicit `temperature` / `top_p` / `top_k`.

So the obvious translation

```python
engine.generate(texts, n_samples=1, max_new_tokens=N)
#  ->  [engine.generate(t, max_new_tokens=N).text for t in texts]
```

drops the configuration on the floor. Measured with the two helpers the
live pipeline already uses, both pure functions of a `Config`:

| config | `_prefix_text` | `_generate_kwargs` | what a naive port loses |
|---|---|---|---|
| plain | `''` | `{max_new_tokens: 16}` | nothing |
| `system_prompt` | `'You are a helpful assistant.\n'` | `{max_new_tokens: 16}` | **29-char scaffold not prepended** |
| `temp_1.5` | `''` | `{max_new_tokens: 16, temperature: 1.5, top_p: 0.95, top_k: 0}` | **sampling → greedy** |

The second row is the serious one. `HFEngine.generate` computes
`do_sample = (temperature != 1.0) or (top_p < 1.0)`, so the defaults make
it greedy. **A temperature-1.5 variant's accuracy would be measured
under greedy decoding** — measuring a different configuration than the
one the report names, which is the entire premise of the tool.

Neither failure raises. Both produce numbers.

### 1.3 What the port actually needs

`_prefix_text(config)` and `_generate_kwargs(config, max_new_tokens)`
already exist — module-private in `_pipeline.py`. `Task` needs both, and
`Task` holds no `Config`.

Three options, and only one avoids a second copy:

1. **Move both helpers to a shared home** and have `_pipeline` and
   `tasks/` import them. `lmdiff/_prompting.py` or similar — engine-free,
   `Config`-only, so it violates no layering rule.
2. Import across `_pipeline`'s private boundary. Works, reads as a
   mistake, and invites someone to "fix" it by copying.
3. Write a second copy. This is L-035 exactly, and the note in
   `_prefix_text`'s docstring — *"keeping the trailing `\n` matters for
   byte-equivalence with the v0.2.x calibration baseline. Don't strip"* —
   is precisely the kind of detail a second copy loses.

**Recommend (1).** It is the only part of this port that is design rather
than mechanics, and it is small.

### 1.4 The line count, measured

I estimated ~15. Measured, it is **19 engine-touching lines**, and they
split into two jobs rather than one:

| file | lines | Protocol gaps |
|---|---|---|
| `tasks/base.py` | 3 | `.model_name` |
| `tasks/loglikelihood.py` | 3 | `.model_name` |
| `tasks/capability_radar.py` | 5 | `.model_name` ×3 |
| **subtotal — the port proper** | **11** | one attribute rename |
| `metrics/output/behavioral_distance.py` | 8 | `.config` ×2, `.tokenizer` ×2 |

The `tasks/` eleven are genuinely mechanical: `.model_name` → `.name`,
`generate(list)` → a loop, `score(prompts, continuations=…)` → a loop
over `score(prompt, continuation)`. Plus §1.3's helper move and the
progress-bar relocation.

`experiments/family.py` shows zero because it never calls an engine
directly — it constructs `InferenceEngine(vcfg)` and hands it to `Task`.
Which means **that construction is the only thing tying the deprecated
accuracy path to the deprecated engine**, and once `Task` is ported, the
same code works with an `HFEngine`.

---

## 2. What else speaks the old surface

I said "Task, the five evaluators, loglikelihood_accuracy, the radar and
`_accuracy_for_task`". Two corrections:

**The evaluators do not.** All six are pure functions of
`(output, expected, metadata)` and touch no engine at all. They are the
reason this port is cheap, and they need no changes. `_accuracy_for_task`
likewise touches no engine — it builds a `Task` and calls `.run()`.

**`BehavioralDistance` does, and I did not list it.** It is in
`metrics/`, not `tasks/`, and it is reachable only from
`CapabilityRadar.run_pair`, `ModelDiff` and the CLI's `compare`.

Its four non-Protocol lines are more interesting than the count
suggests:

```python
same_tok = engine_a.config.shares_tokenizer_with(engine_b.config)
if same_tok is None:
    same_tok = tokenizers_equivalent(engine_a.tokenizer, engine_b.tokenizer)
```

`.tokenizer` is a **model object**, and CLAUDE.md's first
non-negotiable rule is that metrics receive engine outputs, never model
objects. So BD is already violating that rule, through the legacy engine,
today.

The Protocol anticipated it: `tokenizers_equivalent_to(other)` exists,
with a docstring naming the exact case BD hand-rolls (*"handles
slow/fast tokenizer L-011 case where vocab_size differs but produces
identical token ids"*). **Those four lines collapse to one call**, and
the port makes BD *more* compliant, not less.

So the surface is smaller and cleaner than I reported. What I got wrong
was scope, not direction.

---

## 3. What changes when `accuracy_by_variant` comes back

You were right to ask. **One finding fires unconditionally, and one
becomes permanently dead.**

### 3.1 `BaseAccuracyMissingFinding` fires on every run

```python
# _findings.py:487
base_acc = meta.get("base_accuracy") or meta.get("accuracy_base")
variants_with_accuracy = [k for k, v in acc_by_variant.items() if …]
if variants_with_accuracy and not base_acc:
    findings.append(BaseAccuracyMissingFinding(…))
```

**Nothing in `lmdiff/` ever sets `base_accuracy` or `accuracy_base`.**
Grep finds exactly two hits: this read, and a test fixture that
constructs it by hand to check the negative case. And the deprecated
accuracy phase loops `for vname, vcfg in variant_cfgs.items()` — it
never scores the base engine either.

The finding is dormant only because `_extract_accuracy_findings` bails
at line 432 when `acc_by_variant` is empty, which it always is on the
live path. **Restore accuracy and every report grows a permanent caveat**
saying "Base accuracy not measured; Δaccuracy comparison skipped" —
true, and useless, on every run forever. A caveat that fires always
carries no information, which is the L-039 shape from the other
direction: not a claim without its condition, but a condition that is
always met.

**The fix is to score the base engine, and the live path is where that
becomes cheap.** `run_family_pipeline` already holds `base_engine` in
scope for the whole run — it is the thing every δ is computed against.
The deprecated path could not do this without a sixth `InferenceEngine`
construction; the live path gets it for one extra pass over the probes.
Then `base_accuracy` is populated, the finding fires only when base
scoring genuinely failed, and Δaccuracy — which the finding's own text
says was the point — becomes computable.

### 3.2 `AccuracyArtifactFinding` needs task names the live path lacks

```python
_GENERATIVE_TASKS = frozenset({
    "gsm8k", "longbench_2wikimqa", "longbench_hotpotqa",
    "longbench_narrativeqa", "longbench_qasper",
})
```

It fires per *task name*, gated on membership in that hardcoded set. The
deprecated path keys `accuracy_by_variant[variant][task]` by lm-eval task
name because it partitions probes per task. The live path has
`probe_domains` and no task partition; `v01` probes carry no task name at
all, and lm-eval probes carry one only in `metadata["task_name"]`.

So there is a decision the port forces: **is restored accuracy keyed by
task or by domain?**

- **By domain** is the natural fit for the live path and for `v01`, and
  it matches every other per-cell quantity in a `GeoResult`. It makes
  `AccuracyArtifactFinding` dead code, because no domain name is in
  `_GENERATIVE_TASKS`.
- **By task** keeps the finding alive and matches the existing renderers'
  "Per-task accuracy" heading, but requires the pipeline to partition by
  `metadata["task_name"]`, which is absent for non-lm-eval sets.

**Recommend by domain**, with `_GENERATIVE_TASKS` retired rather than
left dead. It is a third hardcoded list of which tasks are generative —
after `KNOWN_TASK_DOMAINS.output_type` and `TASK_SCORINGS` — and since
v0.4.4 the probes carry the answer: a cell is generative iff its probes
are `output_type == "generate_until"`. That is the same collapse the
evaluator registry just did to `_EVALUATOR_MAP` and
`GENERATE_EVALUATORS`, and it is the third instance of the pattern.

### 3.3 The renderers cope, and already handle `None`

Better news. `_build_accuracy_table` (html), and the markdown and
terminal equivalents, iterate whatever keys are present in
`accuracy[variant]` and render them as columns. They are shape-agnostic:
domain keys render as well as task keys. And `val is not None else
"n/a"` is already there in all three, so the `float | None` that v0.4.4
introduced needs no further work.

The only cosmetic consequence of §3.2 is a heading reading "Per-task
accuracy" over domain columns. One string.

---

## 4. `CapabilityRadar` — keep or remove?

**Split it. `run_single` is worth keeping and ports for free;
`run_pair` should not survive in its present form.**

`run_single` is *"run a Task per domain and collect accuracy"*. After
the port it is a thin, correct wrapper over machinery that works, and it
is the only thing in the tree that produces per-domain accuracy for one
engine. Its cost is the three `.model_name` lines.

`run_pair` is a different proposition. It exists to pair accuracy with
BD on shared generations (L-010 — under sampling decode, two independent
`generate()` calls diverge, so accuracy and distance would describe
different samples). That is a real and well-reasoned constraint. But:

- It drags in `BehavioralDistance`, i.e. §2's eight lines including the
  layering violation.
- **The live path does not use BD at all.** `_pipeline` computes δ
  directly via `engine.score`, deliberately: `geometry.py`'s module
  docstring says *"Zero-coupled with `metrics/*`: we compute CE directly
  via `engine.score` instead of reusing `BehavioralDistance.compute`."*
- Its only in-tree caller is `ModelDiff.capability_radar`, which is on
  the removal list.

So `run_pair` after v0.5.0 is a pairing mechanism, for a metric the live
path does not compute, with no caller.

**Recommend:** port `run_single`, keep it, and **export it** — its
absence from `__init__.py` is why nobody noticed it had stopped working.
Move `run_pair` to the v0.5.0 removal list alongside `ModelDiff`, and if
the accuracy-and-δ-on-shared-generations idea is wanted on the live path
later, it belongs in `_pipeline`, where both engines and the shared
generations already are.

That also answers the omission question from Z.4 item 6: the radar's
absence from the removal list was an omission, and the right correction
is not "add the whole class" but "remove the half that has no future and
fix the half that does".

---

## 5. Scope for v0.4.5

**In:**

1. `lmdiff/_prompting.py` (or similar) — `_prefix_text` and
   `_generate_kwargs` moved out of `_pipeline`, imported by both. §1.3.
2. `Task.run` → the `Engine` Protocol: loop `generate`, `.name`, accept
   the prefix and decode kwargs, relocate the progress bar. §1.4.
3. `loglikelihood_accuracy` → the Protocol: loop `score`, `.name`.
4. `CapabilityRadar.run_single` → the Protocol; exported.
5. **Accuracy on the live path**, keyed by domain, **including the base
   engine** — §3.1 is why the base pass is not optional.
6. `_GENERATIVE_TASKS` retired in favour of `output_type`. §3.2.
7. Byte-equivalence check: the same probe set through the old path and
   the new one must produce identical accuracy. This is the one test
   that matters and it needs a GPU.

**Out:**

- `BehavioralDistance`'s port and `run_pair` — §4 says remove rather
  than port, and that is v0.5.0's business.
- Deleting `InferenceEngine`. This release makes the removal *possible*;
  v0.5.0 performs it.
- Any change to what accuracy *means*. Restoring a capability and
  redefining it in the same release is how you cannot tell which one
  broke.

**The GPU question.** Items 1–4, 6 are CPU-testable with stubs. Item 5's
correctness claim is not: "the ported path produces the same accuracy"
needs a real model, and the natural vehicle is the existing Llama-2
4-variant calibration, which already has a committed baseline. Budget one
calibration run.

**Revertability**, which is the reason this is its own release: items
1–4 and 6 are additive or mechanical, and item 5 is the only one that
changes what a report says. If the port goes wrong, `v0.4.5` reverts to
`v0.4.4` without touching a single deletion — which is exactly what
would not be true inside v0.5.0.

---

## 6. Scripts

| script | produced |
|---|---|
| `port_probe.py` | §1.2 — the prefix/decode table; what a naive port drops |
| `port_inventory.py` | §1.4 — 19 engine-touching lines, checked against the Protocol's 15 members |

Both findings in §1.2 and §3.1 are invisible in the source: the first
lives in a default-argument fallback two calls deep, and the second in a
key that nothing writes.

---

## 8. Implementation record

Scope §5 held. Two things it did not anticipate changed the design, and
both were found by running rather than reading.

### 8.1 A second generation pass would have moved δ

Guard: *stop if base scoring moves any existing number rather than
filling in a missing one.* It would have.

`_delta_for_variant` pins the seed with `manual_seed` **once, at probe
0**, then lets RNG advance naturally through the generate loop (Fix 3,
v0.4.0 PR #15 — the alternative reseeds every probe and over-correlates
the sampling). So an extra generation pass consumes RNG, and for any
sampling variant whose seed resolves to `None` — legitimate, and
documented as non-reproducible by design — δ shifts.

The resolution is better than the one scoped, and it was sitting there:
**score variants on the completions δ was already computed from.** The
δ loop produces `v_outputs`; that is exactly what an evaluator needs. So
`_delta_for_variant` returns it, and `Task.run` grows an `outputs=`
parameter.

Three consequences, all good:

- no second generation, so no RNG consumed and δ provably unmoved;
- no extra GPU time for the variant half — the expensive part is
  already done;
- the L-010 pairing for free. Accuracy and δ describe *one* sample, not
  two, which is the constraint `CapabilityRadar.run_pair` was built to
  satisfy and which now holds on the live path without BD.

Base still needs its own pass — the δ loop generates the *variant's*
output and scores it under both engines; base is never asked to produce
anything. That pass runs **after every variant's δ loop**, so it cannot
perturb the RNG state any variant depended on. One ordering constraint,
no torch, no state save/restore.

### 8.2 Multiple-choice probes would have been scored by string match

Not in the scope at all, and it would have shipped a wrong number rather
than a missing one.

`from_lm_eval` sets `scoring=None` for `multiple_choice` probes — by
design, since they are scored over stored choices rather than by an
evaluator. `Task` therefore falls back to `ContainsAnswer` and matches
the **generated text** against the gold choice. On `hellaswag` that is
not a low accuracy; it is a different measurement.

**Three of the five calibration tasks are multiple-choice**, so this was
the common case:

```
hellaswag                       multiple_choice   scoring=None
arc_challenge                   multiple_choice   scoring=None
mmlu_college_computer_science   multiple_choice   scoring=None
gsm8k                           generate_until    gsm8k_number_match
longbench_2wikimqa              generate_until    f1
```

`_accuracy_by_domain` now dispatches on `output_type` — multiple-choice
through `loglikelihood_accuracy` over the stored choices, everything
else through `Task` on the reused completions. This is
`_accuracy_for_task`'s dispatch, moved to the live path, and it is what
makes the deprecated-path comparison in §9 mechanical rather than
merely indicative.

It also forced one more change: the variant's accuracy is computed
**inside** the per-variant loop rather than after it, because
log-likelihood scoring needs the engine and the cache releases engines
look-ahead-by-one. `score` is a deterministic forward pass and consumes
no RNG, so this does not reintroduce §8.1's problem.

### 8.3 Reported, not fixed

**`task_max_new_tokens` overrides no longer suppress the artifact
caveat.** Accuracy is keyed by domain from v0.4.5; the override dict is
keyed by lm-eval task name. `_effective_max_new_tokens` looks the two up
with the same key, so a `{"gsm8k": 256}` override will not match a
`math` cell and the caveat fires anyway. Fixing it means deciding how a
task-keyed override maps onto a domain-keyed cell, which is a design
question the scope does not cover — and the run config emits
`task_overrides` in the task-keyed form, so the answer has to be
consistent with that too.

**The artifact caveat narrows to live-path results.** `_GENERATIVE_TASKS`
was a hardcoded frozenset of five lm-eval task names; it is now derived
from `probe_output_types`, which is domain-keyed. Results from the
deprecated path key accuracy by task name and so lose the caveat — one
release before the path itself goes. Carrying a hardcoded list of five
task names for one more release, to serve a path being deleted, is the
duplication this change exists to remove.

**`run_pair` is deprecated, not deleted.** §4 recommended removal, and
v0.4.5 gets it as far as a `DeprecationWarning` naming v0.5.0. Nothing
in `lmdiff/tasks/` has ever carried one, and every other v0.5.0 removal
had a minor cycle of notice. Deleting a public method with none would be
the exception.

---

## 9. GPU verification — commands, what to read, pass criteria

Not run. Nothing below has been executed on a GPU; where a CPU stand-in
exists it is labelled and is not offered as verification.

### 9.1 Marker activation, resolved

`pyproject.toml` sets `addopts = "-m 'not slow and not gpu'"`, which
deselects both markers on every run. **`-m ""` clears it** — an empty
expression selects everything.

### 9.2 Run 1 — the gate: geometry must not move

```bash
mamba run -n lmdiff python -m pytest \
    tests/integration/test_calibration_regression.py -m "" -v -s
```

Runs `lmdiff.family(**build_run_kwargs())` from
`tests/integration/_v041_4variant_spec.py` — the same call the fixture
was generated from — against
`tests/fixtures/calibration_v041_4variant_baseline.json` at
`TOLERANCE = 1e-6`.

**Look for:** every test passing, and specifically
`test_change_vectors_match`, `test_cosine_matrix_match`,
`test_selective_cosine_matrix_match`, `test_magnitudes_match`,
`test_magnitudes_normalized_match`.

**Pass: all pass, no skips.** A skip means the baseline fixture is
missing and the run verified nothing — look for `baseline not present`
and stop.

**Any failure fails the release.** §8.1 closes the two mechanisms that
could move these; this is what tests that it did.

### 9.3 Run 2 — accuracy, which has no reference

```bash
mamba run -n lmdiff python scripts/_v045_accuracy_report.py
```

One `family()` call; prints the geometry comparison *and* the accuracy
tables, so 9.2 and 9.3 can be a single GPU pass. `--skip-geometry`
suppresses the duplicate comparison.

**Judged by eye — there is nothing to compare against.** The 4-variant
baseline has no accuracy in it: `accuracy_by_variant` has been `{}`
since v0.4.1, so the fixture cannot serve as a reference. Plausibility
means:

- **`base` is populated.** Empty means `BaseAccuracyMissingFinding`
  will fire on every report forever — the specific thing the base pass
  prevents. The script asserts this and fails the run.
- **`commonsense` ≈ 0.55–0.60, `reasoning` ≈ 0.40–0.45**, matching
  published Llama-2-7B hellaswag / arc_challenge figures. Both are
  scored by log-likelihood over stored choices, so they are directly
  comparable to the literature. Anything near 0.25 is a four-way guess
  and something is wrong.
- **`code` highest for the `code` variant, `math` for `math`.**
  Specialization should be visible in accuracy, not only in δ.
- **`long-context` is `—` in every column.** 91 of 100
  `longbench_2wikimqa` probes exceed Llama-2-7B's 4096-token window, so
  they were never attempted and are excluded rather than counted wrong.
  A number here is the bug.
- **`math` low across the board is expected.** gsm8k is scored on the δ
  generation budget, which is short. If it reads ~0 the
  `AccuracyArtifactFinding` caveat should appear — the script prints
  which ones fired, and its *absence* would be the defect.

**Pass:** base populated; nothing in `long-context`; commonsense and
reasoning within range; no negative, >1.0 or NaN value.

### 9.4 Run 3 — the deprecated-path reference

```bash
mamba run -n lmdiff python scripts/_v045_legacy_accuracy.py
```

**Worth running, and the budget correction is accepted — but it is a
mechanical reference for three of the five tasks, not all five, and
saying which prevents an expected difference reading as a regression.**

| task | domain | comparable? | why |
|---|---|---|---|
| `hellaswag` | commonsense | **yes, 1e-6** | both paths score by log-likelihood over the same stored choices with the same prefix |
| `arc_challenge` | reasoning | **yes, 1e-6** | same |
| `mmlu_college_computer_science` | code | **yes, 1e-6** | same |
| `gsm8k` | math | plausibility only | the deprecated path generates fresh at `TASK_MAX_NEW_TOKENS["gsm8k"]`; v0.4.5 scores the δ generations at the run's `max_new_tokens` |
| `longbench_2wikimqa` | long-context | n/a | out of the base window |

That the three MC tasks *are* exactly comparable is a consequence of
§8.2 — before that fix they would have been string-matched and nothing
here would have lined up.

Two differences that are design, not drift: the deprecated path keys by
task name where v0.4.5 keys by domain (1:1 in this spec, mapping above),
and it has **no base column**, because it never scored the base engine.

**Pass: the three multiple-choice rows agree to 1e-6.** This is the only
mechanical check available anywhere, and v0.5.0 deletes the path that
provides it — the last release in which it can be asked. Disagreement
means the port changed a measurement rather than relocating one:
blocker.

`gsm8k` differing is expected. A difference *larger than the generation
budget accounts for* — the new path at 0.00 against the old at 0.15,
say — is worth stopping on, because it would mean reusing the δ
generations is not equivalent to generating afresh for accuracy.

### 9.5 Cost, and what to cut if the box is tight

Runs 1+2 are one `family()` over 4 variants × 5 tasks × 100 probes — the
call the v0.4.1 calibration already makes, plus the new accuracy work:
log-likelihood scoring for the three MC tasks (~100 probes × 4 choices ×
5 engines including base) and one extra generation pass for base.

Run 3 is a separate `run_family_experiment` with `skip_accuracy=False`,
reloading each variant through the deprecated `InferenceEngine`. It is
the expensive one.

**Run 1 is the release gate. Run 3 is the one I would still argue for**
if time is short: a plausibility judgement on Run 2 can be revisited any
time, and Run 3 cannot be run at all after v0.5.0.
