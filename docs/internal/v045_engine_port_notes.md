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
