# LESSONS.md

Append-only incident log. Each entry records a bug, artifact, or design decision that took non-trivial time to work out. Future-you (and future-Claude) should grep this before debugging anything that looks familiar.

Format: L-NNN (zero-padded, sequential), never renumber, never delete entries (strike through with ~~text~~ if superseded).

## Index

- L-001: v01 probe style artifact on base models
- L-002: TokenKL requires full vocab (not top-k)
- L-003: BD without self-entropy baseline is meaningless
- L-004: BD inflation by degenerate outputs in smaller/distilled models
- L-005: Binary task evaluators produce low-SNR signal on base models
- L-006: ContainsAnswer substring match on short expected is biased
- L-007: Typer CliRunner mixes stderr warnings into stdout
- L-008: JSON determinism requires two layers, don't optimize one away
- L-009: TokenKL/TokenEntropy crash on sequence length mismatch
- L-010: CapabilityRadar double-generate breaks accuracy↔BD coupling under sampling
- L-011: tokenizers_equivalent wrongly split slow/fast tokenizer variants

---

## L-001: v01 probe style artifact on base models

**Date:** 2026-04-16  
**Phase:** 1  
**Severity:** Would have corrupted results.

**Symptom:**
- BD(gpt2, distilgpt2) on math = 2.42 nats (overall 1.37)
- CE(gpt2, distilgpt2_output) > 4.6 nats on 10/10 math probes
- Two probes with different text (17+25 vs 7*8) produced CE values identical to 4+ decimal places — effective n=1 not n=10
- 10/30 distilgpt2 outputs were pure `\n`

**Root cause:** v01 math probes were instruction-style ("What is 17 + 25? Answer with just the number."). gpt2 and distilgpt2 are base models with no instruction tuning. gpt2 echoed the instruction back; distilgpt2 emitted `\n` spam. BD was measuring distance between two degenerate-output modes, not capability difference.

**Fix:** Rewrote v01 as completion-style ("17 + 25 = "). Added "Probe design principles" section to CLAUDE.md requiring completion style for base-model probe sets.

**Diagnostic signature (match any two):**
- Two semantically different probes → identical CE to 4+ decimals
- ce_ba or ce_ab > 4 nats on closely-related models
- Output is pure whitespace/newlines or echoes the prompt
- Per-domain BD > 1 nat between distillation siblings

**Diagnostic tool:** `scripts/inspect_v01_bd.py` — dumps per-probe outputs + four CE values. Run on any new probe set before trusting aggregate BD.

**Prevented going forward:** CLAUDE.md "Probe design principles".

---

## L-002: TokenKL requires full vocab

**Date:** 2026-04-16  
**Phase:** 1  
**Severity:** Silent correctness bug.

**Symptom:** TokenKL with topk > 0 would run without error but produce nonsense — `engine.get_logits(topk=K)` returns each engine's own top-K values + indices; the top-K index sets for A and B are different, so position-wise KL over those tensors is aligning disjoint vocabulary subsets.

**Fix:** `TokenKL.compute` forces `topk=0`. Added vocab-size warning for >100k vocab (memory cost scales linearly; full-vocab KL is ~20 MB per position on gemma-sized models).

**Signature:** If you see nan/inf or bizarrely large KL values and topk was passed, this is it.

**Future:** Sparse KL over top-K union is a real optimization (Amini et al. 2025). Not implemented in Phase 1.

---

## L-003: BD without self-entropy baseline is meaningless

**Date:** 2026-04-16 (inferred from CLAUDE.md formula)  
**Phase:** 1  
**Severity:** Conceptual — affects all BD interpretations.

**Why:** Raw CE(A,B) conflates (a) how different A and B are with (b) how predictable A's output is in general. Degenerate outputs (e.g. `\n` spam) have low self-CE (~0.5) but high cross-CE (>4) — this looks like "A and B are very different" when really "A and B produce different flavors of garbage".

Subtracting self-entropy CE(X,X) from cross-entropy CE(Y,X) gives "excess surprise B has about A's output beyond A's own uncertainty" — which is what we actually want to measure.

This was validated by L-001: without baseline subtraction, the v01 math artifact would have looked like genuine 4+ nat divergence.

**Rule:** CLAUDE.md "Common mistakes to avoid" bullet: "Do NOT compute BD without self-entropy baseline subtraction."

---

## L-004: BD inflation by degenerate outputs in smaller/distilled models

**Date:** 2026-04-16  
**Phase:** 1  
**Severity:** Interpretation trap — not a bug. Affects what aggregate BD means.

**Symptom:** On v01 (completion-style, post-L-001), BD(gpt2, distilgpt2):
- All-probe:      1.17 overall, math=1.20 knowledge=0.89 code=1.42
- Healthy-only:   0.79 overall, math=0.59 knowledge=0.95 code=0.73
- ~1/3 of aggregate BD signal was driven by distilgpt2 degenerate outputs (43% of probes: repetition loops on code 8/10, math 4/10, knowledge 1/10).

**Not the same as L-001:** L-001 was bad probe design forcing BOTH models into degenerate modes. L-004 is an intrinsic model property — distilgpt2 on harder prompts collapses even with well-formed probes. Cannot be fixed at probe layer.

**Sanity check for filter correctness:** knowledge domain had healthy-only BD (0.95) slightly *higher* than all-probe (0.89), not lower. This proves the healthy-probe filter is not cherry-picking similar outputs — it's removing genuinely degenerate ones.

**Diagnostic signature:**
- Multiple probes with ce_ba or ce_ab identical to 2+ decimals
- ≥80% of tokens in an output are a single repeated id
- Pure whitespace/newline outputs
- Per-domain BD drops >0.3 nats when restricted to healthy probes

**GPT-2 family note:** `\xa0` (token id 1849) prefix is a known GPT-2 quirk, NOT degeneracy. Do not filter these.

**Implication:** Report BD in two forms — raw + healthy-only — whenever degeneracy rate exceeds 10% for either model. Single-number BD is honest only on well-matched model pairs with low degeneracy.

**Prevented going forward:** Degeneracy detection added to `BehavioralDistance.compute` as `bd_healthy` field + `degeneracy_rate_a/b` in details.

---

## L-005: Binary task evaluators produce low-SNR signal on base models

**Date:** 2026-04-16  
**Phase:** 1  
**Severity:** Interpretation trap — affects what task accuracy means in Phase 1.

**Symptom:** Running Task(v01, ContainsAnswer) on gpt2 vs distilgpt2 (v01 at n=10 per domain):
- overall accuracy: 10% = 10% (3/30 each)
- per-domain n_correct: 0-3 on 10 probes
- BD metric shows 0.79 nat distance between same pair
- Binary match rate cannot distinguish the two models, but distributional metric can

**Root cause:** Base models on completion probes usually produce plausible continuations that are semantically-adjacent but lexically different from `expected`. "import numpy as " → "np" is matched; "The capital of France is " → " a country in Europe..." never hits expected="Paris".

**Design decision for Phase 1:** accept low accuracy as honest measurement, do not tune evaluators to inflate it. Task accuracy is a *floor* on capability, not an estimate of it.

**Implication:**
- Per-domain n=10 on ContainsAnswer means any single probe flip changes accuracy by 10 percentage points. Do not interpret gpt2=0% vs distilgpt2=10% on knowledge as a meaningful difference.
- BD (continuous, uses full distribution) is the higher-SNR metric on base models. Task accuracy is the interpretable-but-coarse one.
- Both should be reported. Neither alone tells the full story.

**Confirmed by L-006:** v01 expansion to n=30 per domain did NOT lift knowledge/math accuracy off the floor. The ceiling was evaluator design, not sample size — see L-006 for follow-up analysis.

**Phase 2 direction (not blocking):**
- Per-probe multiple expected values (expected: ["Paris", "paris"])
- FlexibleMatch evaluator (word-boundary regex, punctuation-stripped)
- Loglikelihood-based evaluator (score expected under model; no generation)

**Not prevented by any rule** — inherent property of measuring base-model capability with strict string match.

---

## L-006: ContainsAnswer substring match on short expected is biased

**Date:** 2026-04-16  
**Phase:** 1  
**Severity:** Evaluator design flaw — biases accuracy measurements.

**Symptom:** In v01 0.2.0 (90 probe, 30 per domain) running ContainsAnswer:
- code (long-ish structural expected like "ZeroDivisionError"): gpt2 47%, distilgpt2 17% — clean differentiation
- knowledge (short nouns like "Paris", "yen"): gpt2 0%, distilgpt2 3% — floor effect, no signal
- Noted probes with expected="che" (for cheetah) or expected="blue" (for blue whale) are susceptible to false positives via substring matching unrelated tokens

**Root cause:** ContainsAnswer uses `expected in output` with case-insensitive match. On:
- Short expected (≤4 chars): high false positive rate — "che" matches "cache"/"chef"/"cheese"; "Au" (gold symbol) matches "Australia"/"August"
- Long expected ("ZeroDivisionError", "import"): much lower false positive rate, accuracy is more meaningful

**Interaction with base models:** Completion-style base models often emit semantically-adjacent continuations ("The capital of France is a major European city...") that never contain the specific short noun we expect ("Paris"). So:
- Short expected → either floor-effect 0% (no match) or inflated by false positive substring hits
- Neither mode is a clean capability measurement

**Implication for interpretation:**
- Code-domain ContainsAnswer accuracy is the most trustworthy on base models (expected values tend to be longer, more structural)
- Knowledge-domain ContainsAnswer accuracy on base models is near-zero regardless of true capability — do not treat 0% as "no capability"
- BD remains the higher-SNR metric for base models; task accuracy is a coarse sanity check

**Not fixed in Phase 1.** Phase 2 directions:
- FlexibleMatch evaluator: regex-anchored match with word boundaries (\bParis\b not substring Paris)
- Multiple-expected probes (expected: ["Paris", " Paris", "Paris,"])
- Loglikelihood-based evaluator (score expected continuation under the model; no generation needed) — standard in lm-eval-harness for base-model knowledge evaluation

**Validates L-005.** n=30 per domain was not enough to make binary accuracy meaningful on knowledge/math — the ceiling was evaluator design, not sample size.

---

## L-007: Typer CliRunner mixes stderr warnings into stdout

**Date:** 2026-04-16  
**Phase:** 1  
**Severity:** Test-infrastructure gotcha — silent parse failures if unhandled.

**Symptom:** E2E CLI test parsed `result.output` as JSON and got JSONDecodeError, even though the CLI command itself produced valid JSON. Transformers emits HF warnings (`Some weights were not initialized...`, progress bars, etc.) to stderr. Typer's CliRunner captures both stdout and stderr by default and merges them into `result.output`, so JSON parsing fails on the mixed stream.

**Root cause:** CliRunner is convenient but not stream-isolated. Anything a dependency prints to stderr during import or model load ends up in `result.output` before the command's actual JSON output.

**Fix:** Introduced `_extract_json()` helper in test_cli_e2e.py that finds the first `{` in output and json.loads from there. Alternative (cleaner) would be to use `mix_stderr=False` on the CliRunner, but that's a newer typer API and not in all versions.

**Signature:** `json.JSONDecodeError` on what "should" be valid CLI JSON output → suspect CliRunner capturing stderr noise.

**Prevented going forward:** `_extract_json()` helper in test_cli_e2e.py. For new CLI E2E tests that expect JSON output, either use `--output FILE` (reads from disk, no stderr mixing) or route through `_extract_json()`.

---

## L-008: JSON determinism requires two layers; don't optimize one away

**Date:** 2026-04-16  
**Phase:** 1  
**Severity:** Maintenance trap — code looks redundant but isn't.

**Symptom:** `report/json_report.py` has two seemingly redundant determinism mechanisms:
  1. Each `to_json_dict` handler writes fields in alphabetical order manually (explicit key ordering in the dict literal)
  2. `json.dumps(..., sort_keys=True)` in the `to_json()` entry point

A well-meaning optimizer might remove layer 1 ("dict insertion order is preserved in Python 3.7+, and sort_keys handles ordering anyway").

**Why both are needed:**
- Layer 1 ensures the Python `dict` returned by `to_json_dict` has deterministic insertion order. Any caller using the dict directly (not the JSON string) gets deterministic iteration.
- Layer 2 ensures the final JSON string is byte-identical across runs even if some nested structure (e.g. a details dict from a metric) has nondeterministic key order from upstream code.
- Layer 2 alone isn't enough if we ever move to a format that doesn't offer sort_keys (YAML, MessagePack, Parquet metadata).
- Layer 1 alone isn't enough because nested user-supplied dicts (metric details) may not be sorted.

Both layers are belt-and-suspenders, and **both are load-bearing**.

**Prevented going forward:** This comment/lesson. If refactoring json_report.py, verify `to_json(r) == to_json(r)` byte-for-byte in tests and keep both mechanisms.

**Related test:** `TestDeterministic.test_same_output_twice` in tests/test_json_report.py.

---

## L-009: TokenKL/TokenEntropy crash on sequence length mismatch

**Date:** 2026-04-19
**Phase:** 1 (v0.1.1)
**Severity:** Hard crash in TokenKL, silent wrong answer in TokenEntropy.

**Symptom:** `ModelDiff(base_config, variant_with_system_prompt_config).run()` crashes in TokenKL with `RuntimeError: The size of tensor a (10) must match the size of tensor b (16) at non-singleton dimension 0`. TokenEntropy does not crash but the delta averages across prefix positions too, inflating the signal.

**Root cause:** `InferenceEngine.get_logits()` tokenized `_build_prompt(probe)` — i.e. prefix text joined with probe text — as a single string. When configs had different `system_prompt`/`context`, the two engines produced logits tensors with different `seq_len` (one had prefix tokens, one did not). TokenKL's per-position subtraction then broadcast-failed. TokenEntropy silently summed entropy over prefix positions that don't belong to the probe, so the "delta" was not purely a probe-distribution comparison.

**Fix:**
- Engine: `_encode_for_model(probe)` tokenizes prefix with `add_special_tokens=True` and probe with `add_special_tokens=False`, concatenates token IDs, returns `(full_ids, probe_slice)`. The probe now occupies exactly `len(probe_ids)` positions regardless of prefix content (no BPE-boundary drift either).
- `ForwardResult.probe_slices: list[slice] | None` exposes where probes live in `input_ids`.
- Metrics: shared `metrics/output/_slicing.py` helpers (`probe_predicting_logits`, `safe_probe_slice`) convert the probe_slice into the logit-range that predicts probe tokens (accounting for the causal-LM off-by-one). TokenKL and TokenEntropy tail-align when one side has no prefix (P=0, L-1 positions) and the other does (P≥1, L positions).
- `generate` and `score` also use `_encode_for_model` so BD self-baseline stays consistent with the new tokenization.

**Architecture note:** `_slicing.py` is a shared private helper, not a metric, so it does not violate the zero-coupling rule (same pattern as `_degeneracy.py`).

**Signature:** If a future metric crashes with a size mismatch on dim 0 after calling `get_logits` on two engines, the caller is almost certainly using `result.logits[i]` directly instead of slicing with `probe_slices[i]`.

**Diagnostic tool:** the `TestProbeSliceAlignment` mock tests in `tests/test_metrics_output.py` reproduce the exact 10 vs 16 mismatch without loading models.

---

## L-010: CapabilityRadar double-generate breaks accuracy↔BD coupling under sampling

**Date:** 2026-04-19
**Phase:** 1 (v0.1.1)
**Severity:** Silent correctness bug under sampling decode; wasted compute under greedy.

**Symptom:** `CapabilityRadar.run_pair` internally called `engine.generate(...)` once inside `Task.run` (for the accuracy evaluator) and then `BehavioralDistance.compute` called `engine.generate(...)` again. Under greedy decoding these two calls produce identical outputs, so only compute was wasted. Under sampling decode (`decode={"strategy": "sample"}`) the two generations diverge — accuracy and BD then describe different samples of the same prompt.

**Root cause:** `Task.run` and `BehavioralDistance.compute` both owned their own `.generate()` call. There was no mechanism to share generations across the two views.

**Fix:**
- `Task.run(engine, pre_generated=None)` accepts a pre-existing `GenerationResult` and reuses it instead of calling `engine.generate`.
- `BehavioralDistance.compute(..., pre_gen_a=None, pre_gen_b=None, **kwargs)` reads generations from kwargs when provided.
- `CapabilityRadar.run_pair` generates once per engine per domain, passes the same `GenerationResult` to both the task evaluator and to BD.

**Side effect of the fix:** generate call count dropped from 2×N_domains to 1×N_domains per engine. `TestRunPairMock.test_pair_calls` was updated from `call_count == 6` to `call_count == 3` for the 3-domain fixture.

**Signature:** If accuracy and BD disagree in non-obvious ways under sampling decode (e.g. accuracy says the model answered correctly but BD shows large distance to its own generation), suspect that two independent `generate()` calls were made and only one was evaluated.

---

## L-011: tokenizers_equivalent wrongly split slow/fast tokenizer variants

**Date:** 2026-04-20
**Phase:** 1
**Severity:** Silent false-negative — metrics that require matching tokenizers (TokenKL, TokenEntropy) would refuse to run on valid same-tokenizer pairs, and BD would fall back to BPB normalization unnecessarily.

**Symptom:** Loading `meta-llama/Llama-2-7b-hf` with `use_fast=True` (→ `LlamaTokenizerFast`) and again with `use_fast=False` (→ `LlamaTokenizer`) should produce the same token ids. But `tokenizers_equivalent(tok_slow, tok_fast)` returned False. TokenKL/TokenEntropy then raised "requires matching tokenizers" and BD silently switched to BPB mode when neither was warranted.

**Root cause (two bugs stacked):**
1. **Class-name gate.** The old implementation rejected pairs where `type(a).__name__ != type(b).__name__`. Slow/fast variants are different Python classes by design; rejecting on class name treated a purely implementation-level distinction as a semantic difference.
2. **`encode()` default drift.** The canary-string comparison used `tok.encode(text)`. HuggingFace's `encode()` takes `add_special_tokens=True` by default, but the *effect* of that default varies across tokenizer subclasses — some add BOS, some don't, and the slow/fast variants of the same tokenizer can disagree on which tokens count as "special" in `encode()`. So even without the class-name gate, encoded ids could legitimately differ for the same input string despite identical underlying vocabularies.

**Fix:**
- Dropped the `type().__name__` check entirely.
- Replaced `tok.encode(text)` with `tok(text, add_special_tokens=False)["input_ids"]`. Explicit `add_special_tokens=False` removes the subclass-dependent default and guarantees we compare only the textual tokenization.
- Rationale is attached to the docstring so a future refactor doesn't reintroduce either trap.

**Signature:** If a metric that declares `requires matching tokenizers` refuses to run on what should be identical tokenizers, or if BD reports `bpb_normalized: True` between two configs that use the same base model, re-run `tokenizers_equivalent` and check whether the canary-string comparison is being thrown off by `add_special_tokens` defaults.

**Test:** `tests/test_tokenizer_utils.py::TestTokenizersEquivalent::test_llama2_slow_vs_fast` (slow — requires the llama2 weights/tokenizer files).
