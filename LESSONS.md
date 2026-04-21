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
- L-012: Mock engine tests silently take BPB path when Config.model strings differ
- L-013: Geometry uses global NaN filter, not BD's per-pair skip
- L-014: v01 math probes function as an entropy-direction detector
- L-015: config-only variant δ mixes behavior shift with conditioning asymmetry
- L-016: InferenceEngine.__new__ reuse trick is load-bearing
- L-017: δ = constant + selective decomposition recovers cosine resolution
- L-018: JSON dict keys come back alphabetical, not in insertion order
- L-019: default code axis uses capability MCQ, not codegen task
- L-020: lm-eval tasks can sys.exit() instead of raising
- L-021: tests of optional-dep features need skipif, or CI breaks silently

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

---

## L-012: Mock engine tests silently take BPB path when Config.model strings differ

**Date:** 2026-04-20
**Phase:** 2 (geometry)
**Severity:** Silent test artifact — real code was correct, test was wrong, both produced numerically plausible results. Would have been committed and re-hit on the next multi-engine metric test.

**Symptom:** `TestChangeVectorComputation.test_delta_matches_manual` hand-computed δ = 2.0 on a specific probe. The test got δ ≈ 2.885. Ratio 2.885 / 2.0 ≈ 1.4427 = 1/log(2), i.e. the nat→bit conversion inside `bpb_from_ce`. BPB was running when nothing in the test scenario asked for it.

**Root cause:** The mock engines were built with `Config(model="mock-a")` for base and `Config(model="mock-b")` for the variant to look "realistic." Consequence:
1. `config_a.shares_tokenizer_with(config_b)` returns `None` (different strings, can't decide)
2. geometry falls back to `tokenizers_equivalent(tok_a, tok_b)`
3. MagicMock tokenizers compare unequal (different Mock instances)
4. geometry silently enters `use_bpb=True`
5. CE gets scaled by `(n_tokens / log(2)) / byte_count`

The geometry code was doing exactly what the spec said. The test's "realistic" setup was accidentally triggering a valid but unintended code path.

**Fix:** Share `model=` string across mocks and differentiate via `name=`:
```python
Config(model="mock-model", name="base")
Config(model="mock-model", name="A")
Config(model="mock-model", name="B")
```
Now `shares_tokenizer_with` returns `True` on the string-equality fast path, BPB is skipped, raw nat CEs are compared directly.

**Diagnostic signature:**
- Test asserts expected value X, got ≈ 1.4427 × X (or 1/log(2) × X = 0.6931 × X if the direction was reversed)
- Inspect `result.metadata["bpb_normalized"]` — if `True` for mock variants with no intentional tokenizer difference, this is it

**Prevented going forward:** Any new multi-engine metric test must either
(a) use the same `model=` string across configs and differentiate via `name=`, OR
(b) explicitly `patch.object(Config, "shares_tokenizer_with", return_value=True)` around the compute call.

Pattern (a) is preferred because it mirrors production usage (same tokenizer family = same model string prefix most of the time).

**Related test:** `tests/test_geometry.py::TestChangeVectorComputation`

---

## L-013: Geometry uses global NaN filter, not BD's per-pair skip

**Date:** 2026-04-20
**Phase:** 2 (geometry)
**Severity:** Design-decision record — the code is correct, but the reason it deliberately diverges from BD's NaN handling needs to be discoverable before anyone "fixes" it.

**Why this is not a bug:** BD is a scalar over pair (A, B). When probe i has NaN CE, BD drops that probe from the per-pair aggregate and no one else sees it. Per-pair skip is sufficient because the output (one BD number) doesn't depend on any other pair.

ChangeGeometry is different. Its output is a cosine matrix over N variants: `cos(δ_A, δ_B)` requires δ_A and δ_B to have both the same **dimension** AND the same **probe basis**. If variant A skips probe 1 but variant B skips probe 3, their change vectors drift out of alignment — either the dimensions mismatch outright, or index i refers to different probes in the two vectors, making the inner product meaningless.

**Design:** `ChangeGeometry.analyze()` pre-computes raw δ per variant (NaN-preserving, length = n_total_probes), then takes the intersection of "probes where ALL variants produced valid CE" as the universal basis. Every variant's `change_vectors[v]` is restricted to this basis. Reported in `metadata["n_skipped"]`.

**Cost:** one broken variant (e.g. emitting empty continuations on half the probes) drags down `n_probes` for every other variant too. This is the right tradeoff — the alternative (per-pair alignment) requires materializing a different basis per `(i, j)` pair in the cosine matrix, which defeats having a single matrix at all.

**Implication for future N-way metrics:** any metric that operates on multiple variants and requires cross-variant aggregation (cos, PCA on δ vectors, average distance to centroid, steering-vector composition) MUST use global filtering. Per-pair skip is correct only for strictly pairwise metrics like BD.

Metrics that are per-variant only (e.g. "mean δ magnitude per variant") can use per-variant NaN skip — no cross-variant alignment needed.

**Not a forward-proofed rule for rectangular N-way structures:** this lesson is specifically about geometries where the output is a symmetric cos-like matrix over variants. If someone designs a metric with asymmetric N-way structure (e.g. "which variant's δ best predicts variant X's δ"), the filter design may need to be reconsidered per-row.

**Related code:** `lmdiff/geometry.py::ChangeGeometry.analyze` — the `valid_indices` construction
**Related test:** `tests/test_geometry.py::TestNaNHandling`

---

## L-014: v01 math probes function as an entropy-direction detector

**Date:** 2026-04-20
**Phase:** 2 (family geometry experiments)
**Severity:** Interpretation rule — the probe set has a directionality property not documented before.

**Observation:** Running ChangeGeometry on Llama-2-7b against every entropy-reducing variant, the per-domain δ magnitude ordering is always `math > knowledge > code`, independent of which modification produced the variant:

```
variant      type         math    know    code    note
yarn         weight_mod   9.91    6.56    5.08    RoPE scaling
long         weight_mod   8.04    7.71    5.68    long context
code         weight_mod   7.25    4.16    4.36    code pretrain (BPB)
sysprompt    config_only  10.96   8.55    4.92    "You are a helpful assistant"
temp         config_only  2.29    2.75    5.34    temperature=1.5 sampling
```

`temp` is the only variant with the opposite ordering (`code > know > math`).

**Mechanism:** v01 math probes (`"17 + 25 = "`, etc.) are a region of the probe space where Llama-2 base has high self-entropy — the base distribution is not very peaked on the right continuation. Any modification that tightens the output distribution (lowers entropy) drops CE on these probes by more than it drops the base-scoring-the-variant CE, so δ = CE(base of V) − CE(V of V) is large on math, smaller on the already-confident knowledge/code domains. This is not a probe-set bias — the probe set is functioning as a direction detector.

**Diagnostic signature:** `math > knowledge > code` per-domain ordering → entropy-reducing modification; reverse ordering → entropy-raising. This is a more robust 2-bit behavioral signal than the raw magnitude number.

**Implications for reporting:**
- Magnitude ordering tells you "how much the variant moved in v01's high-entropy region," not "how much it moved overall."
- Cross-variant magnitude comparisons are valid as numbers (same unit) but reflect probe-basis-specific shift, not universal shift.
- Paper and README should carry a probe-basis caveat whenever magnitudes are compared across variants.

**Validation:** partial (3 variants) + extended (5 variants) experiments — consistent signature. A future Phase 2 Step 4 run with lm-eval-style probes would check whether the ordering persists when the probe distribution changes.

---

## L-015: config-only variant δ mixes behavior shift with conditioning asymmetry

**Date:** 2026-04-20
**Phase:** 2 (extended family geometry)
**Severity:** Interpretation rule — a naive magnitude comparison between weight-mod and config-only variants is misleading.

**Observation:** `sysprompt` variant (same Llama-2-7b weights, `system_prompt="You are a helpful assistant."`) has magnitude 14.75, larger than every weight-mod variant in the extended experiment (yarn 12.93, long 12.50, code 9.43). Taken at face value this says "adding a system prompt is a bigger behavioral change than fine-tuning 7B parameters on 32k context." The face value is wrong.

**Mechanism:** A config-only variant's δ vector carries two components that token-level CE cannot separate:
1. The variant actually produced a different continuation distribution given the prompt (genuine behavioral shift).
2. Scoring asymmetry: during `_run_config_only`, base scores with only the bare prompt, but variant's self-scoring sees `system_prompt + prompt`. The variant's self-CE therefore benefits from extra conditioning information the base didn't have. This inflates δ irrespective of actual behavioral change.

Weight-mod variants don't have component 2 — base and variant see the same bare prompt at score time.

**Rule:**
- Config-only and weight-mod magnitudes share units (nats / bpb) so arithmetic comparison is well-defined, but they are not semantically comparable.
- `GeoResult.metadata["variant_types"]` marks each variant as `"weight_mod"` or `"config_only"`. Any downstream report that ranks variants by magnitude MUST disclose the type; cross-type claims like "config change matters more than fine-tuning" are not supported by the current measurement.
- Truly separating components 1 and 2 requires a fixed-reference-continuation scoring mode where both base and variant score the same reference text. Not implemented yet — future Phase 2 work.

**Validation:** extended experiment observed, bit-identical across two seeded runs, so not sampling noise.

---

## L-016: InferenceEngine.__new__ reuse trick is load-bearing

**Date:** 2026-04-20
**Phase:** 2 (extended family geometry)
**Severity:** Maintenance trap — a deliberate bypass of `__init__` will silently break when the engine gains a new required attribute.

**Symptom/use:** `scripts/run_family_geometry_extended.py` needs to run a config-only variant (same weights, different `system_prompt` or `decode`) without reloading 7B weights. The base engine already occupies ~14 GB; a second load would OOM on a 32 GB GPU. The script reuses the base engine's weights via:

```python
variant_engine = InferenceEngine.__new__(InferenceEngine)
variant_engine.config = variant_config
variant_engine.device = base_engine.device
variant_engine._model = base_engine._model
variant_engine._tokenizer = base_engine._tokenizer
# then variant_engine.generate / score work normally
```

**Why it works:** Today, `InferenceEngine.__init__` only sets four attributes downstream methods read — `config`, `device`, `_model`, `_tokenizer`. Bypassing `__init__` skips `_load()` (weight download + GPU placement) but still satisfies `generate()` and `score()`.

**Why it's fragile:** Any new required attribute added to `__init__` (e.g. `_chat_template`, `_adapter_state`, `_quantization_config`) makes the script fail not at construction but deep inside `generate()` or `score()` as an `AttributeError` at the first attribute access. The script's VRAM reuse is the load-bearing property; there is no test for it because creating two 7B engines would blow CI VRAM budgets.

**Protection:**
- A future cleanup could expose `InferenceEngine.from_shared(other, config)` as a first-class API. Not worth it for one caller — the hack is documented instead.
- Anyone adding required attributes to `InferenceEngine.__init__` should grep for `InferenceEngine.__new__` and update the script (or promote the pattern to a helper).

**Validation:** extended experiment run #1 + run #2 both completed with Phase B VRAM peak ~18.6 GB (base weights + activations only). A reload would have pushed peak to ~28 GB.

---

## L-017: δ = constant + selective decomposition recovers cosine resolution

**Date:** 2026-04-20
**Phase:** 2 (extended family geometry, Step 1.5)
**Severity:** Analysis rule — the original cosine matrix alone can be misleadingly compressed.

**Observation:** Decomposing each variant's change vector as `δ = c·𝟙 + ε` (with `c = mean(δ)`) on the 5-variant extended experiment:

| variant     | type          | magnitude | const_frac | selective_mag |
|-------------|---------------|-----------|------------|---------------|
| yarn        | weight_mod    | 12.93     | 0.73       | 6.76          |
| long        | weight_mod    | 12.50     | 0.86       | 4.73          |
| code        | weight_mod    | 9.43      | 0.74       | 4.76          |
| sysprompt   | config_only   | 14.75     | 0.78       | 6.94          |
| temp        | config_only   | 6.43      | 0.35       | 5.18          |

Four entropy-reducing variants (yarn, long, code, sysprompt) sit at constant_fraction ∈ [0.72, 0.86] (mean 0.77). The entropy-raising variant (temp) drops to 0.35.

**Original cosine vs selective cosine** (selective cosine = cosine after centering both vectors = Pearson correlation):

```
pair                   original  selective
yarn vs long             0.868    0.402
yarn vs code             0.856    0.457
yarn vs sysprompt        0.853    0.410
long vs code             0.873    0.388
long vs sysprompt        0.912    0.533
code vs sysprompt        0.846    0.353
--- temp-involving ---
yarn vs temp             0.560    0.131
long vs temp             0.599    0.165
code vs temp             0.514    0.007
sysprompt vs temp        0.535    0.032
```

**Implications:**
- The [0.85, 0.91] narrow band in the original cosine matrix is mostly illusory. Four different modification mechanisms all happen to lift δ roughly uniformly across probes; that uniform offset contributes a near-parallel `c·𝟙` component to each δ, collapsing their cosines into a tight cluster.
- Selective cosine removes the offset and recovers ~3× resolution: entropy-reducing variants still cluster but now in [0.35, 0.53] (real shared selective pattern), and entropy-reducing vs entropy-raising cosines collapse to [0.00, 0.17] (effectively orthogonal, not weakly correlated as the original cosine suggested).
- The three metrics carry orthogonal information:
  - `magnitudes` — how far δ moved in total
  - `cosine_matrix` — raw directional agreement (dominated by constant offset when constant_fraction is high)
  - `selective_cosine_matrix` — directional agreement after removing the uniform offset
  - `constant_fractions` — how much of the δ energy is "uniform offset" vs "selective pattern"

**Framework implications:**
- `GeoResult` stores all four; downstream chooses what to show.
- Terminal and JSON reports emit both cosine matrices plus the constant_fraction column.
- Paper and README must name which cosine they are reporting; both should be reported together.

**Validation:** Extended experiment (5 variants, 90 probes). Robustness to probe set change is a Phase 2 Step 4 item.

---

## L-018: JSON dict keys come back alphabetical, not in insertion order

**Date:** 2026-04-20
**Phase:** 2 (post-geometry extended analysis)
**Severity:** Silent analysis bug. The script ran, produced 90-d vectors on both sides, and spat out Pearson r — the number was just noise. No exception, no shape mismatch, no log anomaly.

**Symptom:** A prompt-length hypothesis script tried to compute the Pearson correlation between selective δ and probe token count for each variant. It sourced probe text via `list(per_probe[name].keys())` (tokenizing each to get token counts), and sourced δ via `change_vectors[name]`. The two 90-element sequences were misaligned: `per_probe` keys came out `['# Compute the factorial...', '# Reverse a string...', '10 squared is ', '100 - 37 = ', '11 * 11 = ', ...]` (alphabetical), while `change_vectors[name]` was still in v01.json order (`'17 + 25 = '`, `'144 / 12 = '`, ...). Every index i paired a δ from probe A with a token count from probe B.

**Root cause:** `lmdiff/report/json_report.py::to_json` calls `json.dumps(d, sort_keys=True, ...)`. The `sort_keys=True` is intentional — it gives byte-exact determinism so round-trip tests, CI fixtures, and version diffs are stable. But it applies to every dict in the output tree, including `per_probe[name]`, which is keyed by probe text. After round-tripping through JSON, those keys come out alphabetically sorted. Lists don't have this problem, so `change_vectors[name]` (a list of floats) keeps prompts order.

In-memory GeoResult and JSON-roundtripped GeoResult are NOT interchangeable on `per_probe` key order. This is the specific trap: during construction, `per_probe` was built in prompts order and `list(per_probe[name].keys())` would have worked — but the moment it survives one serialize/deserialize cycle, the keys are re-sorted.

**Rule:**

- To iterate δ by probe order → use `change_vectors[v]` (list, order preserved).
- To look up δ for a specific probe text → use `per_probe[v][probe_text]` (dict lookup by value, order-independent).
- **Do NOT** use `list(per_probe[v].keys())` as a proxy for probe order, especially in scripts that consume the serialized JSON.

**How to align probe text + δ in a downstream script:**

1. Read probe text + order from the original ProbeSet / probe-set JSON.
2. Read δ values from `change_vectors[v]` (prompts order).
3. Assert `metadata["n_skipped"] == 0`; otherwise `change_vectors[v]` has been NaN-filtered and is shorter than the probe list, so index-by-index alignment breaks. If `n_skipped > 0`, recover `valid_indices` from the alphabetized `per_probe[v]` keys and re-sort both sources to match.

**Not fixing `sort_keys=True`.** The determinism guarantee is load-bearing for two-layer JSON output stability (see L-008) and for round-trip tests.

**Docs updated:** module docstring of `json_report.py`, and the `per_probe` field docstring on `GeoResult` in `geometry.py`, both now flag this behavior inline.

**Diagnostic signature:** analysis pipeline emits plausible-looking but suspiciously uniform Pearson r values across variants whose δ vectors visibly differ in magnitude — check whether the text-side and value-side sequences were sourced from two different ordering domains.

**Related code:** `lmdiff/report/json_report.py::to_json` (sort_keys=True); `lmdiff/geometry.py::GeoResult.per_probe`.
**Related script:** `scripts/analyze_prompt_length_hypothesis.py` (uncommitted; already follows the documented rule after the fix).

---

## L-019: default code axis uses capability MCQ, not codegen task

**Date:** 2026-04-20
**Phase:** 2 (Step 4 — lm-eval integration)
**Severity:** Design-decision record — the default code task for the accuracy radar is not the one most readers would expect, and the reason needs to be discoverable.

**Context:** Step 4 adds `from_lm_eval` adapter and a `KNOWN_TASK_DOMAINS` registry used by future experiment scripts to pick default tasks per axis. The natural choice for the code axis is HumanEval or MBPP — both are iconic LLM code benchmarks. Both have `pass@k` as native metric, which requires code execution.

**Decision:** Default code axis uses `mmlu_college_computer_science` (multi-choice accuracy, no execution). HumanEval and MBPP are in `KNOWN_TASK_DOMAINS` with `requires_execution=True` and remain available for δ-magnitude-only experiments (no accuracy scoring), but are excluded from the default accuracy task set.

**Reasoning:**
- Accuracy radar is part of v0.2.0 user-facing output; its task set has to use honest native metrics.
- Non-execution code metrics (BLEU, CodeBLEU, ChrF, CodeBERTScore) correlate poorly with functional correctness (Evtikhiev et al. 2023; CodeScore paper). Using them as a HumanEval-axis accuracy would mislead users more than omitting the axis.
- Sandboxing for pass@k is ~300 lines of code plus a security review; out of Step 4 scope.
- `mmlu_college_computer_science` probes code understanding as a multi-choice capability question. Not code generation, but a clean accuracy axis.
- The framework still distinguishes code-related capability (mmlu_cs) from codegen behavior (humaneval δ-only); they just live on different axes of different radars.

**Implication:** δ-magnitude radar and accuracy radar may have different axis sets when executional tasks are involved. This is by design — behavior and result probe different things.

**Enforcement:** `KNOWN_TASK_DOMAINS["mmlu_college_computer_science"].notes` includes "Default code axis." and `KNOWN_TASK_DOMAINS["humaneval"].requires_execution == True` with a note about magnitude-only usage. Default-task-picker scripts should check `requires_execution` before adding a task to the accuracy radar axis.

---

## L-020: lm-eval tasks can `sys.exit()` instead of raising

**Date:** 2026-04-20
**Phase:** 2 (Step 4)
**Severity:** Script-level gotcha — `except Exception` is insufficient, `except BaseException` is required.

**Symptom:** While building `scripts/discover_lm_eval_tasks.py` to enumerate all lm-eval tasks, the unfiltered 14069-task run would exit abruptly with `exit code 1` and no Python traceback, after processing a portion of the registry. Changing the exception type in the `try: tm._get_config(name)` block from `Exception` to `BaseException` fixed it.

**Root cause:** Some lm-eval task configs depend on system-level binaries (e.g. SWI-Prolog for prolog tasks, specific toolchains for certain code tasks). When the dep is missing, lm-eval's config loader calls `sys.exit(1)` during `_get_config` — this raises `SystemExit`, which inherits from `BaseException` **not** `Exception`. A `try: ... except Exception:` around the config read lets `SystemExit` propagate up and kills the script. `sys.exit()` skips traceback printing, so the failure looks mysterious.

**Fix:** In discovery / enumeration code that sweeps across many lm-eval tasks, wrap per-task config reads in `except BaseException:` with an explanatory comment. Specifically in `scripts/discover_lm_eval_tasks.py::read_task_config_shallow`.

**Signature:** Script exits with code 1 partway through enumerating many lm-eval tasks; no traceback; last line of stderr looks normal (e.g. `Error: SWI-Prolog (swipl) is not installed...`). Don't debug `_get_config`; widen the exception handler.

**Broader implication:** Any code path that invokes lm-eval internals across a large task set (future batch loaders, validation suites, benchmarks against the full registry) needs `except BaseException:`. Confining `except Exception:` to narrow, per-call contexts where a `sys.exit` would be a clear bug on lm-eval's side is still fine; but sweep code that must keep going must catch broader.

**Related code:** `scripts/discover_lm_eval_tasks.py::read_task_config_shallow`.

---

## L-021: tests of optional-dep features need skipif, or CI breaks silently

**Date:** 2026-04-21
**Phase:** 2 (Commit A follow-up)
**Severity:** CI regression surfaces silently — local suite passes, CI red. Wasted about half an hour the first time.

**Symptom:** Commit A added `GeoResult.cluster(...)` which lazy-imports `scipy.cluster.hierarchy`. Local `pytest tests/` was 388 passed; CI run `24709354990` failed on three `TestCluster` tests with `ImportError: scipy required for hierarchical clustering. Install with: pip install lmdiff-kit[viz]`. Rerunning locally still passed.

**Root cause:** scipy lived only in the `[viz]` optional extra. The local env had installed lm-eval earlier, which transitively pulls scipy, so scipy was present for my test runs without me realizing it. GitHub Actions installed only `[dev]` (via `pip install -e ".[dev]"`), which does not include scipy. Hence the cluster tests hit the real `ImportError` path — which was the code under test for one of them, but fatal for the three that expected scipy to be available.

**Fix (two parts):**
1. In `tests/test_geometry.py` introduced a `scipy_required` `pytest.mark.skipif` decorator mirroring the `matplotlib_required` pattern already used in `tests/test_viz_radar.py`. Applied to the three tests that need a working scipy call. The four tests that exercise ValueError / bad-method / bad-metric / ImportError-when-scipy-missing do NOT get the skip — they exercise paths that short-circuit before scipy is imported, so they must run in every env.
2. Updated `.github/workflows/test.yml` to install `[dev,viz]` so CI actually runs the skipped-ifs instead of skipping. Contributor envs that install only `[dev]` still work (tests skip gracefully).

**Rule:** any test that touches an optional-dep path must either (a) install the dep in CI *and* be decorated with `pytest.skipif(not _dep_available(), reason=...)` so contributor envs degrade cleanly, or (b) test only the ImportError-with-install-hint branch (which works without the dep installed). Never rely on "the dep happens to be in my env because some other extra pulled it in."

**Diagnostic signature:** local `pytest` green, GitHub CI red on the same commit, failing test invokes an optional import. The delta is usually which extras got pip-installed.

**Related code / workflow:** `tests/test_geometry.py::TestCluster`, `tests/test_viz_radar.py::TestRender`, `.github/workflows/test.yml`.
