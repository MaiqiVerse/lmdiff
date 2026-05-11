# v0.4.1 — Measurement Validity Framework + pdn Correction (Design Audit)

> **Status:** AUDIT ONLY. This document is a design specification produced for
> user review before any implementation begins. No production code is changed
> by this PR.

## Background

Lab feedback raised "我们额外的 normalization（根号 T）似乎没什么根据" against the
v0.3.2 PR #11 share-calculation correction. The follow-up audit chain (Y.1–Y.3)
identified three layered failures:

1. **Dimensional inconsistency.** The current `magnitudes_per_domain_normalized`
   formula `sqrt(Σ δ² / Σ T)` was derived under the implicit assumption that
   `δ` carries units of `nats` (total CE per probe). The implementation actually
   produces `δ` in `nats/token` (per-token CE difference, T-invariant by
   construction). Dividing by `Σ T` once more over-corrects, yielding units of
   `nats / token^1.5` — not a meaningful per-token measure.

2. **Long-context σ inflation.** Long-context probes in the demo set average
   ~9,000 prompt tokens, well beyond the 4,096 context window of the Llama-2-7B
   base model. The base scores those probes outside its trained context — the
   per-token CE diff `δ` for long-context probes is **catastrophic-failure
   noise**, not real measurement signal. Any per-token formula (Formula C and
   beyond) will surface this noise as a dominant share. Formula B accidentally
   hid it by per-token-averaging across the huge `Σ T`.

3. **Self-consistent mockup evaded validation.** The v0.3.2 PR #11 prototype
   numbers and the production implementation both used the same formula, so
   the mockup confirmed itself rather than the methodology.

The fix needs to be measurement validity (drop the noise) **before** any
formula change. With out-of-context probes removed, Formula C (a clean per-token
RMS) becomes the correct aggregator and dimensions line up.

This audit lays out the design for all 8 v0.4.1 components.

---

## Table of contents

1. Per-probe validity check
2. Domain status taxonomy
3. Share calculation with validity
4. pdn formula correction
5. Schema migration
6. Visualization
7. Documentation
8. Tests
9. Open questions for user review

---

## 1. Per-probe validity check

### 1.1 Current pipeline tokenization audit

Tracing `lmdiff.family(...) → run_family_pipeline → _delta_for_variant`:

| step | file:line | what happens |
|---|---|---|
| Entry | `_api.py:435` | `family()` constructs `base_engine` eagerly, builds an `engine_factory` for variants (Fix 4 path) |
| Pipeline entry | `_pipeline.py:407` | `run_family_pipeline` invoked; `prompts = list(probe_set.texts)` |
| **Per-probe token count** | `_pipeline.py:432` | `all_probe_tokens = [base_engine.token_count(p) for p in prompts]` — runs **once, base-only**, BEFORE the variant loop |
| Variant loop | `_pipeline.py:454+` | Per-variant cache + look-ahead release; constructs `v_engine` lazily |
| Generate | `_delta_for_variant`, `_pipeline.py:204` | `v_engine.generate(prompts[i], prefix_text=v_prefix, ...)` — produces continuation text |
| Score base on variant output | `_pipeline.py:241` | `base_engine.score(prompts[i], v_outputs[i], prefix_text=base_prefix)` |
| Score variant self | `_pipeline.py:285` | `v_engine.score(prompts[i], continuation_ids=v_ids_per_probe[i], prefix_text=v_prefix)` |
| GeoResult assembly | `_pipeline.py:528` | `avg_tokens_per_probe` populated from `all_probe_tokens` after universally-valid-indices filter |

**Key findings:**

- `T_i` for the **probe prompt alone** is computed once at line 432, base-only.
- The probe prompt + continuation length (what actually enters `score()`) is
  not tracked: `_delta_for_variant` calls `score` per probe but doesn't record
  combined `T_i = T_prompt + T_continuation`. Continuation length varies per
  variant (different `generate` outputs).
- The variant's tokenizer's count is never asked. CodeLlama and Llama-2 share
  vocab → same count. DeepSeek-coder (future) would differ → currently invisible.

### 1.2 Context window queries per engine type

`HFEngine.__init__` (line `_engine.py:410`) accesses `self._model.config.{num_hidden_layers, hidden_size}`. **It does not currently read context-window fields.** Per-model audit:

| model id | HF config field | typical value |
|---|---|---|
| `meta-llama/Llama-2-7b-hf` | `max_position_embeddings` | 4096 |
| `meta-llama/Llama-2-7b-chat-hf` | `max_position_embeddings` | 4096 |
| `codellama/CodeLlama-7b-hf` | `max_position_embeddings` | 16384 |
| `EleutherAI/llemma_7b` | `max_position_embeddings` | 4096 |
| `togethercomputer/LLaMA-2-7B-32K` | `max_position_embeddings` | 32768 |
| `NousResearch/Yarn-Llama-2-7b-128k` | `max_position_embeddings` | 131072 (RoPE-extrapolated) |
| `mistralai/Mistral-7B-v0.1` | `max_position_embeddings` | 32768 (sliding-window 4096) |
| GPT-2 family | `n_positions` (alias `n_ctx`) | 1024 |

**Yarn caveat:** the model card says "128K" and `max_position_embeddings = 131072` is honored by HF, but actual quality degrades past the original training distribution. For lmdiff purposes — measuring distributional drift, not benchmarking — we trust the model's stated capacity as the validity threshold. The user can override on a per-engine basis.

**MinimalEngine** (`engines/minimal.py:75`): no model object. Recommend adding an overridable hook `_max_context_impl(self) -> int | None` analogous to `_token_count_impl`, defaulting to `None` (unknown/unlimited).

**MockEngine** (`tests/fixtures/mock_engine.py:26`): tests don't care about context limits — default to `None` (treat as unlimited). Tests that want to exercise the validity path will override.

**Proposed Engine Protocol addition** (no implementation this turn — just spec):

```python
# lmdiff/_engine.py — Engine Protocol
def max_context_length(self) -> int | None:
    """Largest sequence length this engine can score without truncation.

    None means unknown/unlimited (the caller treats every probe as valid).
    HFEngine returns ``model.config.max_position_embeddings``; subclasses
    can override (e.g. an APIEngine that knows its provider's hard cap).
    """
    ...
```

For HFEngine the implementation is one line: `return self._model.config.max_position_embeddings` (with `getattr` fallback to `n_positions` for GPT-2-style configs).

### 1.3 Where to check validity (Options A/B/C)

| option | when | cost | trade-off |
|---|---|---|---|
| **A — at probe-load** | `_pipeline.py:432`, right after token count | None (already iterating prompts) | Need every engine's `max_context` loaded; works for base eagerly, variants must be probed (≤ N engine instantiations just to ask context-length) |
| **B — at score time** | inside `_delta_for_variant`, per probe | One conditional per (variant, probe) — negligible | Already in the per-probe loop; lazy engine has already been constructed by Fix 4 by this point so `v_engine.max_context_length()` is available |
| **C — post-pipeline** | at GeoResult assembly | Cheapest in Python; **most expensive in GPU** — invalid probes get scored anyway | Wastes ~30 minutes of GPU on the 7-variant demo (long-context probes are by far the slowest); won't catch tokenizer-error case |

**Recommendation: Option B**, hybrid pre-screen.

Concretely:
- At the top of `_delta_for_variant`, after `v_engine` is in scope (lazy-loaded or cached), build the per-probe validity record for `(base_engine, v_engine)` over all probes. One `T_i + max_context_length` comparison per (engine, probe). No GPU calls.
- In each of the three per-probe sub-loops (generate, score-base, score-variant), skip probes flagged invalid for the relevant engine. Skipped probe contributes `δ = NaN`, which the existing `_universally_valid_indices` filter already drops.

Why not A: requires loading every variant engine just to ask `max_context_length()`. That breaks Fix 4's lazy load.

Why not C: 91 of 100 long-context probes are >4K in the demo. Letting `base_engine.score(prompt, continuation)` run on those wastes ≥ 30 min of GPU per family run and may produce numerically degenerate logprobs (BF16 attention on out-of-context sequences is undefined behaviour, not an exception).

### 1.4 ProbeValidity dataclass design

Iterating on the spec:

```python
# lmdiff/_validity.py (new module — Protocol-clean, no torch)
from dataclasses import dataclass

@dataclass(frozen=True)
class EngineValidity:
    engine_name: str          # the engine's .name property
    max_context: int | None   # None = unknown / unlimited
    T_i: int                  # tokens this probe contributes for this engine
    is_valid: bool
    reason: str               # "valid" | "exceeds_context" | "tokenizer_error" | "unknown_limit"

@dataclass(frozen=True)
class ProbeValidity:
    probe_id: str             # probe.id from ProbeSet
    domain: str | None        # mirrors probe.domain (denormalized for convenience)
    per_engine: dict[str, EngineValidity]   # keys: base name + each variant name

    @property
    def valid_for_base(self) -> bool: ...
    @property
    def valid_for_all(self) -> bool: ...
    def valid_for(self, engine_name: str) -> bool: ...
```

**Iteration notes vs the spec:**

- Renamed `T_total` → `T_i` and moved it onto `EngineValidity`, not the parent
  record. Different engines may tokenize the same probe text to different `T_i`
  (CodeLlama and Llama-2 happen to agree, DeepSeek-coder will not). Storing
  `T_i` once at the parent level would lose information.

- Added `domain` to `ProbeValidity` for fast group-by-domain lookups in section 2,
  avoiding a second pass over the ProbeSet.

- Convenience predicates (`valid_for_base`, `valid_for_all`, `valid_for(name)`)
  keep call sites readable: `if not validity.valid_for_all: ...`.

- `reason` is a string enum, not a `Literal` type, so the schema is forward-
  compatible with future reasons (e.g. `"adapter_unsupported"` once steering
  lands). Documented values are stable.

### 1.5 Edge cases

1. **Tokenizer differences across variants.** `T_i` is computed per engine.
   For variants sharing base's tokenizer (`tokenizers_equivalent_to` returns
   True — the fast path), we can short-circuit by reusing the base count.
   For genuinely different tokenizers (BPB path), every engine computes its
   own `T_i`.

2. **system_prompt prefix.** The pipeline currently passes `prefix_text=v_prefix`
   to `engine.score`. `T_i` must include the prefix tokens: the same probe
   has a different `T_i` for base (no `system_prompt`) vs `system_prompt`
   variant (prefix + probe). Use `engine.token_count(prefix_text + probe_text)`
   per variant, NOT the raw probe-only count from line 432.

3. **ICL examples** (via `Config.icl_examples` / `Config.context`): same as
   system_prompt — `T_i` must include the demonstrations. Use the assembled
   prompt's token count.

4. **Generation prompt vs continuation.** `score()` scores `prefix + prompt +
   continuation`. The validity check should compare `T_full = T_prefix +
   T_prompt + T_continuation_estimate` against `max_context`. We don't know
   the continuation length until after generation. Practical recommendation:
   use `T_prefix + T_prompt + max_new_tokens` as the upper bound (worst-case)
   for the pre-generation validity check. If that exceeds context, the probe
   can't be scored — skip. For post-generation validation (an extra defensive
   layer), check actual continuation length against the remaining budget; if
   it overflows, mark the probe `exceeds_context` post-hoc and flag NaN.

---

## 2. Domain status taxonomy

The four states from the spec:

- `full` — every probe in the domain is valid for base AND every variant
- `partial` — domain has both valid-for-both and invalid-for-some probes
- `variant_only` — every probe is invalid for base, ≥1 valid for some variant
- `out_of_range` — every probe invalid for every engine

### 2.1 Computation logic

```python
def compute_domain_status(
    probes_in_domain: list[ProbeValidity],
    base_name: str,
    variant_name: str,
) -> str:
    base_valid = [p.valid_for(base_name) for p in probes_in_domain]
    var_valid = [p.valid_for(variant_name) for p in probes_in_domain]
    n = len(probes_in_domain)
    n_both = sum(1 for b, v in zip(base_valid, var_valid) if b and v)
    n_base_only = sum(1 for b, v in zip(base_valid, var_valid) if b and not v)
    n_var_only = sum(1 for b, v in zip(base_valid, var_valid) if v and not b)
    n_neither = sum(1 for b, v in zip(base_valid, var_valid) if not b and not v)

    if n_both == n:
        return "full"
    if n_neither == n:
        return "out_of_range"
    if n_both == 0 and n_var_only > 0:
        return "variant_only"
    return "partial"  # any mix
```

**Hybrid case (80 valid-for-both + 20 valid-for-variant-only):** classified as
`partial`. Rationale: `partial` is the most inclusive state and triggers the
"include valid probes, surface variance" branch in section 3. The 20
variant-only probes participate in the (v0.5.0+) `variant_only_metrics`
sub-table; for v0.4.1 share computation they're excluded from the denominator.

### 2.2 Per-variant or aggregate?

**Recommendation: per-(variant, domain) pair.**

Storage: `domain_status: dict[tuple[str, str], str]` — keyed by `(variant_name, domain_name)`. Slightly heavier (V × D entries vs D), but the alternative — aggregate per domain — collapses information that's actually different between variants. Concrete example from the demo:

- For `(yarn, long-context)`: 100/100 probes are within Yarn-128K's context, but 0/100 are within Llama-2-7B base's 4K. So this pair is `variant_only`.
- For `(chat, long-context)`: chat uses the same 4K base context — `out_of_range`.
- For `(temp_1.5, long-context)`: temp_1.5 is a runtime-mod of base — `out_of_range`.

An aggregate per-domain "long-context = ???" cannot answer "for which variant is this domain measurable?". The (V, D) pair can.

**Storage shape note:** JSON keys can't be tuples; we serialise as nested dicts:

```python
domain_status: dict[variant_name, dict[domain_name, str]]
# e.g. domain_status["yarn"]["long-context"] == "variant_only"
```

### 2.3 Share interaction per state

| status | include in pdn? | include in share denominator? | share[v][d] value |
|---|---|---|---|
| `full` | yes, all probes | yes | `pdn²/Σpdn²` |
| `partial` | yes, valid probes only | yes | `pdn²/Σpdn²` (computed on valid subset) |
| `variant_only` | no (deferred to v0.5.0 variant_only_metrics) | no | `None` |
| `out_of_range` | no | no | `None` |

---

## 3. Share calculation with validity

### 3.1 New formula

```python
def share_per_domain_for_variant(
    pdn: dict[str, float | None],
    domain_status: dict[str, str],   # for this variant
) -> dict[str, float | None]:
    valid_domains = {d for d, status in domain_status.items()
                     if status in {"full", "partial"}}
    total_sq = sum(pdn[d]**2 for d in valid_domains if pdn.get(d) is not None)
    result: dict[str, float | None] = {}
    for d, status in domain_status.items():
        if d in valid_domains and pdn.get(d) is not None:
            result[d] = (pdn[d]**2 / total_sq) if total_sq > 0 else 0.0
        else:
            result[d] = None
    return result
```

Rows still sum to 1.0 over the valid-domain subset. Invalid domains carry an
explicit `None` so consumers can render "—" instead of mistakenly showing 0.0
("no drift") for "didn't measure" (which is what `0.0` previously implied).

### 3.2 Sentinel choice (None vs NaN vs 0.0)

**Recommendation: `None`.**

| sentinel | pros | cons |
|---|---|---|
| `None` | unambiguous "not measured"; survives JSON round-trip as `null` | requires `is None` checks in viz; existing `_clean_value` recursion handles it |
| `NaN` | float arithmetic stays numeric (avoids type-narrowing pain in numpy) | indistinguishable from "measurement failed numerically"; JSON encodes as `null` anyway (lossy on round-trip without a custom encoder) |
| `0.0` | always numeric, never breaks downstream | semantically misleading: "no drift" reads identically to "didn't measure" |

`None` is the choice that preserves measurement provenance into the downstream
consumer (viz, report). Visualisation code that walks share rows must
explicitly branch on `None` → render hatched cell.

### 3.3 Edge cases

- **All domains out-of-range for a variant** (hypothetical: tiny base with
  long-context-only probe set). Recommendation: set `share_per_domain[v] = None`
  (not `{d: None for d in ...}`). Tells callers "this variant has no
  measurable drift on any domain" without burdening them with a row of Nones.
- **Zero `pdn` for every valid domain** (variant doesn't differ from base on
  anything in-context): each valid `share[v][d] = 0.0`, row sums to 0 (not 1).
  Document this; viz renders as a flat row with all-zero bars.
- **Variant has zero valid domains because everything is `variant_only`:**
  same as all out-of-range — set `share_per_domain[v] = None` and surface in
  `variant_only_metrics` (stub in v0.4.1).

---

## 4. pdn formula correction

### 4.1 New formula

```python
def pdn_per_domain_for_variant(
    deltas_by_domain: dict[str, list[float]],  # valid probes only
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for d, deltas in deltas_by_domain.items():
        if not deltas:
            result[d] = None  # no valid probes in this domain
            continue
        result[d] = math.sqrt(sum(x * x for x in deltas) / len(deltas))
    return result
```

This is the unweighted RMS of per-token CE diffs over **valid probes**. Units
clean: `δ` in `nats/token` → `σ` in `nats/token`. Equivalent to the
token-weighted RMS `sqrt(Σ T_i δ_i² / Σ T_i)` when within-domain `T_i` variance
is small — the typical post-validity case (long-context probes that mix
9K-token and 200-token are dropped together with the other out-of-context
probes, so the remaining set is more homogeneous).

### 4.2 Numerical impact on demo data

Computed from `_demo_check/runs/v032-rerendered/family_geometry.json` (7-variant,
n=497 probes).

> **Footnote on `avg_tokens_per_probe`** (cross-ref §9.6): the existing
> field — used in this analysis and in the current `_compute_per_domain_normalized` — stores **prompt-only** token counts.
> `_pipeline.py:432` populates it via `base_engine.token_count(p)` for each
> `p` in `probe_set.texts`, before any continuation is generated. The
> validity check in §1 needs `T_full = T_prefix + T_prompt +
> T_continuation`, which is **a different number** than what the demo JSON
> tabulates here. Per-domain RMS results in this section are therefore
> conservative for the validity-cutoff comparison (real `T_full` is
> larger). The "91/100 long-context probes >4K" stat is a strict lower
> bound on the count of out-of-range probes; the true count after adding
> typical continuation length (16-256 tokens depending on task) is the
> same or higher.

Per-domain token stats:

| domain | n | T_min | T_max | T_mean | probes > 4K |
|---|---|---|---|---|---|
| code | 100 | 45 | 241 | 116 | 0 |
| commonsense | 98 | 13 | 101 | 57 | 0 |
| **long-context** | 100 | 1186 | 19145 | **9010** | **91** |
| math | 99 | 33 | 168 | 72 | 0 |
| reasoning | 100 | 10 | 105 | 32 | 0 |

The 91/100 long-context probes >4K are catastrophic-failure noise for any
Llama-2-7B-based engine other than yarn/long.

Per-variant share comparison (B = current `sqrt(Σδ²/ΣT)`; B′ = current
formula with long-context dropped from `partial`/`out_of_range`; C = proposed
`sqrt(mean(δ²))` with long-context dropped):

```
                                      biggest move under
variant         B               B′ (LC dropped)     C (LC dropped, proposed)
chat            reasoning 30%   reasoning  30%      math         26%
code            code      32%   code       36%      code         50%
long            reasoning 66%   reasoning  68%      reasoning    45%
math            math      35%   math       39%      math         42%
system_prompt   commonsense 60% commonsense 61%     commonsense  60%
temp_1.5        reasoning 34%   reasoning  34%      code         35%
yarn            commonsense 51% commonsense 82%     commonsense  72%
```

**Reading:** for the 6 deterministic-decode variants, **the "biggest move"
domain is preserved or sharpened under B′ and C** (math stays math, code stays
code, long-context's true variant long stays long-context-y in spirit via
reasoning, system_prompt stays commonsense). The chat variant shifts from
reasoning to math under C — both are intuitively plausible ("instruction-tuned
shifts reasoning/math both"), worth double-checking but not destructive.

Detailed per-(variant, domain) pdn table:

```
▶ chat                                  pdn_B    pdn_B′   pdn_C
   code                                 0.111    0.111    1.200
   commonsense                          0.140    0.140    1.060
   long-context (mostly out-of-range)   0.032    0.020    [excluded]
   math                                 0.148    0.148    1.253
   reasoning                            0.153    0.153    0.870
▶ yarn
   code                                 0.020    0.020    0.218
   commonsense                          0.093    0.093    0.706
   long-context                         0.080    0.005    [excluded for base path]
   math                                 0.030    0.030    0.258
   reasoning                            0.023    0.023    0.132
```

Full table is in `docs/internal/v041_audit_analysis.py` output.

**Key observations:**

1. Within in-context domains (code/commonsense/math/reasoning), `pdn_B` and
   `pdn_B′` are essentially identical to 3 decimals — Formula B's `Σ T`
   normalization is doing the right scaling within short-prompt domains.
2. `pdn_C` is 8–11× larger numerically because it doesn't divide by `√T̄`.
   This is the dimensional fix: `pdn_C` is in `nats/token` while `pdn_B` is
   in `nats/√token` (the dimensional-mismatch unit). The **shares** are what
   matter for user-facing reports — they're scale-invariant within a row, so
   the numerical jump in pdn_C doesn't propagate to user-visible numbers.
3. Long-context's pdn under B′ (current formula, valid probes only) **drops
   to ~0** for non-long-context variants — the 9/100 in-context long-context
   probes weren't where the action was. So `partial` for `long-context` will
   show ~0 share for most variants; only yarn/long will retain interesting
   long-context shares (under their proper context windows, surfaced via
   `variant_only_metrics` in v0.5.0).

### 4.3 Numerical stability

- `len(deltas) == 0` after filter ⇒ `pdn[v][d] = None` (treated as
  out-of-range in share calculation).
- `sum(x*x) == 0.0` (variant identical to base on this domain) ⇒ `pdn = 0.0`,
  share for this domain = 0 in the row, no division-by-zero downstream.
- Total share `Σ_d' pdn[v][d']² == 0.0` (variant identical to base on every
  valid domain) ⇒ each share = 0.0; row sums to 0, not 1. Document the
  edge case so figure code doesn't normalize-to-1 unconditionally.
- Floating-point note: `δ` values are stored as float64 in `change_vectors`
  per `_pipeline.py:262` (`np.asarray(..., dtype=np.float64)`). RMS of ~100
  squared float64s has no precision concern in the demo's value range
  (`pdn_C` values 0.1 to 1.5).

### 4.4 Math derivation + Oyama citation

**δ definition** (`_pipeline.py:262, 308`):

```
δ[v][i]  =  CE_base|variant_output[i]  −  CE_variant_self|variant_output[i]
         =  (-1/T_i) Σ log P_base(y_t | x, y_<t)
            − (-1/T_i) Σ log P_v(y_t | x, y_<t)
```

Both terms are per-token mean cross-entropies. Their difference is also
per-token (units: `nats/token`). Crucially: `T_i` cancels out in expectation
when the per-token logprob distribution is stationary across the probe — so
`δ` is **T-invariant by construction**.

**Per-domain aggregation:**

For a domain `d` with valid probe set `V_d`,

```
σ_d² = mean_{i ∈ V_d}(δ_i²) = (1 / |V_d|) Σ δ_i²
σ_d  = sqrt(σ_d²)  [units: nats/token]
```

This is the unbiased per-token-CE-diff RMS estimator, treating each valid probe
as one i.i.d. sample from the per-domain drift distribution. No extra `1/√T`
factor; the T-invariance is already in `δ`.

**Oyama et al. (2025) citation:**

Oyama, R., et al. (2025). "Logarithmic Likelihood Vectors for Probabilistic
Language Models." *ACL Long Paper*. — proposes a per-prompt log-likelihood
vector `q(x) = log P(x)` as a model representation, with cosine and Euclidean
distances over the prompt distribution.

**Analogue, not derivative:** lmdiff's `δ` is the per-token version of Oyama's
per-prompt difference. Where Oyama operates on log-likelihood scalars
`log P(x_i)`, lmdiff operates on per-token cross-entropy differences `δ_i`.
The per-token form is what makes lmdiff's metric comparable across probes of
wildly different lengths (the long-context probes wouldn't fit in Oyama's
fixed-length framing). The validity framework is needed precisely because
the per-token reformulation surfaces a base-model-failure noise floor that
the prompt-scalar form doesn't expose.

---

## 5. Schema migration

### 5.1 Current schema version

`SCHEMA_VERSION = "5"` declared at `lmdiff/report/json_report.py:49`. Defined
in the schema as the unified shape including `share_per_domain` and
`magnitudes_per_domain_normalized` (post-v0.3.2 PR #11 fix). The
`geo_result_from_json_dict` loader (line 221) supports v1–v5 with on-the-fly
recomputation of v3/v4 saves via `_ensure_per_domain_normalized_views`.

**Discrepancy with Y.4 §5 wording.** PHASE_PLAN_v6.md Update 5 Y.4 component
5 says "Schema additions (GeoResult v6 → v7)." That phrasing assumes v0.4.0
already shipped a v6, which it did not — the v0.4.0 backend cutover (PR #15)
was a pipeline/engine change with no schema modification. This audit treats
the live `SCHEMA_VERSION = "5"` as the baseline. The v0.4.1 bump should
therefore be **v5 → v6**, not v6 → v7. Flagged as a planning artefact; the
spec's intent is clear from context (one bump, new fields). Confirm naming
preference in §9.2 below.

### 5.2 Proposed additions to GeoResult

```python
# lmdiff/geometry.py — append to @dataclass GeoResult
probe_validity: dict[str, ProbeValidity] = field(default_factory=dict)
"""Per-probe validity records keyed by probe.id. Empty for legacy
GeoResult instances (pre-v0.4.1) — see schema migration below."""

domain_status: dict[str, dict[str, str]] = field(default_factory=dict)
"""Per-(variant, domain) state: 'full' | 'partial' | 'variant_only' |
'out_of_range'. domain_status[variant_name][domain_name] = state."""

# share_per_domain semantics CHANGE (value type widens) — no rename:
share_per_domain: dict[str, dict[str, float | None]] = field(...)
"""Per-variant per-domain energy share. v0.4.1: None for out-of-range /
variant_only domains; float for full / partial. Rows sum to 1.0 over
valid domains."""

# magnitudes_per_domain_normalized → renamed conceptually to "pdn":
# keep the long name as the schema field for backward compat; new formula
# is sqrt(mean(δ²)) over valid probes.

# Stub for v0.5.0+:
variant_only_metrics: dict[str, dict[str, dict[str, float]]] | None = None
"""Per-(variant, domain) metrics for domains where base can't measure
but variant can. v0.5.0 will populate; v0.4.1 ships the field
stubbed-to-None."""
```

### 5.3 Schema bump v5 → v6 or additive?

**Recommendation: bump to v6, but keep v5 loader path.**

Two reasons to bump rather than do additive-with-v5-tag:

1. The semantics of `share_per_domain[v][d]` change: pre-v0.4.1 it's always a
   float; post-v0.4.1 it can be `None`. Loaders that assume `float` in their
   type annotations will break on the first `None`. Bumping the version
   signals "these values may now be None" loudly.

2. The pdn formula changes. Loading a v0.4.0 save and presenting its
   `magnitudes_per_domain_normalized` as if it were the v0.4.1 quantity would
   silently mislead. A version bump forces explicit awareness of the formula
   shift.

**Migration path:**

- v6 saves emit the full schema (validity records, domain_status, None-able
  share, corrected pdn).
- Loading a v5 save: two design options, see §9.8 for the open question.
  - **Option A (recompute on load)**: synthesize empty validity, treat all
    probes as valid, recompute `pdn` and `share` with the new formula on
    existing `change_vectors`. Numbers in memory differ from what was
    saved; user gets the v0.4.1-formula view of legacy data.
  - **Option B (preserve as saved)**: synthesize empty validity, but leave
    `pdn` and `share` as the literal saved values (pre-v0.4.1 formula).
    User gets exactly the numbers they remember; advised via
    `DeprecationWarning` to re-run for the v0.4.1 numerics.

  Whichever option is chosen, the §5.5 heuristic auto-flag for legacy
  long-context probes still emits a `DeprecationWarning` and (under Option A)
  participates in the recomputation; under Option B it's only documented in
  the warning text without altering values.

- Loading any earlier version (v1–v4): goes through the existing v5
  upgrade path first, then through the v5→v6 path above.

### 5.4 JSON serialization

- `ProbeValidity` and `EngineValidity` need a `to_json_dict` registration in
  `lmdiff/report/json_report.py` (existing pattern around line 100). Plain
  dataclass field dump, plus the dict of `EngineValidity` keyed by engine name.
- `None` in `share_per_domain` serializes to JSON `null` natively. Loader
  must accept `None` and `null`.
- `domain_status` is `dict[str, dict[str, str]]` — pure JSON, no encoder
  needed.
- `variant_only_metrics: None` serializes to `null`; absent when not set.

### 5.5 Auto-flagging legacy data

**Y.4 §5 specifies a length-heuristic auto-flag** on legacy load:
"v6 georesults load with all probes assumed valid (no validity field) →
DeprecationWarning + auto-flag long-context domain as `partial` based on
probe length heuristic." Align with the spec.

**Plan:**

1. On loading any pre-v6 save with `probe_domains` and `avg_tokens_per_probe`
   populated, walk the per-probe token counts. Use the heuristic:

   ```python
   # Default Llama-2 base context, can be overridden via metadata key
   # `base_max_context` (added in v0.4.1 saves) when present.
   LEGACY_BASE_CONTEXT = 4096
   base_max = result.metadata.get("base_max_context", LEGACY_BASE_CONTEXT)
   for i, T_i in enumerate(result.avg_tokens_per_probe):
       if T_i > base_max:
           # mark probe i invalid for base; rebuild domain_status
           ...
   ```

2. Re-derive `domain_status` per (variant, domain) from the flagged probes
   (variants on legacy data are assumed to share base's context window since
   we don't know better — same `base_max` threshold). Domains where some
   probes flag invalid become `partial`; domains where all flag invalid
   become `out_of_range`.

3. Recompute `pdn` and `share_per_domain` with the new formula on the
   valid-probe subset. Emit one `DeprecationWarning` per file:

   > "loaded GeoResult schema v{N}; pre-v0.4.1 saves lack per-engine
   > context-window metadata. Auto-flagged probes with `T_i > 4096`
   > (Llama-2 base default) as out-of-context; re-run with v0.4.1+
   > for accurate per-engine validity classification. Set
   > metadata['base_max_context'] = <int> to override the heuristic
   > threshold."

**Caveats to document in the warning text** (the heuristic IS brittle, even
when we align with the spec):

- Hard-coded `4096` is wrong for Mistral (`32768`), Llama-3 (`8192`),
  GPT-2 (`1024`), and many others. The `metadata['base_max_context']`
  override is the escape hatch.
- Multi-base setups (different `base_engine.config.model` across saves)
  can't be auto-detected without storing the base model id in metadata.
  v0.4.1 saves should add `metadata['base_max_context']` alongside
  existing `metadata['probe_set_name']` etc.
- Variant context windows aren't recovered from legacy data; the
  heuristic assumes all variants match base's threshold. This
  under-flags `variant_only` cases (Yarn-128K's 9000-token success
  isn't recoverable from a pre-v0.4.1 save). User must re-run for
  variant-only data.

A `lmdiff validate-result path/to/old.json --base-model <hf-id>` CLI helper
(out of v0.4.1 scope; tracked separately) would load the right `max_position_embeddings` per supplied model id.

---

## 6. Visualization

### 6.1 `drift_share_dual.png` changes

Current implementation: `lmdiff/viz/drift_share.py` (362 lines). Two
side-by-side heatmaps consuming `result.magnitudes_per_domain_normalized`
(left, sequential blue) and `result.share_per_domain` (right, diverging
purple-orange).

**v0.4.1 rendering for invalid cells:**

- Cell value `None` ⇒ background color `#cccccc` (neutral grey), heavy
  cross-hatch pattern, no numeric label ("—" instead).
- Cell with `partial` status but a valid float value (i.e. only some
  probes in the domain were dropped) ⇒ regular color from the colormap,
  light diagonal hatching, numeric label suffixed with "*".
- Cell with `variant_only` status ⇒ same heavy-hatch as out-of-range for
  v0.4.1 (no separate variant-only color until v0.5.0 lights up
  `variant_only_metrics`).

```python
# matplotlib sketch (NOT to be edited into production code this turn)
PATTERNS = {
    "full":         "",                # no hatch
    "partial":      "////",            # light diagonal
    "out_of_range": "xxxx",            # heavy cross-hatch
    "variant_only": "xxxx",            # same as out_of_range in v0.4.1
}
NA_BG = "#cccccc"

for i, variant in enumerate(variants):
    for j, domain in enumerate(domains):
        status = result.domain_status[variant][domain]
        share = result.share_per_domain[variant][domain]
        if share is None:
            ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                       facecolor=NA_BG,
                                       hatch=PATTERNS[status],
                                       edgecolor="black", linewidth=0.5))
            ax.text(j + 0.5, i + 0.5, "—",
                    ha="center", va="center", fontsize=9, color="dimgray")
        else:
            color = share_cmap(share)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color,
                                       hatch=PATTERNS[status],
                                       edgecolor="black", linewidth=0.5))
            label = f"{share*100:.0f}%"
            if status == "partial":
                label += "*"
            ax.text(j + 0.5, i + 0.5, label,
                    ha="center", va="center",
                    fontsize=9, color=text_color_for(color))
```

### 6.2 Heatmap / table changes

- **Per-domain magnitude heatmap (left pane of `drift_share_dual`):** same
  treatment as the share pane — `None` cells get `NA_BG` + `xxxx` hatch.
  Numeric labels suppress to "—".

- **Per-variant table** (`lmdiff/report/markdown.py` + terminal renderer):
  invalid cells render as `n/a` (terminal) or `<sub>n/a</sub>` with a
  superscript link to the legend (markdown).

- **`change_size_bars.png`** (`lmdiff/viz/change_size.py`): hatched bar
  segments for the dropped-probes portion. Existing right pane already
  shows per-token-normalized magnitude; add a small text annotation
  "(N probes excluded as out-of-range for base)" under variants where
  any domain is non-`full`.

### 6.3 Legend additions

Single legend block, top-right of figure:

```
✓ Full measurement              (no hatching)
△ Partial — some probes out of context (light hatching, "*" on label)
✗ Out of range — base context window exceeded (heavy hatching, grey fill, "—")
← Variant-only valid — [v0.5.0+]    (stub; same rendering as ✗ in v0.4.1)
```

### 6.4 Color choice

- Avoid `0.0`-mapped color for `None` — the diverging share colormap
  centers near grey, which would conflict visually.
- The neutral grey `#cccccc` is intentionally outside both heatmaps'
  active colormap ranges so users immediately read "missing".
- Hatching distinguishes `partial` from `out_of_range` even when print-only
  (no color).

---

## 7. Documentation

### 7.1 `docs/methodology/normalization.md` (new)

Outline:

1. **What δ measures.** Per-token CE difference; T-invariant by construction.
   Show the formula from `_pipeline.py:262, 308` and walk through the
   "T cancels out" derivation.
2. **Per-domain aggregation.** RMS over valid probes. Why a token-weighted
   variant collapses to plain RMS when within-domain T variance is small;
   why we don't bother with the weighted form (one less code path,
   measurably equivalent in practice).
3. **Why measurement validity is upstream of normalization.** Long-context
   probes outside base context window produce catastrophic-failure noise,
   not signal. Any per-token aggregator that includes those probes will
   surface that noise as "drift." The fix is to drop them, not to add
   another normalization step.
4. **History.** v0.3.2 PR #11's `√T̄` was an ad-hoc fix that papered over
   the long-context catastrophic-failure issue by accidentally
   down-weighting long-context contribution via `√Σ T`. v0.4.1 corrects
   methodologically.
5. **Citation.** Oyama et al. (2025) for the underlying log-likelihood-vector
   framework; lmdiff's `δ` is the per-token analogue of their per-prompt `q`.
6. **Alternatives considered.** Path B (specialization layer on top of raw
   per-token RMS) and Path C (rank-based) were considered during the
   audit chain. Validity framework was chosen because the noise was real,
   not just inconvenient: dropping invalid probes is honest measurement
   science, not a presentation trick.

### 7.2 README update

A new "Status" subsection (replacing or extending the existing v0.3.2
"share corrected" note):

```
### v0.4.1 — measurement validity (May 2026)
Per-domain shares now exclude probes that exceed the base model's context
window. Long-context probes (>4K for Llama-2-7B) are flagged
"out-of-range" rather than treated as zero-magnitude. Legacy GeoResult
files load with a DeprecationWarning and recomputed numbers; re-run for
accurate validity classification. See
docs/methodology/normalization.md and CHANGELOG [0.4.1].
```

### 7.3 CHANGELOG [0.4.1] (outline only — full text in implementation turn)

- **Breaking:** `share_per_domain[v][d]` may now be `None` for out-of-range
  domains. Pre-v0.4.1 code that assumed `float` will need a `None` check.
- **Breaking:** `magnitudes_per_domain_normalized` formula change.
  Numerical values shift; same-rank ordering preserved within in-context
  domains for the 6 deterministic variants in the demo set (see
  `docs/methodology/normalization.md`).
- **Added:** `probe_validity`, `domain_status`, `variant_only_metrics`
  (stub) fields on GeoResult.
- **Added:** `Engine.max_context_length()` Protocol method;
  `HFEngine`/`MinimalEngine` impls.
- **Added:** Schema v6.
- **Deprecated:** loading v1–v5 saves emits `DeprecationWarning` about
  validity unavailability.

### 7.4 Migration doc (`docs/migration/v040-to-v041.md`)

Outline:

- Why numbers shift: validity framework + formula correction.
- What users need to do: re-run their comparisons; v0.4.0 numbers are not
  directly comparable for variants with long-context probes.
- How to read the new fields: walk through `probe_validity`,
  `domain_status`, and the `None` semantics in `share_per_domain`.
- How to interpret `partial` vs `out_of_range` in figures.
- FAQ: "My yarn variant used to show 38% long-context share — now it's
  marked `variant_only`. Why?" → because base couldn't actually measure
  it; that 38% was noise. v0.5.0 will surface the legit yarn-on-long-context
  measurement via `variant_only_metrics`.

---

## 8. Tests

### 8.1 Unit tests (`tests/unit/`)

| file | covers |
|---|---|
| `test_validity.py` | `ProbeValidity` / `EngineValidity` construction; `valid_for_*` predicates; `compute_domain_status` against synthetic inputs covering all four states; hybrid 80/20 case |
| `test_pdn_formula.py` | New `sqrt(mean(δ²))` matches expected on synthetic deltas; `pdn = None` for empty domains; cross-check with hand-computed values |
| `test_share_with_invalid.py` | `share_per_domain` returns `None` for out-of-range / variant_only; valid rows sum to 1.0; all-out-of-range variant returns `share_per_domain[v] = None` |
| `test_schema_migration.py` | v5 JSON loads under v0.4.1 with one `DeprecationWarning`; validity stubs populated as legacy; share/pdn recomputed |
| `test_max_context_length.py` | HFEngine returns model's `max_position_embeddings` (mocked HF config); MinimalEngine default `None`; MockEngine overridable per fixture |

### 8.2 Integration tests (`tests/integration/`)

- **Existing 4-variant calibration regression** (`test_calibration_regression.py`):
  fixture **must be regenerated** alongside the 7-variant. The original §8.2
  hypothesis ("4-variant byte-equivalent post-v0.4.1") was theoretically
  wrong — see §8.4 for the empirical refutation. Two compounding changes:
  (a) the corrected pdn formula scales every (variant, domain) value by
  `√T̄_d`, breaking byte-equivalence at the 1e-6 tolerance everywhere;
  (b) the 4-variant probe set actually does include long-context (the "4"
  refers to the 4 variants `yarn / long / math / code`, not the 5-domain
  probe set, which is shared with the 7-variant) — so long-context becomes
  `partial` for every variant under the validity framework, further shifting
  the numerics. Plan: rename `calibration_v032_baseline.json` to
  `calibration_v041_4variant_baseline.json` and regenerate from a v0.4.1
  GPU run.

- **Existing 7-variant calibration regression** (`test_calibration_regression_7variant.py`):
  long-context domain becomes `out_of_range` for 5 of 7 variants
  (`variant_only` for yarn/long). Share numbers shift materially.
  **Regenerate the fixture from a v0.4.1 GPU run.** Update the variant
  parametrize lists in `_v040_7variant_spec.py`: long-context-related
  assertions need new tolerances or omission for variants where that
  domain is now `None`.

- **New `test_validity_regression.py`**: construct a synthetic GeoResult
  with a known mix of in-context and out-of-context probes; assert
  domain status assignment is correct for every (variant, domain) pair;
  assert share rows sum to 1.0 over valid domains.

### 8.3 Visualization regression

If existing image-diff tests exist (check `tests/integration/test_figures.py`
if present — current codebase doesn't appear to have one):
- Render `drift_share_dual` on synthetic input with mixed validity.
- Save reference PNGs in `tests/fixtures/figures/` and diff against rendered
  output.
- Tolerance: 1% pixel difference allowed for antialiasing on the legend
  font.

If image-diff tests don't exist: skip the regression layer in v0.4.1.
Manual sanity check is enough for one figure update.

### 8.4 Calibration fixture evolution — empirically verified

Verified via `docs/internal/v041_audit_4variant_check.py` (CPU, read-only)
on the existing `tests/fixtures/calibration_v032_baseline.json`. Results
overturn the §8.2-original hypothesis that 4-variant could stay
byte-identical:

**Within-domain T_i variance is large** (CoV = std/mean):

| domain | n | T_min | T_max | T_mean | T_std | CoV |
|---|---|---|---|---|---|---|
| code | 100 | 45 | 241 | 116.1 | 50.7 | 0.436 |
| commonsense | 100 | 13 | 101 | 57.5 | 20.3 | 0.352 |
| long-context | 100 | 1186 | 19145 | 9010.3 | 4304.7 | 0.478 |
| math | 100 | 33 | 168 | 71.7 | 25.6 | 0.358 |
| reasoning | 100 | 10 | 105 | 32.4 | 17.0 | 0.525 |

CoV 0.35–0.53 — within-domain T_i is *not* approximately constant. The §4.1
"token-weighted ≈ unweighted RMS when within-domain T_i variance is small"
caveat applies less strongly than the audit doc originally implied. Plain
unweighted RMS (Formula C) and token-weighted RMS will diverge by a few
percent within each domain. This doesn't change the design choice (Formula C
is still the right primitive) but should be acknowledged in §4.1 / methodology.

**pdn_C / pdn_B = √T̄_d everywhere** (theoretical prediction confirmed):

| domain | √T̄_d | observed C/B ratio (all 4 variants) |
|---|---|---|
| code | 10.78 | 10.78 |
| commonsense | 7.58 | 7.58 |
| long-context | 94.92 | 94.92 |
| math | 8.47 | 8.47 |
| reasoning | 5.69 | 5.69 |

Ratios are byte-identical to `√T̄` to machine precision. The pdn formula
change is **a uniform per-domain rescaling** in this dataset; share values
shift because the per-domain rescaling factors differ across domains.

**Max diffs vs the existing 1e-6 tolerance:**

- `max |pdn_C − pdn_B|` over all (variant, domain): **7.77** (long-context
  pdn for `long` variant: 0.083 → 7.85)
- `max |share_C − share_B|`: **0.85** (long-context share for `yarn`:
  37.8% → 98.9%)
- both *vastly* exceed the existing 1e-6 byte-equivalence tolerance

**Conclusion:**

1. **4-variant fixture must be regenerated.** Plan: file rename
   `calibration_v032_baseline.json` → `calibration_v041_4variant_baseline.json`,
   regenerate via the same scripted-spec pattern as 7-variant
   (`scripts/_regenerate_v041_4variant_fixture.py` + `_v041_4variant_spec.py`,
   mirroring the v0.4.0 prep work).
2. **7-variant fixture must be regenerated** for the same reason plus the
   validity-driven domain-status changes (long-context becomes `partial`
   for every variant in the 4-variant set, `variant_only` for yarn/long
   in the 7-variant).
3. **Both regenerations need GPU.** Estimated 30 min (4-variant) + 1.5 h
   (7-variant) = 2 h GPU. Same Llama-2-7B-base + variant set, just two
   separate `family()` runs with the v0.4.1 code.
4. **Test parametrize lists must be updated** to drop or special-case
   long-context byte assertions for variants where it's `partial` or
   `out_of_range`.

Numerical values per (variant, domain) are in
`docs/internal/v041_audit_4variant_check.py` output; reproduce via:

```bash
mamba run -n lmdiff python docs/internal/v041_audit_4variant_check.py
```

---

## 9. Open questions for user review

> **Note on Y.6.** PHASE_PLAN_v6.md Update 5 Y.6 specifies a two-phase lab
> feedback cycle for v0.4.1: *pre-implementation* (this audit doc shared
> with lab) and *post-implementation* (lab demo of v0.4.1 with corrected
> numbers). This PR is the pre-implementation collection point. The
> open questions below correspond to the structured ask in Y.6, plus two
> design-detail decisions surfaced during the audit itself (§9.5, §9.6,
> §9.7).

### 9.1 ProbeValidity check placement: Options A / B / C

Recommended **B (per-probe check inside `_delta_for_variant`)**. Confirm or
override.

If the user picks **A (pre-pipeline)**: we need to eagerly construct every
variant engine just to query `max_context_length()` — that reverts Fix 4's
lazy-load contract. Workaround: add a Protocol-level static helper or a
"lightweight introspection" mode where the engine class can answer
`max_context_length(config)` without loading the model. Considerable extra
design effort.

If **C (post-pipeline)**: cheapest in code but wastes ~30 min of GPU per
7-variant run scoring invalid probes. Defensible only if "v0.4.1 ships
fast, optimize in v0.4.2" is the priority.

### 9.2 Schema bump v5 → v6 or additive

Recommended **v6 bump**, with v5 loader retained. Justification: semantic
type widening of `share_per_domain` (now `float | None`) plus formula
change in `pdn` warrant the explicit version signal.

Counter-argument for additive: avoids a `SCHEMA_VERSION` change, simpler
backwards path. But hides a real type signature change from consumers.
User decision.

### 9.3 `variant_only_metrics` stub in v0.4.1 or wait for v0.5.0

Recommended: **ship the stubbed field in v0.4.1** (`None` default), populate
in v0.5.0. Rationale: the schema needs to be forward-stable so v0.4.x
users don't break when v0.5.0 starts filling the field. Costs ~10 lines
of code now to save a breaking change later.

Alternative: don't ship the stub; introduce in v0.5.0 as an additive change.
Slightly cleaner v0.4.1, but `variant_only` domain status in v0.4.1 will
have nowhere meaningful to surface the variant-side data — it's just
"None in share, no variant-only data anywhere."

### 9.4 Visualization hatching pattern

Recommended matplotlib patterns from section 6:
- `full`: no hatch
- `partial`: `////` (light diagonal)
- `out_of_range`: `xxxx` (heavy cross-hatch)
- `variant_only` (v0.4.1 stub): same as `out_of_range`

Alternative palette: dotted (`..`) for partial, dashed (`---`) for
out_of_range — softer visual but harder to distinguish at small sizes.

User decision. Whatever lands becomes the legend in
`docs/methodology/normalization.md` figure-style guide.

### 9.5 (added during audit) Renaming `magnitudes_per_domain_normalized` to `pdn`

The spec calls the field `pdn` colloquially throughout. The actual field name
is `magnitudes_per_domain_normalized` (`geometry.py:186`). Three options:

1. **Keep the long name** (status quo). Backward-compatible. Long for users
   to type.
2. **Add a `pdn` alias** as a property that returns the same dict. Both work
   in user code; can phase out the long name in v0.5.0+.
3. **Rename hard** with `pdn` and remove the long name. Breaking, requires
   migration in every user notebook.

Recommended: **option 2 (alias)**. Costs one property method, removes user
friction, defers the rename decision.

### 9.6 (added during audit) `T_i` definition for validity check: prompt-only vs prompt+continuation

The current code records prompt-only `T_i` (`_pipeline.py:432`). For
validity we need `T_prefix + T_prompt + T_continuation`. The continuation
is unknown pre-generation. Section 1.5 recommended using
`T_prefix + T_prompt + max_new_tokens` as the worst-case bound. User
confirmation that this is the right conservative choice (rather than
`T_prefix + T_prompt + mean_continuation_length` or post-hoc check only).

### 9.7 (added during audit) HFEngine fallback when `max_position_embeddings` missing

Some HF model configs use `n_positions` (GPT-2 family) or `max_seq_len`
(some custom configs). Recommended fallback chain in `HFEngine.max_context_length()`:

```python
cfg = self._model.config
for attr in ("max_position_embeddings", "n_positions", "max_seq_len"):
    val = getattr(cfg, attr, None)
    if val is not None:
        return int(val)
return None  # unknown
```

Confirm naming order and behaviour for the "unknown" return.

### 9.8 v5 loader behavior on legacy data: recompute vs preserve

Two options for what `geo_result_from_json_dict` does when it encounters
a v5 save under v0.4.1:

- **Option A — recompute on load** (current §5.5 plan): re-derive
  `pdn` and `share_per_domain` using the v0.4.1 formula on the saved
  `change_vectors`. Auto-apply the §5.5 length heuristic to flag
  long-context probes. Numbers in memory differ from the saved
  numbers. `DeprecationWarning` notes the recomputation. This is
  what `_ensure_per_domain_normalized_views` already does for the
  v3→v5 path (precedent).

- **Option B — preserve saved values** (user-leaning): leave `pdn`,
  `share_per_domain`, `magnitudes_normalized` exactly as saved.
  Synthesize empty `probe_validity` and full-status `domain_status`
  to satisfy the new schema, but don't touch the existing numbers.
  `DeprecationWarning` reads "values are pre-v0.4.1 formula; re-run
  for v0.4.1 numerics." User opening an old JSON gets the numbers
  they remember.

**Trade-off:** Option A is consistent with the v3→v5 precedent (lmdiff
historically prefers "give the user the corrected number on load");
Option B is more conservative and matches the principle "saved means
saved." User leans Option B — the v0.3.2 → v5 recompute was driven by
a known bug in the saved formula, while v0.4.1's change is an
improvement to a non-buggy formula. Re-presenting different numbers
under the same field name without re-execution may surprise users.

**Implementation diff:** Option B is *less* code — skip the recompute
branch entirely on v5 load, just attach validity stubs. Option A reuses
the existing `_ensure_per_domain_normalized_views` machinery with one
extra step (apply length heuristic before computing).

**Recommend Option B**, given user lean and the principle. Preserves
backward number-stability for users who don't re-run.

### 9.9 Mistral sliding-window attention validity

Mistral-7B's `max_position_embeddings` is 32768 but the model uses a
sliding-window attention mechanism that bounds each token's effective
context to 4096 — long prompts can be scored without crash, but
quality degrades within the sliding window in ways the validity
framework doesn't capture.

Under v0.4.1 as designed:

- `HFEngine.max_context_length()` reads `max_position_embeddings = 32768`
- A 30000-token probe is `is_valid = True` for Mistral-7B-base
- Per-token CE is computed without error but reflects degraded
  attention, not real per-token drift signal

**Question:** is this acceptable for v0.4.1? Defer
"sliding-window-aware validity" to v0.5.0+?

**Recommendation:** accept v0.4.1 as the simple `max_position_embeddings`
threshold. Document the Mistral caveat in
`docs/methodology/normalization.md` and CHANGELOG. v0.5.0+ adds an
optional per-engine override (e.g. `Engine.effective_context_length()`
that returns the sliding-window size when applicable, falling back to
`max_context_length` otherwise) plus a finer status (`degraded` —
in-context but quality-suspect — alongside `partial` / `out_of_range`).

Same pattern applies to other quality-degradation cases beyond hard
context limits (e.g. flash-attention numerical edge cases at very
long sequences, RoPE extrapolation in Yarn beyond original training
distribution). v0.4.1 ships the validity skeleton; quality-degradation
flags are a separate v0.5.0+ feature.

---

**End of design audit.**

When the open questions are resolved, implementation in a separate turn lands:

- `lmdiff/_validity.py` (new module, ~150 LOC)
- Engine Protocol additions (`max_context_length` + HFEngine/MinimalEngine impls)
- `_pipeline._delta_for_variant` validity-skip branch
- `geometry.py` GeoResult new fields + `_compute_per_domain_normalized` rewrite
- `report/json_report.py` v6 serialization + v5 loader
- `viz/drift_share.py` hatched-cell rendering
- Five new unit test files + the 7-variant fixture regeneration
- `docs/methodology/normalization.md` + migration doc + CHANGELOG entry

Estimated 2 days CC work + ~1.5h GPU re-run + 1 day user review.
