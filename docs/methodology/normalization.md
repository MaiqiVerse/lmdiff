# Per-Domain Normalization in lmdiff (v0.4.1+)

> **Status:** Stable. The formulas described here ship in v0.4.1 and replace
> the v0.3.2 √T̄ form. The change is breaking for downstream consumers who
> compared numerical pdn / share values across versions; see
> [`docs/migration/v040-to-v041.md`](../migration/v040-to-v041.md).

## What δ measures

For a base config `B` and a variant config `V`, lmdiff produces a per-probe
"change scalar" `δ_i` defined as the difference of two per-token cross
entropies:

```
δ_i = (-1/T_i) · Σ_t log P_B(y_t | x_i, y_<t)        ← base scoring variant's output
    − (-1/T_i) · Σ_t log P_V(y_t | x_i, y_<t)        ← variant scoring its own output
```

Both terms are **per-token mean cross-entropies** in nats per token.
Their difference inherits the same units. The sum/T construction means
`δ_i` is **T-invariant by construction**: a probe whose continuation is
twice as long but whose per-token logprob distribution is identical
produces the same `δ_i` as the shorter probe.

The full computation lives in [`lmdiff/_pipeline.py:262, 308`][pipe-ce].

[pipe-ce]: ../../lmdiff/_pipeline.py

## Per-domain aggregation (v0.4.1, Q9.10 Formula A)

To aggregate per-probe `δ_i` into a per-domain summary that's comparable
across domains of different sizes, lmdiff computes the **plain unweighted
RMS** over the *valid* probe set:

```
σ_d = sqrt(mean_{i ∈ d ∧ valid}(δ_i²))      [units: nats/token]
```

stored as `result.magnitudes_per_domain_normalized[v][d]` (alias `result.pdn`).

When the (variant, domain) pair is `out_of_range` or `variant_only` per
the validity framework (next section), the value is `None` rather than
a number — distinguishing "didn't measure" from "measured zero drift."

The corresponding share row uses squared-share normalization over the
valid domains only:

```
share[v][d] = pdn[v][d]² / Σ_{d' valid} pdn[v][d']²    when status ∈ {full, partial}
share[v][d] = None                                       when status ∈ {out_of_range, variant_only}
```

Valid rows sum to 1.0 over themselves.

The overall per-variant magnitude is the per-domain RMS over valid
domains:

```
magnitudes_normalized[v] = sqrt( (1/D_valid) · Σ_{d valid} pdn[v][d]² )
```

Each valid domain weighted equally — a single long-prompt domain doesn't
dominate the overall number.

## Why measurement validity is upstream of normalization

Long-context probes (e.g. `longbench_2wikimqa` averaging ~9000 tokens)
are larger than the trained context window of common base models
(Llama-2-7B: 4096 tokens). Beyond that window, RoPE position embeddings
extrapolate to untrained ranges, attention patterns degrade
catastrophically, and per-token cross-entropy inflates substantially —
**for both the base and the variant**.

Under any per-token aggregator (Formula A, token-weighted RMS,
ranks, …), this catastrophic-failure noise gets surfaced as "drift,"
because per-token CE diff is genuinely large at out-of-context positions.
But it's not the *kind* of drift the user is asking about — it's not
"variant V specializes more on long-context than on math"; it's "neither
base nor variant could read the prompt, and they failed differently."

The fix has to be at the **measurement layer**, not the normalization
layer. v0.4.1 introduces per-(engine, probe) validity records:

```python
@dataclass(frozen=True)
class EngineValidity:
    engine_name: str
    max_context: int | None        # from Engine.max_context_length()
    T_i: int                       # T_prefix + T_prompt + max_new_tokens
    is_valid: bool                 # T_i ≤ max_context (or max_context is None)
    reason: str                    # "valid" | "exceeds_context" | "unknown_limit"

@dataclass(frozen=True)
class ProbeValidity:
    probe_id: str
    domain: str | None
    per_engine: dict[str, EngineValidity]
```

Probes flagged invalid for the relevant engine are skipped at the
per-probe sub-loops in `_pipeline._delta_for_variant`; their `δ` values
are NaN and the global `_universally_valid_indices` filter drops them
from `change_vectors`. The per-(variant, domain) status follows from the
per-probe records:

| status | meaning | share treatment |
|---|---|---|
| `full` | every probe in the domain valid for both base and variant | included; numeric share |
| `partial` | some probes valid, some not | included; numeric share computed on valid subset; "*" suffix in viz |
| `variant_only` | every probe invalid for base, ≥1 valid for variant | excluded; share = None; surface via v0.5.0+ `variant_only_metrics` |
| `out_of_range` | every probe invalid for every engine | excluded; share = None |

## History — v0.3.2 PR #11 → v0.4.1

The v0.3.0–v0.3.2 share formula was `‖δ_d‖² / Σ ‖δ_d'‖²` — raw L2
length-weighted. A 100-token-per-probe long-context domain would
dominate ~99 % of every variant's share regardless of per-token drift.

v0.3.2 PR #11 introduced `pdn = sqrt(Σ_{i∈d} δ_i² / Σ_{i∈d} T_i)`
intending "per-token RMS." This had two problems:

1. **Dimensional inconsistency.** With `δ` in `nats/token`, the formula
   evaluates to `nats / token^1.5` — not a meaningful unit. The
   formula was derived under the implicit assumption that `δ` is
   total CE difference per probe (units: nats), which the
   implementation never produced.

2. **Self-consistent mockup.** The v6 §13 calibration mockup was
   hand-derived from the same formula, then the implementation
   produced numbers matching the mockup, then tests asserted
   implementation matches mockup. Validation only confirmed
   implementation-vs-spec, not spec-vs-truth. See [LESSONS L-033][L-033].

[L-033]: ../../LESSONS.md

The √T̄ over-correction *incidentally* mitigated long-context dominance
(the divide-by-Σ-T term scales down long-context contributions). But it
worked by the wrong mechanism, and as a result the published v0.3.2
showcase numbers (e.g. "long → reasoning 66%") were
self-consistent-but-not-validated rather than methodologically grounded.

v0.4.1 corrects this in two parts:

- **Validity framework upstream**: long-context probes that exceed base's
  trained context window are excluded from per-domain aggregation (not
  re-normalized). This is the methodologically clean answer to "what
  about long-context dominance?"
- **Plain unweighted RMS**: with the noise probes excluded, the
  dimensionally clean Formula A `sqrt(mean(δ²))` gives a meaningful
  per-token RMS in `nats/token`. No magic √T̄ correction needed.

## Citation analogue — Oyama et al. (2025)

Oyama, R., et al. (2025). "Logarithmic Likelihood Vectors for
Probabilistic Language Models." *ACL Long Paper*. — proposes a
per-prompt log-likelihood vector `q(x) = log P(x)` as a model
representation, with cosine and Euclidean distances over the prompt
distribution.

lmdiff's `δ_i` is the **per-token analogue** of Oyama's per-prompt
difference. Where Oyama operates on log-likelihood scalars `log P(x_i)`,
lmdiff operates on per-token cross-entropy differences `δ_i`. The
per-token reformulation makes lmdiff comparable across probes of wildly
different lengths (the long-context probes wouldn't fit in Oyama's
fixed-length framing). The validity framework is needed precisely
because the per-token reformulation surfaces a base-model-failure noise
floor that the prompt-scalar form doesn't expose.

## Alternatives considered

During the v0.4.1 design audit (`docs/internal/v041_validity_design.md`)
three alternative paths were evaluated and not chosen:

- **Path B — Formula A + post-hoc specialization layer.** Compute pdn
  with Formula A; add a specialization metric `pdn[v][d] /
  geomean_v(pdn[v][d])` that divides out the per-domain "every variant
  drifts this much" baseline. The specialization metric does answer
  "where is this variant unusual," but the underlying pdn it builds on
  still exhibits the long-context catastrophic-failure noise from
  Section 3 above. Validity framework first, specialization later
  (v0.5.0+).

- **Path C — Rank-based shares.** Instead of per-token RMS, rank
  variants within each domain ("yarn is rank 1 in long-context;
  CodeLlama is rank 1 in code"). Robust to outliers, no formula
  decision needed. But ranks discard magnitude information: a variant
  that's barely #1 in code reads the same as one that's overwhelmingly
  #1. Rejected as too lossy for the headline showcase.

- **Token-weighted RMS** (`sqrt(Σ T_i δ_i² / Σ T_i)`, Q9.10 option B).
  Statistically principled — when probe T_i varies within a domain, the
  ML-optimal weighting puts more trust in longer probes. Empirically
  the share difference vs Formula A is up to 5.4pp on the 4-variant
  baseline (`docs/internal/v041_audit_pdn_AB_check.py`). Rejected for
  v0.4.1 because: the iid-token assumption ML-weighting rests on
  doesn't strictly hold; Formula A is non-parametric and robust to
  outlier probes; and the simpler formula is easier to explain in user
  docs. Could be revisited in v0.5.0+ if lab feedback prefers
  variance-weighted view.

## Reproducibility

The CPU-side empirical analysis used to validate the formula is
checked in:

- `docs/internal/v041_audit_analysis.py` — 7-variant Formula B vs C
  comparison
- `docs/internal/v041_audit_4variant_check.py` — 4-variant
  empirical confirmation that Formula B and Formula A differ by
  factor √T̄_d (max pdn diff 7.77, max share diff 0.85 vs 1e-6
  calibration tolerance — fixture regen required)
- `docs/internal/v041_audit_pdn_AB_check.py` — plain (A) vs
  token-weighted (B) RMS empirical share-impact (max 5.4pp under the
  v0.4.1 view)

Re-run any of these on a CPU-only box:

```bash
mamba run -n lmdiff python docs/internal/v041_audit_analysis.py
```
