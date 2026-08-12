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
| `partial` | valid-for-both subset meets `min_valid_fraction` | included; numeric share computed on valid subset; "*" suffix in viz |
| `variant_only` | valid-for-both below the floor, variant-only coverage meets it | excluded; share = None; surface via v0.5.0+ `variant_only_metrics` |
| `out_of_range` | nothing measurable, or valid-for-both below the floor with no variant coverage | excluded; share = None |

## Minimum valid fraction

Dropping invalid probes is necessary but not sufficient. A domain can
survive the per-probe filter with so few probes left that its aggregate
is no longer a measurement of the domain — it is a measurement of
whichever probes happened to be short enough.

The Llama-2-7B calibration is the worked example. Of 100
`longbench_2wikimqa` probes, **9** fit inside the 4096-token base
window (`T_i` distribution: min 1202, median 7849, max 19161). Those 9
survivors are not a random sample — they are the short left tail, the
probes that test long-context capability *least*. Aggregating them
produced long-context shares of 27.6 % (`temp_1.5`) and 18.4 % (`chat`),
plotted beside domains measured on all 100 probes, with only a hatch
pattern to mark the difference.

This is the same failure the validity framework exists to fix (L-033),
one level quieter: a number that is arithmetically correct and
substantively meaningless.

So `compute_domain_status` takes a **`min_valid_fraction` floor**,
default `0.5`. A domain whose valid-for-both subset falls below the
floor reports no share at all:

- variant-only coverage also meets the floor → `variant_only`
- otherwise → `out_of_range`

Either way `share` and `pdn` become `None`, and the domain is excluded
from the overall normalized magnitude rather than contributing a share
built on a handful of probes.

**Why 0.5.** It is a round majority-rule choice, not a derived constant,
and it should be read as such. The defensible claim is comparative, not
absolute: *some* floor is needed, and the implicit alternative is worse.
With no floor the effective threshold is `1/n` — one surviving probe out
of a hundred still yields a plotted share — and unlike an explicit
constant, `1/n` is invisible at the call site and drifts silently with
probe count. A named parameter with a documented default can at least be
argued with, and overridden:

```python
compute_domain_status(probes, base, variant, min_valid_fraction=0.0)
```

`0.0` disables the floor and restores pre-v0.4.1 behaviour exactly;
`1.0` collapses `partial` entirely, admitting only fully-measured
domains. Both are supported and unit-tested.

**Effect on the v0.4.1 calibration.** Every non-long-context domain is
`full` (100/100 valid) and unaffected. On long-context, all seven
variants move off `partial`:

| variant | pre-floor | post-floor | valid-for-both / variant-only |
|---|---|---|---|
| `yarn` | `partial` | `variant_only` | 9 / 91 |
| `long` | `partial` | `variant_only` | 9 / 91 |
| `code` | `partial` | `variant_only` | 9 / 80 |
| `math` | `partial` | `out_of_range` | 9 / 0 |
| `chat` | `partial` | `out_of_range` | 9 / 0 |
| `temp_1.5` | `partial` | `out_of_range` | 9 / 0 |
| `system_prompt` | `partial` | `out_of_range` | 9 / 0 |

The long-context column becomes `None` for every variant, and each
variant's remaining four domains renormalize to sum to 1.0.

## What moves when the formula moves

A formula change does not stop at the formula. Two downstream constants
were calibrated against Formula B's numeric range and had to be
converted with it; both were missed on the first pass and caught only by
looking at the rendered figure.

**Drift-magnitude colour bins** (`lmdiff/viz/drift_share.py`,
`_DRIFT_BIN_EDGES`). The left pane of `drift_share_dual.png` bins pdn
into barely / small / moderate / big / huge. Through v0.4.0 the edges
were `0.025 / 0.05 / 0.10 / 0.20`. Formula A exceeds Formula B by
`√T̄_d`, and short-prompt domains have `T̄` of roughly 30–120 tokens, so
`√T̄` lands between 5.7 and 11. Shifting the identical ×2 ladder one
decade up — to `0.25 / 0.5 / 1.0 / 2.0` — is therefore a **unit
conversion, not a retune**: same structure, same spacing, same labels,
restated in the new units. Left unconverted, 26 of 28 measured cells in
the v0.4.1 calibration fell into the top bin and the pane carried no
information at all.

The corollary matters more than the constant: these edges are **not** to
be fitted to whichever dataset is in front of you. An empty `barely` bin
is itself a result — it says nothing in this probe set is near-identical
to base. A ×8 ladder was rejected during review for exactly this reason:
its only argument was that it filled every bin, which is a cosmetic
argument. If the pdn formula changes again, convert these edges by the
same factor rather than re-fitting them.

**Specialization margin** (`lmdiff/_findings.py`,
`_SPECIALIZATION_PEAK_MARGIN`). `SpecializationPeakFinding` fires when a
variant's top domain exceeds a 30 % share. That test asks whether the
peak is *large*; it never asked whether the peak was *ahead*. Dropping
long-context renormalized every row over four domains instead of five,
and `chat`'s peak rose from 29.9 % (silent, below the floor) to 32.0 %
(a specialization claim) — while leading its runner-up by 2.5 pp. The
floor fix had manufactured a finding for one of the two variants with
the weakest claim to one.

So the peak must now also lead the runner-up by `_SPECIALIZATION_PEAK_MARGIN`,
5 pp. There is no derivation for 5 pp, but there is an anchor: the
calibration regression grants sample-decode variants a 2 pp tolerance on
this very metric, which is measurement noise we have already committed
to in writing. A claim whose entire margin sits under twice the noise we
tolerate should not be stated as a result. 5 pp is 2× that tolerance,
rounded up.

Suppression alone would have been the wrong fix. A variant with no
finding is indistinguishable from a variant that was never measured, so
below the margin the variant now reports `UndifferentiatedFinding`:

```
chat: no dominant domain (math 32%, code 29%, commonsense 23%)
temp_1.5: no dominant domain (commonsense 31%, reasoning 27%, code 24%)
```

That is a positive result, not the absence of one. A modification
touching several domains at once — instruction tuning, a decoding change
— genuinely has no single target, and reporting the spread says more
than naming whichever domain won by a point. Every variant with any
measurable domain now yields exactly one verdict: a named peak or an
explicit "no dominant domain".

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
