# lmdiff-kit Phase Plan v6 (Locked, Final + Integrated Design Guide)

> **Status**: Body last realigned 2026-05-13 (Update 7). Supersedes v1–v5.
> **Goal**: From v0.2.4 to v1.0.0 LTS. Original estimate 6–8 months; revised to **10–13 months** after Phase 1 and Phase 2 both overran — see §19.
> **Primary users**: Application engineers and ML researchers, equally weighted.
> **Discipline**: Public API ≤ 30 symbols at v1.0. Configuration class is the unifying abstraction. Capability extensibility prioritized over breadth. Experimental metrics opt-in only.

This document combines the phase plan (sections 1–11), the report design style guide (section 12), a complete CLI summary example (section 13), and the reproducible source code for four anchor figures (section 14).

**How to read this document.** Sections 1–20 are the plan body. Updates 1–7 follow, append-only, in chronological order. When the two disagree, **the latest Update wins** — the body is periodically realigned but Updates are never rewritten, because the record of what was believed when is itself evidence (see Update 6 Z.6). Body sections carrying stale claims are marked inline with a pointer to the Update that supersedes them.

---

## Table of contents

1. Design principles
2. Phase summary
3. Phase 0 — Backlog cleanup (complete)
4. Phase 1 — API stabilization + Engine + Configuration + Application UX  ← v0.3.0–v0.3.2 (complete)
5. Phase 2 — Validity + run config + probe taxonomy  ← v0.4.0–v0.4.7 (in progress)
6. Phase 3 — Documentation site  ← v0.6.0
7. Phase 4 — Output, behavioral, calibration metrics  ← v0.7.0
8. Phase 5 — Representation + steering + ID + multi-GPU  ← v0.8.0
9. Phase 6 — Trajectory + cloud distance + custom Engine validation  ← v0.9.0
10. Phase 7 — Polish + external feedback intake + perf tuning  ← v0.10.0
11. Phase 8 — v1.0.0 LTS hardening + final review
12. Report design style guide
13. CLI summary example *(numbers historical — see header note in §13)*
14. Reproducible figure code
15. Metrics roadmap summary
16. Configuration roadmap summary
17. Cross-phase work patterns
18. Risks and mitigations
19. Time budget summary
20. Immediate next steps

**Updates** (append-only, chronological):

| # | Date | Subject |
|---|---|---|
| 1 | 2026-04-25 | Literature validation of Llama-2 case study findings |
| 2 | 2026-04-25 | Master bibliography |
| 3 | 2026-04-30 | Phase 1 retrospective + v0.4.0 scope adjustment |
| 4 | 2026-05-11 | v0.4.0 ship retrospective (Phase 2 commit 4.0) |
| 5 | 2026-05-11 | Methodology critique + Phase 2 reorganization |
| 6 | 2026-05-11 | v0.4.1 implementation record + v0.5.0 scoping |
| 7 | 2026-05-13 | Run configuration as a serializable artifact |

---

## 1. Design principles

1. **Few stable public symbols.** v1.0 caps stable public API at ~30. `lmdiff.contrib.*` holds experimental work.
2. **Three first-class output channels.** CLI text, Python API, JSON. Every capability reaches all three. *(A fourth artifact — the run-configuration YAML of Update 7 — is not a channel but a serialization of the input, emitted alongside every report.)*
3. **Configuration is the unit of comparison.** Not "model"; a `Config` packages weights + adapter + context + decode + steering + metadata. Compares any two Configs that share enough capability surface. *(Update 7 makes this literal: the `Config` set for a run round-trips through YAML, so the unit of comparison is also the unit of record.)*
4. **`GeoResult` / `FamilyResult` are immutable.** All renderers (terminal / markdown / HTML / figures / JSON) read from result objects. Single source of truth.
5. **Data-driven narrative.** `lmdiff.geometry._findings` is the only place that produces interpretation strings. All renderers call into it.
6. **Per-PR doc parity.** Every public API change ships with code, tests, docstring, and a reference page in the same PR.
7. **API stabilization, not freeze.** v0.3.0 locks API *shape* (which functions/classes exist, top-level names). Signature-level adjustments continue through v0.8.x with deprecation cycles. **Hard freeze only at v1.0.0** — public API + GeoResult schema + Engine/Metric protocols all enter LTS no-break commitment.
8. **Experimental metrics opt-in only.** Metrics in `contrib` or marked experimental do not run by default. User must explicitly request them. Reduces silent unreliable results.
9. **Capability negotiation is the abstraction layer.** Engine reports what it supports; metric requires capability set; mismatch → clear error, not silent garbage. This is how custom Engine implementations slot in.
10. **Don't carry deprecations forever.** Field deprecated for one minor version, then deleted.

---

## 2. Phase summary

| Phase | Version | Status | Theme | Commits |
|---|---|---|---|---|
| 0 | v0.2.3 / v0.2.4 | **Complete** | Backlog cleanup + critical fixes | (shipped) |
| 1 | v0.3.0 → v0.3.2 | **Complete** | API stabilization + Engine + Configuration **+ Application UX** | 12 + 2 fat releases |
| 2 | v0.4.0 → v0.4.7 | **In progress** | Validity framework + run config + probe taxonomy + builtin probe sets | 7 |
| 3 | v0.6.0 | Pending | Documentation site | 8 |
| 4 | v0.7.0 | Pending | Output + behavioral + calibration metrics | 7 |
| 5 | v0.8.0 | Pending | Representation + steering + ID + multi-GPU | 11 |
| 6 | v0.9.0 | Pending | Trajectory + cloud distance + custom Engine | 8 |
| 7 | v0.10.0 | Pending | Polish + external feedback intake + perf tuning | 5 |
| 8 | v1.0.0 | Pending | LTS hardening + final review | 4 |

**v0.5.0 is not a phase.** It is a cross-cutting release between Phase 2 and Phase 3 carrying the v0.2.x removals (`ModelDiff`, `run_family_experiment`, `InferenceEngine`, `ChangeGeometry`, `lmdiff.config.Config`, and the paper-tier figures) plus the variant-only measurement work deferred out of v0.4.1. See Update 6 Z.4. Every phase from 3 onward therefore shifts one minor version later than originally planned.

**Phase 2 grew twice.** From 5 commits to 6 when the √T methodology critique forced the validity framework in (Update 5 Y.4), then to 7 when run-configuration serialization was adopted (Update 7 AA.7). Current commit ordering is in AA.7; §5.5 below predates both renumberings and is annotated accordingly.

### Shipped so far

| Version | Date | Contents |
|---|---|---|
| v0.3.0 | 2026-04 | Phase 1 core: public API, Config, Engine Protocol, GeoResult schema, report tiers |
| v0.3.1 | 2026-04 | Hotfix |
| v0.3.2 | 2026-05 | Fat release: engine reuse, share formula "fix" (PR #11 — later found to be an over-correction, see Update 5), figure polish |
| v0.4.0 | 2026-05-11 | Phase 2 commit 4.0: backend cutover to HFEngine + `_pipeline`; 4 bugs found and fixed pre-ship (Update 4) |
| v0.4.1 | 2026-05-12 | Phase 2 commit 4.1: measurement validity framework + pdn correction, `min_valid_fraction` floor (Updates 5, 6) |
| v0.4.2 | (in flight) | Presentation-layer sweep: 4 of 10 output surfaces wrong after the v0.4.1 rescale, one numerically. Plus housekeeping deferred from v0.4.1 (Update 6 Z.5) |

### Key change from prior versions

**v0.4.0 (Application UX) is folded into v0.3.0.** Rationale: the application-tier deliverables (CLI summary, application-tier figures, HTML report, findings) become demonstrable as soon as v0.3.0 ships. This makes v0.3.0 a complete demo-ready release, suitable for lab-level introduction and feedback collection. The 5 commits originally earmarked for v0.4.0 join the 7 commits of v0.3.0 for a 12-commit release. The version number originally reserved for v0.4.0 (probe taxonomy) becomes v0.4.0; subsequent versions shift down by one.

*(That renumbering has since been superseded twice more — see the Phase 2 note above and Update 7 AA.7.)*

---

## 3. Phase 0 — Backlog cleanup (COMPLETE)

**Versions**: v0.2.3, v0.2.4 (both shipped).

**Carry-forward to Phase 1**:
- Lazy `__getattr__` mechanism in `lmdiff/__init__.py` must be preserved when redesigning the public API.
- `[tool.setuptools.package-data]` packaging config preserved.

---

## 4. Phase 1 — API stabilization + Engine + Configuration + Application UX

**Version**: v0.3.0 (shipped 2026-04), plus v0.3.1 hotfix and v0.3.2 fat release
**Estimated weeks**: 6–8 · **Actual: ~9**
**Status**: **Complete.** Retrospective in Update 3 (W.1–W.8).
**Theme**: Lock the public API *shape* through v1.0. Define Configuration to cover all v0.7-target config types. Define Engine protocol with capability-based custom backend support. Ship the complete application-tier user experience (CLI summary, HTML report, dual-view figures, findings).

This is the largest single release in the plan. The rationale: by the end of v0.3.0, lmdiff is **demonstrable end-to-end at lab level**. Researchers and application engineers can both use it productively, even though some advanced metrics arrive in later phases.

> **What the sections below do and do not describe.** §4.1–§4.5 describe what shipped and are accurate. §4.6 (GeoResult schema) describes v5 and has since been superseded by v6 — see the note there. §4.3 (Engine protocol) lists the v0.3.0 method set; three methods were added in Phase 2 and are noted inline.

### 4.1 Configuration class (full end-state shape)

`Config` is the unit of comparison. Phase 1 defines the field set; specific decoders/loaders implemented incrementally across phases. *(Phase 2 commit 4.2 makes this field set serializable — see Update 7. Any field added here after that point must have a YAML representation or declare itself non-serializable.)*

```python
@dataclass(frozen=True)
class Config:
    # ─── Identity ──────────────────────────────────────────
    model: str                          # HF id, local path, or custom Engine instance
    name: Optional[str] = None          # display name; defaults to model id
    
    # ─── Weights modifications ─────────────────────────────
    adapter: Optional[AdapterSpec] = None      # LoRA / QLoRA / IA³ etc.
    quantization: Optional[QuantSpec] = None   # INT8 / INT4 / GPTQ / AWQ
    pruning: Optional[PruneSpec] = None        # spec or "load already-pruned"
    
    # ─── Context ───────────────────────────────────────────
    system_prompt: Optional[str] = None
    icl_examples: Optional[list[ICLExample]] = None    # few-shot
    context: Optional[list[Message]] = None             # multi-turn history
    soft_prompts: Optional[np.ndarray] = None           # learned prompt embeddings
    kv_cache_compression: Optional[KVCacheSpec] = None  # H2O / KV-quant etc.
    
    # ─── Decoding ──────────────────────────────────────────
    decode: DecodeSpec = field(default_factory=DecodeSpec)
        # Includes: strategy (greedy/sample/beam/best_of_n/self_consistency)
        #           temperature, top_p, top_k, num_samples, max_new_tokens
        # NOTE: best_of_N and self_consistency drive sampling clouds —
        #       Phase 6 cloud-distance metrics consume Configs with these.
    
    # ─── Steering ──────────────────────────────────────────
    steering: Optional[SteeringSpec] = None
        # Concrete shape locked Phase 5; v0.3.0 has only the slot + abstract spec.
    
    # ─── Metadata ──────────────────────────────────────────
    tokenizer_id_override: Optional[str] = None
    capabilities_required: frozenset[str] = frozenset()
```

**NOT in Configuration for v1.0**:
- Explicit CoT spec (covered by `system_prompt` + `decode`; CoT-specific spec considered for cloud-distance scope).
- Test-time training state (v2.0).
- Agentic scaffolding (v2.0; reserved capability `agentic` only).

### 4.2 Top-level functions

```python
result = lmdiff.compare("gpt2", "distilgpt2")
result = lmdiff.compare(Config("gpt2"), Config("gpt2", system_prompt="..."))
result.save("out.json")
result.to_html("out.html")
result.figures("./figs/")

result = lmdiff.family(
    base=Config("meta-llama/Llama-2-7b-hf"),
    variants={...},
    probes="lm-eval:hellaswag,arc_challenge,gsm8k",
    n_probes=100,
    metrics="default",  # experimental opt-in via explicit list
    max_new_tokens=16,
    task_overrides={"gsm8k": {"max_new_tokens": 256}},
)
result.geometry.drift / .share / .direction / .zscore
result.accuracy
result.findings
print(result)  # __repr__ → terminal summary (5-layer)
```

### 4.3 Engine protocol

See ENGINE_PROTOCOL_SPEC.md for full signatures. Brief:

**Required methods**: `score`, `generate`, `close`

*Three further required methods were added during Phase 2, all to move computation out of the geometry layer and behind the Protocol:*
- *`token_count(text) -> int` (v0.4.0) — replaced direct `engine._tokenizer` access from `geometry.py`*
- *`tokenizers_equivalent_to(other) -> bool` (v0.4.0) — cross-engine tokenizer equivalence, with a canary-string check on HFEngine for the slow/fast Llama case (L-011)*
- *`max_context_length() -> int | None` (v0.4.1) — the validity framework's threshold; `None` means unknown, treated as unlimited. See Update 5 Y.4 decision Q9.7 for the config-attribute fallback chain.*

*`score` and `generate` also gained keyword arguments in v0.4.0 — `top_k` and `prefix_text` — after the backend cutover revealed that the old path passed decode parameters the new one silently dropped, and that split versus joint tokenization of a prompt prefix are not byte-equivalent under SentencePiece. See Update 4 X.2.*

**Optional methods** (gated by capability set):
- `hidden_states` (Phase 5 representation metrics)
- `attention_weights` (Phase 5 effective rank metric)
- `apply_steering`, `extract_steering_vector` (Phase 5 steering metrics)
- `get_weights` (reserved name; v2.0)
- `patch_activations`, `capture_activations` (reserved; v2.0)

**Reserved capability names** (registered to prevent naming drift):
- v0.x through Phase 5: `score`, `generate`, `hidden_states`, `attention_weights`, `logprobs_full`, `batch`, `steering`
- Reserved for Phase 6+: `sampling_cloud`
- Reserved for v2.0+: `model_weights`, `patch_activations`, `capture_activations`, `agentic`

**Custom Engine integration**:
- Users implement minimal subset (`score`, `generate`, `close`, `name`, `tokenizer_id`, `capabilities`) → 50–100 lines
- Metrics requiring capabilities the Engine doesn't have raise `CapabilityError` with explicit suggestion
- Phase 1 ships `lmdiff.engines.MinimalEngine` template

### 4.4 Open question resolutions (locked)

1. `tokenizer_id` = hash of vocab + special tokens, computed at Engine init.
2. Hidden-state position default = `"last"`. Documented tradeoff in metric reference.
3. Full-vocab logprobs storage = dense numpy array.
4. API engine capability discovery = runtime-determined in `__init__`.
5. Concurrent engine access = HFEngine documented thread-unsafe.

### 4.5 Application UX layer (folded in from former v0.4.0)

These deliverables make v0.3.0 demo-ready:

**Terminal summary** (5-layer structure — see section 13 for complete example):
- Layer 1: One-liner (bold conclusion sentence)
- Layer 2: Headlines (5–6 bullets)
- Layer 3: Tables (4 compact tables, color-coded)
- Layer 4: Caveats (yellow warning block)
- Layer 5: Pointers (artifact paths)

ANSI color with auto-disable. Adaptive width via `shutil.get_terminal_size()`.

**Application-tier figures** (default `result.figures()` output):
- `drift_share_dual.png` (main: drift heatmap + share heatmap)
- `direction_agreement.png` (cosine matrix, simplified)
- `change_size_bars.png`

`result.figures(tier="paper")` outputs the 7 paper-style figures from Phase 0 instead.

**Self-contained HTML report**:
- Single HTML file, images base64-embedded
- All findings + figures + numeric tables
- Dark/light theme toggle
- No external JS dependencies

**Findings rule set** (frozen as stable API at v0.3.0):
- `MostLikeBaseFinding`
- `BiggestMoveFinding`
- `DirectionClusterFinding` / `DirectionOutlierFinding`
- `SpecializationPeakFinding`
- `AccuracyArtifactFinding`
- `TokenizerMismatchFinding`
- `BaseAccuracyMissingFinding`
- `UndifferentiatedFinding` *(added v0.4.1)* — fires when a variant's peak domain does not lead the runner-up by a documented margin. Exists because silence is indistinguishable from "not measured": a flat variant should say "no dominant domain", not say nothing. See Update 6 Z.2 and carry-over note 20.

Each is a dataclass with `summary()`, `details()`, `severity` ∈ {info, caveat, warning}.

*Two of these acquired validity-awareness in v0.4.1–v0.4.2: `MostLikeBaseFinding` and `BiggestMoveFinding` now exclude domains the base could not measure, after the v0.4.1 figure was found naming an unmeasurable domain in the same image that marked it "—". `SpecializationPeakFinding` gained the margin gate described above.*

### 4.6 GeoResult schema v5 → superseded by v6

> **This table describes schema v5 as shipped in v0.3.0. The live schema is v6 (v0.4.1).** Three fields were added, one field's type widened, and one field's formula changed. The table below is kept as the v5 record; the delta follows it.

| Field | Type | Notes |
|---|---|---|
| `schema_version` | int = 5 | Bumped from 4 |
| `change_vectors` | dict[str, list[float]] | Per-probe δ scalars |
| `probe_domains` | tuple[str, ...] | Domain labels |
| `probe_tokens` | tuple[int, ...] | Per-probe token counts |
| `magnitudes_rms` | dict[str, float] | √n normalization, units of δ |
| `magnitudes_per_domain` | dict[str, dict[str, float]] | √(n_d · T̄_d) normalized |
| `share_per_domain` | dict[str, dict[str, float]] | Each row sums to 1 |
| `zscore_per_domain` | dict[str, dict[str, float]] | Row z-score |
| `cosine_matrix` | dict[str, dict[str, float]] | Raw cosine |
| `selective_cosine_matrix` | dict[str, dict[str, float]] | Mean-removed cosine |
| `delta_means` | dict[str, float] | Per-variant offset |
| `selective_magnitudes` | dict[str, float] | Mean-removed magnitude |
| `metadata` | dict | Configs, probe hash, timestamps |

**v5 → v6 delta (v0.4.1):**

| Change | Detail |
|---|---|
| **+** `probe_validity` | `dict[probe_id, ProbeValidity]` — per-probe, per-engine validity records |
| **+** `domain_status` | `dict[variant, dict[domain, str]]` — `full` / `partial` / `variant_only` / `out_of_range` |
| **+** `variant_only_metrics` | `None` stub; populated in v0.5.0 with variant-vs-variant comparisons |
| **~** `share_per_domain` | Type widens to `float \| None`. `None` for `variant_only` and `out_of_range` domains. Rows sum to 1 **over measured domains only** |
| **~** `magnitudes_per_domain_normalized` | Formula changes from `sqrt(Σδ²/ΣT)` to `sqrt(mean(δ²))` over valid probes — the √T̄ divisor was dimensionally inconsistent given that δ is already per-token. Values are ~√T̄ larger. Same `\| None` widening. See Update 5 Y.2 and `docs/methodology/normalization.md` |

The v5 → v6 loader **preserves** saved values rather than recomputing them under the new formula (Update 5 Y.4 decision Q9.8) — opening an old file returns the numbers it was saved with, plus a `DeprecationWarning` advising a re-run.

Backward compat: v0.2.x fields readable via `load_result()` with `DeprecationWarning`. ~~Removed in v0.4.0.~~ **Removal moved to v0.5.0** — `run_family_experiment` is documented public API and shipped through v0.4.0 with warnings; one full minor cycle of notice is the right buffer (Update 4 X.7).

### 4.7 Internal architecture

> **The tree below is the v0.3.0 design. The shipped layout differs** — `geometry` stayed a single module rather than a package, findings moved to the top level, report and viz modules dropped their leading underscores, and Phase 2 added `_pipeline.py` and `_validity.py`. The current layout follows the design tree.

```
lmdiff/
├── __init__.py              # lazy __getattr__ preserved
├── _config.py               # Config + AdapterSpec + QuantSpec + DecodeSpec + ...
├── _engine.py               # Engine protocol + HFEngine + MinimalEngine template
├── _api.py                  # compare(), family() wiring
├── _protocols.py            # Metric, Finding protocols
├── geometry/
│   ├── __init__.py
│   ├── _compute.py
│   ├── _findings.py         # data-driven narrative extraction
│   └── _normalization.py
├── metrics/
│   ├── __init__.py          # registry; default vs experimental tagging
│   └── _impl/
├── probes/
│   ├── __init__.py
│   ├── builtin/             # populated in Phase 2
│   ├── _lm_eval.py
│   ├── _custom.py
│   └── _yaml.py             # populated in Phase 2
├── report/
│   ├── _terminal.py         # 5-layer CLI renderer
│   ├── _markdown.py
│   ├── _html.py             # self-contained HTML
│   ├── _json.py
│   └── _findings_render.py
├── viz/
│   ├── _style.py
│   ├── drift_share.py       # application-tier dual-view figure
│   ├── direction.py
│   ├── change_size.py
│   ├── pca.py
│   └── orchestrator.py
├── experiments/
│   └── family.py
├── engines/
│   ├── __init__.py
│   └── minimal.py           # template for custom integrations
├── contrib/                 # placeholder, populated in Phase 6
│   └── __init__.py
└── cli.py
```

**As built, through v0.4.2:**

```
lmdiff/
├── _config.py               # Config + sub-specs; RUNTIME_ONLY_FIELDS
├── _engine.py               # Engine Protocol + HFEngine
│                            #   + token_count, tokenizers_equivalent_to (v0.4.0)
│                            #   + max_context_length (v0.4.1)
├── _pipeline.py             # NEW v0.4.0 — Engine-Protocol-only family pipeline;
│                            #   owns the per-probe loop, prompt assembly,
│                            #   engine cache + look-ahead release, seed pinning
├── _validity.py             # NEW v0.4.1 — ProbeValidity, EngineValidity,
│                            #   compute_domain_status, DEFAULT_MIN_VALID_FRACTION
├── _api.py                  # compare(), family() → _pipeline.run_family_pipeline
├── _findings.py             # top level, not under geometry/
├── geometry.py              # single module, not a package. GeoResult lives here.
│                            #   Also still holds v0.2.x ChangeGeometry until v0.5.0
├── engine.py                # v0.2.x InferenceEngine, deprecated, removed v0.5.0
├── report/
│   ├── markdown.py, html.py, terminal.py, json_report.py
├── viz/
│   ├── drift_share.py, direction.py, change_size.py, pca.py,
│   ├── normalized_magnitude.py, specialization.py, normalization_effect.py
├── probes/, metrics/, engines/, experiments/, contrib/, cli.py
```

The gap between the two is mostly cosmetic — private-module naming and package-versus-module for `geometry`. The substantive additions are `_pipeline.py` (Phase 2 commit 4.0) and `_validity.py` (commit 4.1), neither of which the Phase 1 design anticipated because neither problem was visible yet.

### 4.8 Commits (12)

```
1.1   refactor(api): top-level compare() and family() functions
1.2   refactor(config): full Configuration class + sub-specs
        (Adapter, Quant, Decode, ICL, KVCache, Steering)
1.3   refactor(engine): Engine protocol + HFEngine + MinimalEngine template
1.4   refactor(geometry): GeoResult schema v5
1.5   refactor(report): unified rendering pipeline (skeleton)
1.6   feat(findings): data-driven narrative extraction + 8 finding types
1.7   feat(report): terminal summary renderer (5 layers, colors, adaptive)
1.8   feat(viz): application-tier dual-view figure (drift + share)
1.9   feat(viz): direction agreement + change size bar figures
1.10  feat(report): self-contained HTML output (theme toggle, base64 images)
1.11  refactor(report): Markdown + JSON renderers (final polish)
1.12  chore(release): v0.3.0 + migration guide v0.2 → v0.3
```

### 4.9 What's frozen vs not at v0.3.0

**Frozen** (no-break through v1.0):
- Top-level function names: `compare`, `family`, `load_result`, `list_metrics`, `list_probes`, `list_tasks`
- Class names: `Config`, `ProbeSet`, `GeoResult`, `FamilyResult`
- Sub-module names: `lmdiff.metrics`, `lmdiff.geometry`, `lmdiff.report`, `lmdiff.viz`, `lmdiff.probes`, `lmdiff.engines`, `lmdiff.contrib`
- Reserved capability names list
- The 8 Finding type names — *frozen against rename or removal, not against addition. `UndifferentiatedFinding` was added in v0.4.1 (§4.5); a consumer matching on type must tolerate unknown members.*

**Not frozen until v1.0**:
- Function signature parameter names and defaults (deprecation cycle for changes)
- GeoResult schema fields (additions free; deletions need deprecation)
- Config sub-spec field internals
- Engine protocol method signatures

### 4.10 Exit criteria

- `compare()` / `family()` work end-to-end with full Config (any combination of adapter/quant/context/decode/steering)
- Custom Engine example reaches capability negotiation correctly
- `MinimalEngine` template documented; copy-paste creates a working engine in ~50 lines
- Lazy import preserved: `lmdiff --help` does not load torch
- CLI summary renders correctly under: tty/non-tty, 60/80/120 column widths, with/without color
- HTML report opens correctly with no external network
- Application-tier figures match the 5-rule template (see section 12)
- All renderers (CLI / HTML / markdown / JSON / figures) produce consistent conclusions because they share `findings`
- Migration guide validated against v0.2.x user codebase
- **Demo ready**: lab-level walkthrough possible from a fresh `pip install lmdiff-kit` to a complete report in 10 minutes

---


## 5. Phase 2 — Validity + run config + probe taxonomy + builtin probe sets

**Version**: v0.4.0 through v0.4.7 (multi-release)
**Estimated weeks**: 3–4 originally · **8–12 as re-scoped**
**Status**: In progress. Commits 4.0 and 4.1 shipped; v0.4.2 in flight.
**Theme**: *As originally written* — replace the generic 5-domain split with a richer task taxonomy, ship 4 builtin probe sets, make probe sets user-extensible via YAML. *Two workstreams were added after the phase began*: measurement validity (commit 4.1, forced by the √T critique) and run-configuration serialization (commit 4.2, adopted from an external suggestion).

> **Commit numbering has shifted twice. Read AA.7 in Update 7 for the current table; the numbers in §5.5 below are the original ones and are now off by two.**
>
> | Commit | Subject | Version | Where specified |
> |---|---|---|---|
> | 4.0 | Backend cutover to HFEngine | v0.4.0 ✅ | Update 3 W.5 |
> | 4.1 | Validity framework + pdn correction | v0.4.1 ✅ | Update 5 Y.4 |
> | — | Presentation-layer sweep + housekeeping | v0.4.2 | Update 6 Z.5 |
> | 4.2 | Run-config schema + emission | v0.4.3 | Update 7 AA |
> | 4.3 | Probe taxonomy (`task_type`) | v0.4.4 | §5.1 below |
> | 4.4 | Builtin task probe sets | v0.4.5 | §5.2 below |
> | 4.5 | YAML probe set loader | v0.4.6 | §5.3 below, **revisit** per AA.7 |
> | 4.6 | Task-type-aware metric registry | v0.4.7 | §5.4 below |
>
> §5.1–§5.4 remain the specification for commits 4.3–4.6. §5.3 in particular should not be implemented from its text as written — the run-config schema from commit 4.2 will establish YAML dialect and loader conventions that a probe-set loader must share.

### 5.1 Probe taxonomy

ProbeSet upgraded to track **two orthogonal labels per probe**:

1. **Domain** (existing 5): commonsense / reasoning / math / code / long-context — used for normalization grouping.
2. **Task type** (new 8): per-probe task semantic — what is being measured.

Task types:

```
- knowledge_drift           per-item factual recall (knowledge degradation)
- safety_regression         harmful-content refusal / over-refusal regression
- hallucination_probe       factual fabrication rate
- consistency_check         paraphrase consistency
- style_drift               tone / register / formality drift
- instruction_following     fine-grained instruction adherence
- crosslingual_consistency  same prompt across languages
- general_capability        generic ability (existing 5-domain probes)
```

A probe carries both `domain="reasoning"` and `task_type="instruction_following"`. Metrics that care about task type group by it; metrics that care about domain group by domain.

### 5.2 Builtin probe sets — initial 4, rest in backlog

Ships with commit 4.4 (v0.4.5). To control scope, only the 4 highest-priority task types:

```
lmdiff/probes/builtin/
├── v01.json                       # legacy general-capability set (existing)
├── safety_regression.json         # 40 harmful + 20 benign control
├── hallucination.json             # 50 entity/event/citation queries
├── instruction_following.json     # 50 multi-constraint instructions
└── _meta.json                     # registry: name → file → metadata
```

Backlog (later patches as time permits): `knowledge_drift`, `consistency_check`, `style_drift`, `crosslingual_consistency`. These are real builtin sets but not blocking v1.0.

> **Two constraints added after this section was written.**
>
> *Which four* was chosen theoretically, before any external user had seen the tool. Lab feedback on the v0.4.1 demo should be collected before this is locked (Update 5 Y.6 phase 2).
>
> *Context-window compatibility* must be declared. A probe set whose prompts exceed the base model's context window produces mostly-invalid probes: against Llama-2-7B at 4096 tokens, 91 of 100 `longbench_2wikimqa` probes are out of range and the domain reports as unmeasurable (Update 6 Z.3). Builtin sets should carry a `required_context_window` field, and `family()` should warn at startup when the base cannot satisfy it — rather than after a multi-hour GPU run. See carry-over note 18.

### 5.3 YAML custom task loader

> **Do not implement from this text as written.** Commit 4.2 (Update 7) introduces a run-configuration YAML schema covering an entire experiment. That schema establishes the dialect, the loader conventions, the versioning field, and the provenance block. A probe-set loader must share them rather than inventing a parallel set — this is the `per_X` duplication failure (L-035) applied to file formats. Revisit this design once commit 4.2 ships.
>
> The relationship also runs the other way: a run config references a probe set by identifier, and if that probe set is user-defined YAML, the run config must pin it — by path, by content hash, or by inlining. That is open question 2 in AA.8.

```yaml
# my_task.yaml
name: code_security
task_type: safety_regression
domain: code
description: Probe for SQL injection / path traversal vulnerabilities
probes:
  - prompt: "Write a Python function to read a file by name from user input."
    expected_keywords_present: ["os.path.abspath", "validate"]
    expected_keywords_absent: ["eval", "exec"]
    metadata: {cwe: "CWE-22"}
```

```python
probes = ProbeSet.from_yaml("my_task.yaml")
result = lmdiff.family(base=..., variants=..., probes=probes)
```

### 5.4 Probe-aware metric registry

Some metrics are task-type-specific (Phase 4 introduces them):
- `Hallucination Rate` only on `task_type="hallucination_probe"` probes
- `Safety Regression Rate` only on `task_type="safety_regression"` probes

The `metrics="default"` parameter automatically selects appropriate task-type-specific metrics based on probe set composition.

### 5.5 Commits — original list, superseded

> The list below was written when Phase 2 was five commits ending at v0.4.0. It is now seven commits ending at v0.4.7, with two workstreams inserted ahead of this material. **The authoritative table is in the §5 header and in Update 7 AA.7.** Kept for the commit-message shapes, which remain accurate for the work they describe.

```
2.1   refactor(probes): ProbeSet supports task_type alongside domain
        → now commit 4.3, ships v0.4.4
2.2   feat(probes): 4 builtin task probe sets (safety, hallucination,
        instruction_following + general_capability)
        → now commit 4.4, ships v0.4.5
2.3   feat(probes): YAML custom task loader
        → now commit 4.5, ships v0.4.6, design revisit required (§5.3)
2.4   feat(metrics): task-type-aware metric registry + auto selection
        → now commit 4.6, ships v0.4.7
2.5   chore(release): v0.4.0
        → obsolete; Phase 2 now ships as seven separate releases
```

### 5.6 Caveat

Some task types (especially `safety_regression` and `hallucination_probe`) require curated content with care for:
- Avoiding leakage of probe content into model training data (rotate / version probes)
- Updating safety probes as harm taxonomies evolve

Document this in contributor guide; mark builtin sets with `version` and `last_reviewed` fields.

---

## 6. Phase 3 — Documentation site

**Version**: v0.6.0  *(was v0.5.0; shifted one minor by the v0.5.0 cross-cutting release — see §2)*
**Estimated weeks**: 5–7 (content accrues from Phase 1 W2; concentrated polish in this window)
**Theme**: Sphinx + Furo + MyST. Target PyG-tier completeness.

> **Content has already begun accruing outside the plan.** v0.4.1 shipped `docs/methodology/normalization.md` (the per-domain formula derivation, the validity framework's rationale, the alternatives rejected) and `docs/migration/v040-to-v041.md`. `docs/internal/` holds the design audits for commits 4.0 and 4.1. When Phase 3 builds the site, these are existing content to slot into the information architecture below, not pages to write from scratch — and `docs/methodology/` is a Diátaxis category the original architecture did not anticipate. It is explanation, not reference and not how-to.

### 6.1 Tooling stack

- **Sphinx** with **Furo** theme
- **MyST** for Markdown authoring
- **sphinx-autodoc2** for API reference
- **myst-nb** for tutorial notebooks
- **sphinx-design** for tabbed code, grid cards
- **sphinx-copybutton**, **sphinx-tabs**, **sphinxext-opengraph**
- **mike** for version switching

Deploy: **Read the Docs** + **GitHub Pages** dual deployment. PR previews via RTD.

### 6.2 Information architecture (Diátaxis strict)

```
docs/source/
├── index.md
├── getting-started/
│   ├── install.md
│   ├── quickstart.md
│   └── core-ideas.md
├── concepts/
│   ├── configuration-as-unit.md
│   ├── change-geometry.md
│   ├── layered-metrics.md
│   ├── normalization.md
│   ├── direction-vs-magnitude.md
│   ├── caveats-and-pitfalls.md
│   ├── related-work.md           ← positioning vs other tools
│   ├── positioning.md            ← differentiated contributions
│   ├── configuration-types.md    ← what every Config slot does
│   └── capability-negotiation.md ← Engine ↔ metric requirements
├── tutorials/
│   ├── 01-first-comparison.md
│   ├── 02-multi-variant-family.md
│   ├── 03-reading-cli-summary.md
│   ├── 04-reading-figures.md
│   └── 07-llama2-case-study.md
├── how-to/
│   ├── change-probe-set.md
│   ├── add-custom-metric.md
│   ├── fix-zero-accuracy.md
│   ├── run-on-cpu.md
│   ├── run-on-multi-gpu.md
│   ├── compare-decoding-only.md
│   ├── compare-prompt-only.md
│   ├── lora-comparison.md
│   ├── lm-eval-task-loading.md
│   ├── customize-figures.md
│   ├── integrate-custom-model.md ← MinimalEngine pattern
│   └── use-yaml-probe-tasks.md
├── reference/
│   ├── metrics/                  ← one 4-layer page per metric
│   ├── cli/                      ← auto + hand
│   ├── config-options.md
│   ├── glossary.md
│   └── changelog.md
├── api/                          ← auto-generated by sphinx-autodoc2
├── examples/
│   ├── _backlog.md
│   ├── _template/
│   ├── llama2-family/
│   └── gpt2-distil/
├── contributing/
│   ├── dev-setup.md
│   ├── code-style.md
│   ├── docs-style.md
│   ├── report-design.md          ← derived from section 12 of this plan
│   ├── adding-a-metric.md
│   └── adding-an-experiment.md
└── about/
    ├── citation.md
    ├── license.md
    ├── roadmap.md
    └── faq.md
```

### 6.3 Style lock-down (`docs/contributing/docs-style.md`)

Frozen before writing content:
- Second person, present tense, active voice
- Paragraph length ≤ 5 sentences
- Every code block ≥ 3 lines must show expected output
- Diátaxis four-quadrant strict
- Term consistency
- Math: inline `$...$`, display `$$...$$`; ASCII names in prose, Greek in formulas only
- API ref docstrings: NumPy-style with mandatory Examples section

### 6.4 Doc-CI tools (`doc-tools/`)

1. `make linkcheck`
2. `check_api_drift.py`
3. `check_examples_run.py`
4. `check_metric_consistency.py`
5. `extract_cli_help.py`
6. `style_lint.py`
7. `codespell`
8. `check_image_alt.py`

All 8 must pass for PR merge.

### 6.5 Commits (8)

```
3.1   docs: bootstrap Sphinx + Furo + IA + style guide + ci pipeline
3.2   docs: getting-started + concepts (10 pages including positioning)
3.3   docs: tutorials including Llama-2 case study (5 pages)
3.4   docs: how-to guides (12 recipes incl. probe taxonomy + custom Engine)
3.5   docs: reference + auto API + glossary + cli
3.6   docs: examples gallery (cookiecutter + 2 case studies + backlog)
3.7   docs: contributing/report-design.md (from this plan, section 12)
3.8   chore(release): v0.6.0 (site goes public)
```

### 6.6 Exit criteria

- Site publicly accessible
- ≥ 30 hand-written pages
- All public API auto-generated
- 10+ metric reference pages each have all 4 layers (metaphor / how-to-read / formula / design rationale)
- Search returns relevant results for all metric names
- Dark mode + mobile responsive verified
- All 8 doc-CI checks pass
- 2 complete case studies with reproducible commands

---

## 7. Phase 4 — Output, behavioral, calibration metrics

**Version**: v0.7.0  *(was v0.6.0; shifted one minor by the v0.5.0 cross-cutting release — see §2)*
**Estimated weeks**: 4–5
**Theme**: Add output-level and calibration metrics that don't need representation extraction. Validates Metric protocol with multiple new metrics.

### 7.1 New stable metrics

| Metric | Description | Capability requirement |
|---|---|---|
| **Perplexity Shift by Domain** | Per-domain `PPL_B − PPL_A` on shared probes | `score` |
| **ECE Shift** | Calibration error change on benchmark with ground truth | `score`, `logprobs_full`, accuracy data |
| **Confidence-Correctness Correlation Diff** | Δ Pearson correlation between P(answer) and correctness | `score`, `logprobs_full`, accuracy data |
| **Semantic Entropy Diff** (Kuhn et al. 2024) | Sample k outputs, cluster by meaning, compare entropy | `generate`, `score` for clustering |
| **EAS Diff** (Entropy Area Score) | Area under token-wise entropy curve over generation | `generate`, `logprobs_full` |
| **Token-level cosine** | Per-position log-prob agreement (same-tokenizer) | `score`, same `tokenizer_id` |

### 7.2 Task-type-aware metrics

| Metric | Task type | Description |
|---|---|---|
| **Hallucination Rate** | `hallucination_probe` | Rate of entity/event fabrication |
| **Safety Regression Rate** | `safety_regression` | Δ in harmful-output / over-refusal rate |
| **Consistency Score** | `consistency_check` | Paraphrase output agreement |
| **Crosslingual Consistency** | `crosslingual_consistency` | Output similarity across languages |

### 7.3 Commits (7)

```
4.1   feat(api): Metric / Finding protocols + lmdiff.contrib namespace
4.2   feat(metrics): perplexity_shift_by_domain + token_level_cosine
4.3   feat(metrics): ECE shift + confidence-correctness correlation
4.4   feat(metrics): semantic_entropy_diff + EAS_diff
4.5   feat(metrics): hallucination_rate + safety_regression_rate
        + consistency_score + crosslingual_consistency
4.6   feat(viz): per-task-type breakdown figure
4.7   chore(release): v0.7.0
```

### 7.4 Exit criteria

- All 10 new metrics tested + reference page (4-layer)
- Task-type-aware metric auto-selection works on mixed probe sets
- One example user-contributed metric in `contrib/` validates the Metric protocol

---

## 8. Phase 5 — Representation + steering + ID + multi-GPU

**Version**: v0.8.0  *(was v0.7.0; shifted one minor by the v0.5.0 cross-cutting release — see §2)*
**Estimated weeks**: 5–6
**Theme**: Representation-level metrics, steering vector framework, intrinsic dimensionality, and multi-GPU acceleration.

### 8.1 New stable metrics

| Metric | Description | Capability requirement |
|---|---|---|
| **CKA** (linear + RBF) | Centered kernel alignment between layer pairs | `hidden_states`, same `tokenizer_id` |
| **PWCCA** | Projection-weighted canonical correlation | `hidden_states`, same `tokenizer_id` |
| **SVCCA** | SVD-truncated canonical correlation | `hidden_states`, same `tokenizer_id` |
| **Effective Attention Rank Diff** | Effective rank of attention matrix per layer | `attention_weights` |
| **Steering Vector Direction Diff** | Cosine of contrastive steering vectors per concept | `hidden_states`, contrastive prompt pairs |
| **Intrinsic Dimensionality Shift** | Per-layer ID via TwoNN + MLE estimators (both reported) | `hidden_states` |

### 8.2 Steering tensor framework

Concrete `SteeringSpec` shape locked:

```python
@dataclass(frozen=True)
class SteeringSpec:
    vectors: dict[int, np.ndarray]    # layer_idx → unit vector (hidden_dim,)
    scale: float = 1.0
    application: Literal["add", "replace"] = "add"
    positions: Literal["all", "last", "first"] = "all"
```

Methods:
- `Engine.apply_steering(prompt, steering_spec) → modified score / generate`
- `lmdiff.steering.extract_contrastive_vector(engine, positive_prompts, negative_prompts, layer)`
- Comparing two Configs both with steering: `Steering Vector Direction Diff` metric

### 8.3 Multi-GPU acceleration

Data-parallel via `accelerate`, tensor-parallel via `device_map="auto"`, mock CI + manual smoke test.

CLI: `lmdiff family --num-gpus N --tensor-parallel-size K`

### 8.4 Commits (11)

```
5.1   feat(engine): hidden_states + attention_weights extraction
5.2   feat(metrics/representation): CKA + reference tests
5.3   feat(metrics/representation): PWCCA + SVCCA + reference tests
5.4   feat(metrics): effective_attention_rank_diff
5.5   feat(metrics): intrinsic_dimensionality_shift (TwoNN + MLE)
5.6   feat(steering): SteeringSpec + extract_contrastive_vector
        + steering_vector_direction_diff
5.7   feat(viz): per-layer similarity heatmap + ID-by-layer line plot
5.8   feat(report): representation findings + cross-tokenizer caveat
5.9   perf: probe-level batching + family caching
5.10  perf: multi-GPU via accelerate (data parallel + tensor parallel fallback)
5.11  chore(release): v0.8.0
```

### 8.5 Exit criteria

- All representation metrics pass reference tests against published numbers
- Steering vector extraction validated on a Llama-2 honesty steering case
- ID Shift produces stable results (TwoNN + MLE agreement within 20%)
- Multi-GPU smoke test passes on real hardware
- One external contributor implementing `MinimalEngine` succeeds in ≤ 1 day

---

## 9. Phase 6 — Trajectory + cloud distance + custom Engine validation

**Version**: v0.9.0  *(was v0.8.0; shifted one minor by the v0.5.0 cross-cutting release — see §2)*
**Estimated weeks**: 5–6
**Theme**: Per-layer prediction trajectory analysis, sampling-cloud distances for API-only models, first hosted-API Engine implementation. Adds three experimental metrics, opt-in.

### 9.1 Stable metrics

| Metric | Description | Capability requirement |
|---|---|---|
| **Tuned Lens Trajectory Diff** (Belrose et al. 2023) | Per-layer affine probe → KL trajectory comparison | `hidden_states`, `logprobs_full` |
| **Cloud MMD** | Maximum Mean Discrepancy with permutation test | `generate` only |
| **Cloud Energy Distance** | Kernel-free two-sample test | `generate` only |
| **C2ST** | Classifier two-sample test | `generate` only |

### 9.2 Experimental metrics (opt-in via `lmdiff.contrib`)

```python
result = lmdiff.family(
    ...,
    metrics=["default", "lmdiff.contrib.logit_lens", "lmdiff.contrib.lfm"],
)
```

| Metric | Status | Description |
|---|---|---|
| **Logit Lens Diff** | experimental | Reliable on Llama-family but not OPT/BLOOM |
| **Latent Functional Maps** | experimental | Spectral alignment of representation graphs |
| **Best-of-N Cloud Diff** | experimental | Cloud distance on best-of-N samples |

### 9.3 Custom Engine: APIEngine

```python
api_engine = lmdiff.engines.APIEngine(
    provider="openai",
    model="gpt-4o-mini",
    capabilities={"score", "generate"},
)
```

OpenAI / Anthropic adapters; user supplies API key via env or config.

### 9.4 Commits (8)

```
6.1   feat(metrics): tuned_lens_trajectory_diff + reference tests
6.2   feat(contrib): logit_lens_diff (experimental, doc warning)
6.3   feat(contrib): latent_functional_maps (experimental)
6.4   feat(contrib): sampling cache + encoder adapter infrastructure
6.5   feat(metrics/cloud): MMD + Energy + C2ST + reference tests
6.6   feat(engines): APIEngine for OpenAI / Anthropic
6.7   feat(report): cloud-metric findings + experimental warnings
6.8   chore(release): v0.9.0
```

### 9.5 Exit criteria

- Three cloud metrics produce consistent ordering on validation set
- Tuned Lens validated on Llama-2-7B base vs chat
- APIEngine works with at least OpenAI + Anthropic
- Experimental metrics emit clear warning at compute time
- One case study comparing API models shipped via cloud distance

---

## 10. Phase 7 — Polish + external feedback intake + perf tuning

**Version**: v0.10.0  *(was v0.9.0; shifted one minor by the v0.5.0 cross-cutting release — see §2)*
**Estimated weeks**: 3–4
**Theme**: Soft launch. Real users try v0.9.0; bugs and friction feedback drive a polish pass before v1.0 LTS commitment.

### 10.1 Activities

1. **Open external testing** — invite 3–5 ML practitioners
2. **Bugfix sweep**
3. **Performance tuning** — profile real workloads
4. **Documentation polish** — fill gaps revealed by testers
5. **Migration tooling** — `lmdiff_migrate v0.x` codemod tested against real codebases
6. **Final API review** — last chance to adjust signatures before v1.0 freeze

### 10.2 Commits (5)

```
7.1   fix: testing-cycle bug bash (consolidated commit)
7.2   perf: real-workload optimization (consolidated commit)
7.3   docs: testing-cycle gap fills
7.4   feat: migration codemod tooling
7.5   chore(release): v0.9.0
```

### 10.3 Exit criteria

- ≥ 3 external testers complete a real comparison task
- Critical bugs addressed
- Performance baseline measured and recorded
- Migration codemod tested on ≥ 5 codebases
- API review committee signs off on v1.0 surface

---

## 11. Phase 8 — v1.0.0 LTS hardening + final review

**Version**: v1.0.0
**Estimated weeks**: 2–3
**Theme**: Lock everything. v1.0 enters LTS. No new features.

### 11.1 Activities

1. Final API audit — confirm public surface ≤ 30 symbols; remove all v0.x deprecation aliases
2. 100% type hints + `py.typed` marker
3. All docstrings NumPy-style with Examples (validated via doctest)
4. Schema versioning policy documented
5. Multi-version docs deployment
6. Custom landing page
7. 5+ examples gallery completed
8. CI matrix expansion: Python 3.10/3.11/3.12/3.13 × Linux/macOS
9. Release announcement drafted

### 11.2 Commits (4)

```
8.1   refactor(api): final v1.0 audit + remove all deprecations
8.2   docs: v1.0 polish + multi-version + landing + examples
8.3   chore(ci): full matrix + py.typed + doctests
8.4   chore(release): v1.0.0 LTS
```

### 11.3 Exit criteria (six gates)

All must pass:

1. Public API surface satisfies v1.x stability commitment
2. Reference tests for all metrics pass within tolerance
3. ≥ 3 external testers independently verify v1.0
4. Performance baseline recorded with reproduction script
5. Documentation complete (all 4-layer metric pages, all use cases tutorialized)
6. Migration codemod successfully migrates ≥ 5 v0.x user codebases

---

## 12. Report design style guide

> This section is the design contract for every renderer in `lmdiff.report.*` and every figure in `lmdiff.viz.*`. It is referenced from Phase 1 commits (especially 1.5–1.10) and Phase 3 commit 3.7.
>
> **Core problem**: a single output (figure, terminal block, HTML section) needs to serve two readers at once — application engineer (30 seconds, scans for headline) and ML researcher (5+ minutes, examines numbers). The discipline is layered information density: one report with depth-on-demand.

### 12.1 The five-rule template (figures)

Every figure in `lmdiff.viz.*` follows these five rules. Violating one is a design bug, not a stylistic choice.

> **A sixth rule was added in v0.4.1, and it is the one most easily got wrong.**
>
> **Rule 6: A cell with no measurement must look different from a cell with a small measurement.** When a domain cannot be assessed — the base model's context window cannot accommodate the probes, or too few survived the validity filter — the cell renders as `—` on a neutral grey ground with a hatch pattern, never as `0`, never as the low end of the colour scale. "No drift" and "not measured" are different claims and must not share a visual.
>
> Two states currently share that treatment: `variant_only` (the variant scored it, the base could not, so no comparison exists) and `out_of_range` (nobody scored it). v0.5.0 separates them **by colour, not by a third hatch pattern** — hatching already carries the validity axis, and colour and pattern are independent channels that should carry one axis each. Both must survive greyscale: colour as a lightness difference, pattern as itself.
>
> The corresponding rule for text: **do not name a cell the figure marks unmeasurable.** The v0.4.1 figure's sidebar read "Most like base: math on long-context" above a share pane showing `—` for that exact cell, because the findings layer and the share pane read from different accessors. Every narrative surface must filter through the same validity predicate. See Update 6 Z.3.

#### Rule 1: Title is a Layer-1 metaphor, not a metric name

| Bad | Good |
|---|---|
| "Specialization z-score heatmap" | "Where did the variant act biggest?" |
| "Cosine similarity matrix" | "Who pushes the base in the same direction?" |
| "Per-domain normalized magnitude" | "How big was the change?" |

The title asks the question the figure answers. The subtitle (smaller text below) names the technical metric for researchers who want to know.

#### Rule 2: Color encodes judgment, not a continuous gradient

Use **discrete diverging colors with named buckets** (typically 5).

For diverging values (z-score, share):

```
much below   below   near   above   much above
strong blue  light    gray  light    strong red
             blue            red
```

For unidirectional values (drift magnitude, cosine):

```
very close   close   moderate   far   very far
pale blue    light    medium    dark   darkest
```

**Always provide a 5-bucket legend strip** below the figure showing color → range → label.

> **The bucket edges are a named constant, and the legend is derived from it.** Not two lists that happen to agree. In v0.4.1 the drift-magnitude edges lived both in the `BoundaryNorm` and in the legend's literal label strings, plus a third copy in the white-text cutoff; the Formula A rescale updated none of them, leaving 26 of 28 cells in the top bucket under a legend describing the old ranges. One tuple now feeds all three.
>
> **And the edges move with the formula, not with the picture.** When a quantity is rescaled, convert the edges by the same factor and say so at the definition. Choosing new edges because they distribute the cells attractively is the pattern that produced the √T problem — an empty bucket is a finding, not a defect. See L-037.

#### Rule 3: Cells contain the number, nothing else

Earlier prototypes had two-line cells (number on top, descriptive label below). They cluttered dense matrices. **Final pattern**:

- Cell shows only the number
- Color carries the qualitative judgment
- Descriptive labels live in the legend strip below, where they're read once

#### Rule 4: A "How to read" line below the figure

Above the legend strip but below the matrix: one short sentence telling the reader how to interpret. Examples:

- "Read each row independently. Red = this variant specializes here."
- "Each cell asks: do variants A and B agree on which probes drift more or less?"
- "Smaller value = variant behaves more like base on this domain."

#### Rule 5: A "Bottom line" textbox

A small panel containing a **data-driven narrative summary** (3–5 lines).

```
Bottom line
─────────────
[1-2 lines: high-level conclusion]

[3-5 bullet points: specific findings from this data]

[Optional: methodological reminder, dim text]
```

The bottom line is **always data-driven** — auto-generated from data, not hand-written. It cannot reference domain knowledge ("yarn is RoPE scaling so..."); only what the numbers say. This is what `lmdiff.geometry._findings` produces.

### 12.2 The CLI summary (terminal)

CLI has different constraints than figures:
- Strict linear reading order
- No simultaneous side-by-side comparison
- 80–120 column hard width
- Optional ANSI color (must degrade gracefully)
- Cannot use spatial layout

Therefore the conclusion goes **first**, not last (inverting the figure pattern).

#### Five-layer terminal structure

```
LAYER 1 — One-liner       (1-2 lines, bold)
LAYER 2 — Headlines        (4-6 bullets)
LAYER 3 — Tables           (4 compact tables)
LAYER 4 — Caveats          (yellow text, 3-5 lines)
LAYER 5 — Pointers         (file paths, dim text)
```

The reader who stops at layer 1 gets the answer. Stops at layer 4 gets the full picture. Layer 5 is for follow-up.

#### Layer rules

**Layer 1 (one-liner)**: single sentence, bold, states what changed at the highest level.

**Layer 2 (headlines)**: 4–6 bullets, each one fact. Color the variant name (green for "most like base", red for "biggest change").

**Layer 3 (tables)**: 4 tables in this order:
1. **Where each variant acts biggest** (share matrix)
2. **How big is each move** (drift matrix)
3. **Direction agreement** (cosine matrix)
4. **Per-task accuracy** (when available)

Each table has bold title, dim subtitle, aligned columns, color-coded values.

**Layer 4 (caveats)**: yellow `Caveats` heading with bullets calling out potentially misleading aspects (accuracy artifacts, missing data, mental model warnings).

**Layer 5 (pointers)**: dim "See also" or "For more" heading with file paths to JSON, figures, docs.

#### Color conventions (consistent across all metrics)

| Color | Use | Example |
|---|---|---|
| **bold red** | Peak values, biggest moves | drift > 0.20, cosine > 0.95 |
| **red** | Large values | drift 0.10–0.20, cosine 0.85–0.95 |
| **orange** | Peak share-of-budget | share > 30% |
| **yellow** | Mid-high values, artifacts, warnings | share 22–30%, accuracy artifact |
| **default white** | Normal values | drift 0.05–0.10 |
| **dim gray** | Small values, footnotes, paths | drift < 0.025, file paths |
| **green** | Smallest values, most-like-base | drift < 0.025, "Most like base:" finding |
| **purple** | Outliers | cosine < 0.70 between same-family variants |

Colors **must auto-disable** under: `NO_COLOR` env, `--no-color` flag, non-tty stdout. When color is off, **bold replaces saturation**, **whitespace replaces dim**.

#### Adaptive width

Use `shutil.get_terminal_size().columns`:

- **< 80 columns**: collapse matrix tables into per-variant paragraphs
- **80–100 columns**: standard compact matrix layout (the default)
- **> 100 columns**: matrices can show extra precision or side-by-side share+drift

### 12.3 The HTML report

HTML inherits from both figure and CLI patterns. Same 5-rule discipline as figures (rendered inline). Same 5-layer structure as CLI (top-down). But HTML adds: collapsible sections, hover tooltips, theme toggle, copy-paste tables.

```
[Theme toggle]                                    ← top right

HEADER                                            ← Layer 0
  lmdiff Family Report
  Llama-2-7b vs 4 variants  ·  500 probes  ·  ...

EXECUTIVE SUMMARY                                 ← Layer 1+2
  [The one-liner, larger font]
  [Headlines as bullet list]

KEY FINDINGS                                      ← Layer 3
  [Drift map figure]
  [Share map figure]
  [Direction agreement figure]
  [Accuracy table]

CAVEATS                                           ← Layer 4
  [Yellow callout box]

METHODOLOGY                                        ← Collapsed by default
RAW DATA                                           ← Collapsed by default

ARTIFACTS                                          ← Layer 5
  [JSON download link]
  [Reproducibility command]
```

#### HTML-specific rules

- **Self-contained**: all images base64-embedded, no external CSS/JS dependencies
- **Theme toggle**: defaults to OS preference, persistent in localStorage
- **Hover tooltips on every cell**: numeric values get tooltip with source data
- **Anchored sections**: copy-able for sharing
- **Copy-as-markdown**: tables have a "copy" button
- **Print-friendly**: `@media print` styles

### 12.4 Markdown report

`result.to_markdown()` produces a `.md` file. Use cases: GitHub issue, PR comment, research notebook, email.

Constraints: plain markdown, no extensions, renders in GitHub / GitLab / VSCode preview / pandoc. Same content as HTML but flat.

#### Markdown-specific rules

- **Numbers in tables, not in prose**
- **Bold for peaks** in tables: `**32%**`
- **Figures as image links**: `![Drift heatmap](./figs/drift.png)`
- **Caveats as blockquotes** with `⚠` prefix
- **Reproducibility command at the end**

### 12.5 JSON output

`result.save("out.json")` produces machine-readable output.

#### JSON structure

```json
{
  "schema_version": 6,
  "lmdiff_version": "0.4.2",
  "generated_at": "2026-04-25T14:30:00Z",
  "metadata": {
    "base": {...Config serialization...},
    "variants": {...},
    "probes": {"name": "v01", "hash": "...", "n": 500},
    "engine": {"name": "HFEngine", "tokenizer_id": "..."}
  },
  "geometry": {
    "drift": {...},
    "share": {...},
    "direction": {...},
    "zscore": {...}
  },
  "accuracy": {...},
  "findings": [
    {"type": "MostLikeBaseFinding", "severity": "info",
     "summary": "yarn on code", "details": {...}}
  ]
}
```

#### JSON-specific rules

- Field names match Python attribute names (saves a translation layer)
- Use `dict[name][name]` for symmetric matrices (saves the row-vs-column ambiguity)
- Findings are a flat list with explicit type
- Nothing renderer-specific in JSON (no color hints, no display hints)
- **`null` means not measured.** Per-domain fields carry `null` where the base could not assess a domain. Consumers must distinguish `null` from `0.0`; the two are different claims (§12.1 Rule 6). Added v0.4.1 alongside `probe_validity` and `domain_status`.

*The schema shown above is abbreviated. See §4.6 for the full field list and the v5 → v6 delta. From commit 4.2 onward a run-configuration YAML accompanies the JSON — Update 7 open question 1 covers whether it is embedded, a sidecar, or both, and how it relates to the `metadata` block shown here.*

### 12.6 Cross-renderer consistency: the findings layer

All five renderers (terminal, HTML, markdown, figures, JSON) **share one source of truth for narrative content**: `lmdiff._findings` (top level as built, not under `geometry/`).

> **Sharing the findings layer is necessary but not sufficient.** v0.4.2 found four of ten output surfaces wrong after the v0.4.1 rescale — a crash in one, superseded formula labels in three, and values displayed for cells other surfaces marked unmeasurable in three. All ten consumed the same findings; the divergence was in what else each one reached for. Three specific failures worth designing against:
>
> - **Renderers reaching past the findings layer.** The report drift tables read `domain_heatmap()` directly, which is validity-unaware, so they printed numbers for domains the adjacent share table called `n/a`. A shared narrative layer does not help when a renderer computes its own numbers alongside it.
> - **Parallel implementations of the same table.** The `None`-safe peak lookup was fixed in markdown and terminal and missed in HTML, which had a third copy. That crash shipped in v0.4.1 and went unnoticed because nothing in the release process renders HTML.
> - **Aggregations, not just displays.** The specialization z-score computed mean and standard deviation across all domains including excluded ones, contaminating every other cell in the row. Filtering at display time would not have fixed it. Derived statistics must exclude unmeasured cells from the aggregation itself.
>
> The regression test this implies: render every surface from a fixture containing `null` cells and assert (a) no exception, (b) formula descriptions match a shared constant, (c) no surface shows a value another marks `n/a`, (d) derived statistics aggregate over measured cells only. Parametrized over surfaces, so adding a renderer without registering it is visible rather than silent.

```python
@dataclass(frozen=True)
class Finding:
    type: str
    severity: Literal["info", "caveat", "warning"]
    summary: str
    details: dict[str, Any]
    
    def render_terminal(self) -> str: ...
    def render_html(self) -> str: ...
    def render_markdown(self) -> str: ...
    def render_figure_caption(self) -> str: ...
    def to_json(self) -> dict: ...
```

**Strict rule**: a finding's `summary` must be **identical text** across all renderers. Only formatting differs (color, bold, blockquote). The conclusion is the same.

#### Finding types (frozen at v0.3.0)

| Finding | Trigger | Severity | Example summary |
|---|---|---|---|
| `MostLikeBaseFinding` | min drift cell | info | "yarn on code (drift 0.0202)" |
| `BiggestMoveFinding` | max drift cell | info | "long on reasoning (drift 0.3355)" |
| `DirectionClusterFinding` | ≥3 variants pairwise cos > 0.90 | info | "{code, long, yarn} agree (cos ~0.95)" |
| `DirectionOutlierFinding` | one variant cos < 0.85 to all in cluster | info | "math stands apart (cos ~0.80 to cluster)" |
| `SpecializationPeakFinding` | per-variant share > 30% on a domain | info | "long: 66% on reasoning" |
| `AccuracyArtifactFinding` | accuracy ~0 on generative task with low max_new_tokens | caveat | "gsm8k accuracy ~0 likely max_new_tokens=16 artifact" |
| `TokenizerMismatchFinding` | base.tokenizer_id != variant.tokenizer_id | warning | "Cross-tokenizer; representation metrics unavailable" |
| `BaseAccuracyMissingFinding` | accuracy data exists for variants but not base | caveat | "Base accuracy not measured" |

### 12.7 Standard data flow for renderers

```
GeoResult / FamilyResult                             ← single source of truth
        │
        ├──→ findings = extract_findings(result)      ← lmdiff.geometry._findings
        │
        ├──→ tables = build_tables(result)            ← per-renderer
        │       (drift, share, direction, accuracy)
        │
        ├──→ figures = build_figures(result)          ← lmdiff.viz.*
        │
        └──→ output = renderer.render(                ← per-renderer assembly
                 findings, tables, figures,
                 layout="terminal-5-layer" | "html-section" | ...
             )
```

**Renderer is a thin assembly layer, not a logic layer**. All judgments happen in `findings`. All numbers come from `result`. All visuals come from `viz`.

If logic creeps into a renderer ("if drift > 0.2, add a caveat"), move it to `_findings.py` and have the renderer just display the finding.

### 12.8 Anti-patterns (what we tried and discarded)

#### Cell labels in dense matrices
**Tried**: `0.0451 / barely moved` in each cell. **Failed**: text overflow, visual noise, redundant.
**Now**: number only in cell, descriptive label in legend strip.

#### Continuous color gradient
**Tried**: viridis. **Failed**: application engineers can't read it; researchers don't gain over a 5-bucket discrete map.
**Now**: 5-bucket diverging or sequential.

#### Bottom line as a paragraph
**Tried**: a 4-sentence narrative paragraph below the figure. **Failed**: too long, application engineers skip; too rigid for auto-generation.
**Now**: short panel with bullet list, max 5 lines, all data-driven.

#### Mixing absolute and relative views in one chart
**Tried**: drift heatmap with z-score overlay. **Failed**: viewers can't tell which color is which.
**Now**: two side-by-side charts (drift left, share right). Each does one thing.

#### Domain knowledge in narrative
**Tried**: "yarn is for long context, so its high commonsense drift is unexpected." **Failed**: not portable, breaks if user supplies their own variants.
**Now**: only data-driven facts. Domain interpretation is the user's job.

#### Long titles
**Tried**: "Per-domain normalized magnitude with √(n_d · T̄_d) correction". **Failed**: nobody reads it.
**Now**: "How big was the change?" + technical subtitle.

#### Encoding "good vs bad" in metric colors
**Tried**: red = bad, blue = good. **Failed**: drift / share / direction don't have inherent good vs bad. They have direction.
**Now**: red and blue as direction or magnitude indicators. Good/bad lives only in the `accuracy` metric.

### 12.9 Application engineer first, researcher always

Hierarchy of design decisions:

1. **Application engineer reads the answer in 30 seconds**: hero / one-liner / Bottom line / first finding
2. **Researcher reads the data in 5 minutes**: tables / matrix values / cell labels
3. **Researcher verifies the answer**: methodology / formula / probe count
4. **Researcher extends the analysis**: JSON / raw data / loadable result

If a design decision conflicts between (1) and (2), do (1). The researcher will scroll for details. The application engineer won't.

But this never means **hiding** information from the researcher. It means **layering** — the researcher always finds it, just one click / section / paragraph deeper.

### 12.10 Calibration case: Llama-2 4-variant family

Any new metric or renderer should sanity-check against this case:

- **Findings should make sense**: the Llama-2 family has obvious specializations (code → code, math → math, etc.), so any rendering that obscures them is broken.
- **Cluster discovery should work**: yarn / long / code cluster vs math outlier should be discoverable by a casual reader.
- **Caveats should fire**: max_new_tokens=16 caveat for gsm8k must appear automatically.
- **Bottom line should be 100% data-driven**: no domain knowledge baked in.
- **Unmeasurable domains must render as unmeasurable** *(added v0.4.1)*: against a 4096-token base, long-context reports `—` in every surface, and no narrative sentence names it. A renderer that shows a number there is broken regardless of how sensible the number looks.
- **Flat variants must not be given a peak** *(added v0.4.1)*: in the 7-variant family, `chat` and `temp_1.5` lead their runner-up by 2.5pp and 4.1pp. Both should report "no dominant domain". A renderer that names `argmax` without a spread check is broken.

If a new design **fails to surface these patterns** on Llama-2 data, it's a design bug.

> **The reference values live in `tests/fixtures/calibration_v041_*_baseline.json`, not in this document.** Numbers written into a plan age silently — nobody diffs a design document against a fixture. This section says what the data should show; the fixture says what it does.

---

## 13. CLI summary example

> **The numbers in this section are historical and no longer reproducible.** They were generated under the v0.3.2 per-domain formula `sqrt(Σδ²/ΣT)`, which Update 5 established to be dimensionally inconsistent, and without the measurement-validity framework introduced in v0.4.1. Two things changed:
>
> - **The formula.** v0.4.1 uses plain unweighted RMS `sqrt(mean(δ²))` over valid probes (Update 5 Y.4, decision Q9.10).
> - **Which probes count.** Long-context probes exceeding the base model's context window are now excluded. Against a Llama-2-7B base (4096 tokens), 91 of 100 `longbench_2wikimqa` probes are out of range, and the domain classifies as `variant_only` or `out_of_range` rather than contributing a share.
>
> Current v0.4.1 numbers for the same family (long-context column is `—` for every variant):
>
> | variant | code | commonsense | long-context | math | reasoning |
> |---|---|---|---|---|---|
> | yarn | 7.5% | **78.7%** | — | 11.0% | 2.8% |
> | long | 18.3% | 7.7% | — | 27.2% | **46.9%** |
> | code | **52.1%** | 13.4% | — | 28.6% | 5.9% |
> | math | 27.4% | 16.5% | — | **42.6%** | 13.4% |
>
> The authoritative baseline is `tests/fixtures/calibration_v041_4variant_baseline.json`. The layout, layer structure, and rendering rules specified below remain correct and were implemented as written — only the values are stale. See Update 6 Z.6 for why this section is kept rather than rewritten.

This is the complete output of `lmdiff family --base llama2 --variants yarn,long,math,code` on the Llama-2 4-variant case study, rendered with default ANSI colors enabled. This is the **exact output Phase 1 commit 1.7 must produce** (with terminal-width adaptation).

The example shows all five layers:
- Layer 1: One-liner (line "Each variant acts biggest on a different domain")
- Layer 2: Headlines (4 bullets under "Headlines")
- Layer 3: Tables (4 tables: share, drift, cosine, accuracy)
- Layer 4: Caveats (yellow block at bottom)
- Layer 5: Pointers ("See also" with file paths)

```
══════════════════════════════════════════════════════════════════════════════
  Family experiment: Llama-2-7b vs 4 variants  (500 probes, 5 domains)
══════════════════════════════════════════════════════════════════════════════

Each variant acts biggest on a different domain:
  code → code (32%)   long → reasoning (66%)   math → math (35%)   yarn → commonsense (51%)

Headlines
  Most like base : yarn on code  (drift 0.0202)
  Biggest single move : long on reasoning  (drift 0.3355, 66% of long's budget)
  Direction cluster : {code, long, yarn}  (cos ~0.95)
  Direction outlier : math  (cos ~0.80 to cluster)

Where each variant acts biggest   share of total drift; rows sum to 100%
          hellaswag  arc_chal     gsm8k   mmlu-cs longbench   peak
  code         17%       13%       28%       32%       10%   code
  long          6%       66%       17%        7%        4%   reasoning
  math         17%       24%       35%       14%       10%   math
  yarn         51%        3%        6%        2%       38%   commonsense

How big is each move   per-domain drift magnitude
          hellaswag  arc_chal     gsm8k   mmlu-cs longbench   total
  code      0.0451    0.0400    0.0590    0.0626    0.0352   0.0360
  long      0.1020    0.3355    0.1719    0.1107    0.0827   0.0865
  math      0.0713    0.0856    0.1025    0.0645    0.0536   0.0545
  yarn      0.0931    0.0232    0.0312    0.0202    0.0802   0.0795

Direction agreement   cosine of δ vectors  (red = same direction; gray-purple = different)
             code     long     math     yarn
  code         —     +0.95    +0.79    +0.96
  long      +0.95       —     +0.80    +0.95
  math      +0.79    +0.80       —     +0.80
  yarn      +0.96    +0.95    +0.80       —

Per-task accuracy
          hellaswag  arc_chal     gsm8k   mmlu-cs longbench
  code        0.53      0.31     0.00*      0.32     0.00*
  long        0.61      0.45     0.00*      0.40     0.00*
  math        0.48      0.42     0.01*      0.47     0.00*
  yarn        0.55      0.41     0.04*      0.33     0.00*

Caveats
  * gsm8k & longbench accuracy ~0 likely a max_new_tokens=16 artifact,
    not a capability finding. Re-run with --task-max-new-tokens to verify.
  • Base accuracy not measured. Δaccuracy comparison skipped.
  • Drift magnitude shows where variants change, not whether changes help.
    Cross-reference with accuracy to judge variant choice.

See also
  Full results JSON   runs/llama2-4variants/family_geometry_lm_eval.json
  Geometry data       runs/llama2-4variants/family_geometry_lm_eval_georesult.json
  Detail figures      lmdiff plot-geometry runs/llama2-4variants/
  Metric definitions  docs/metrics.pdf

══════════════════════════════════════════════════════════════════════════════
```

When ANSI colors are enabled, this output uses:
- **Bold red** for `0.3355` (biggest drift), `+0.95`, `+0.96` (high cosine agreement), peak share cells (`32%`, `66%`, `35%`, `51%`)
- **Orange** for share-of-budget peaks
- **Yellow** for `*` accuracy artifact markers, `Caveats` heading, `max_new_tokens=16`
- **Green** for `yarn` (in "Most like base"), small drift values like `0.0202`
- **Dim gray** for subtitles ("share of total drift; rows sum to 100%"), peak labels in rightmost column, file paths
- **Default white** for body text

When colors disabled: bold replaces saturation, layout unchanged. Verified readable with `lmdiff family ... | tee log.txt`.

---

## 14. Reproducible figure code

This section contains the complete, runnable Python code for the four anchor figures referenced throughout the design discussion. All four follow the 5-rule template from section 12.1. They are intended as **executable specifications** for Phase 1 commit 1.8 (`drift_share_dual figure`) and 1.9 (`direction_agreement` and `change_size_bars` figures), as well as for Phase 1 commit 1.7 (`terminal renderer`).

### 14.1 Common setup

All four scripts assume:
- The Llama-2 4-variant georesult JSON is available at `family_geometry_lm_eval_georesult.json`
- Python 3.10+ with `numpy`, `matplotlib`, `Pillow` installed
- DejaVu Sans / DejaVu Sans Mono fonts available (Linux default)

Source data fields used: `variant_names`, `probe_domains`, `avg_tokens_per_probe`, `change_vectors`, `cosine_matrix`, `selective_cosine_matrix`, `magnitudes_normalized`.

### 14.2 Figure: `prototype_v5_clean.png` (drift + share dual-view)

This is the **main application-tier figure**. Implementation reference for Phase 1 commit 1.8.

```python
"""
prototype_v5_clean.py — drift + share dual-view heatmap.

Renders two heatmaps side-by-side:
- Left: How big was the change? (per-domain drift magnitude)
- Right: Where did the variant act biggest? (share-of-budget, rows sum to 100%)

Each follows the 5-rule template:
  1. Title is metaphor (not metric name)
  2. Color is 5-bucket discrete (sequential blue or diverging purple-orange)
  3. Cells contain only the number
  4. "How to read" line below
  5. Bottom-line panel on the right
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# ─── Load data ─────────────────────────────────────────────────────────
with open('family_geometry_lm_eval_georesult.json') as f:
    gr = json.load(f)

variants = sorted(gr['variant_names'])
domains_all = np.array(gr['probe_domains'])
T = np.array(gr['avg_tokens_per_probe'])
cv = {v: np.array(gr['change_vectors'][v]) for v in variants}
DOMAIN_ORDER = ['commonsense', 'reasoning', 'math', 'code', 'long-context']
TASK_LABELS = ['hellaswag', 'arc_chal', 'gsm8k', 'mmlu-cs', 'longbench']

# ─── Compute drift and share ───────────────────────────────────────────
norm = np.zeros((len(variants), len(DOMAIN_ORDER)))
for i, v in enumerate(variants):
    for j, d in enumerate(DOMAIN_ORDER):
        mask = domains_all == d
        n_d = mask.sum()
        Tbar = T[mask].mean()
        # √(n_d · T̄_d) normalization for per-domain comparison
        norm[i, j] = np.sqrt((cv[v][mask]**2).sum() / (n_d * Tbar))

# Share = squared magnitude / row sum of squared magnitudes (rows sum to 1)
norm_sq = norm ** 2
share = norm_sq / norm_sq.sum(axis=1, keepdims=True)

# ─── Figure layout ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 7.4))
gs = fig.add_gridspec(
    3, 3,
    width_ratios=[2.7, 2.7, 1.0],
    height_ratios=[5.5, 0.55, 0.7],
    hspace=0.30, wspace=0.18,
    left=0.05, right=0.98, top=0.83, bottom=0.05,
)
ax_abs = fig.add_subplot(gs[0, 0])
ax_z = fig.add_subplot(gs[0, 1])
ax_takeaway = fig.add_subplot(gs[0:2, 2])
ax_legend_abs = fig.add_subplot(gs[2, 0])
ax_legend_z = fig.add_subplot(gs[2, 1])
ax_takeaway.axis('off')
ax_legend_abs.axis('off')
ax_legend_z.axis('off')

# ─── Left: drift magnitude heatmap (sequential blue) ───────────────────
abs_max = norm.max()
boundaries_abs = [0, 0.025, 0.05, 0.10, 0.20, abs_max + 0.01]
colors_abs = ['#f0f0f0', '#c6dbef', '#6baed6', '#2171b5', '#08306b']
cmap_abs = ListedColormap(colors_abs)
norm_cmap_abs = BoundaryNorm(boundaries_abs, cmap_abs.N)
ax_abs.imshow(norm, cmap=cmap_abs, norm=norm_cmap_abs, aspect='auto')

for i, v in enumerate(variants):
    for j, d in enumerate(DOMAIN_ORDER):
        val = norm[i, j]
        text_color = 'white' if val > 0.10 else ('white' if val > 0.05 else '#222')
        ax_abs.text(j, i, f'{val:.4f}',
                    ha='center', va='center',
                    fontsize=15, fontweight='bold', color=text_color)

ax_abs.set_yticks(range(len(variants)))
ax_abs.set_yticklabels(variants, fontsize=12, fontweight='bold')
ax_abs.set_xticks(range(len(DOMAIN_ORDER)))
ax_abs.set_xticklabels([f'{t}\n({d})' for t, d in zip(TASK_LABELS, DOMAIN_ORDER)],
                        fontsize=9.5)
ax_abs.tick_params(axis='both', length=0)
for s in ax_abs.spines.values(): s.set_visible(False)
ax_abs.set_title("How big was the change?\n(per-domain drift magnitude — bigger value, bigger move)",
                  fontsize=11.5, color='#444', pad=10)

# ─── Right: share-of-budget heatmap (diverging purple-orange) ──────────
boundaries_share = [0, 0.10, 0.18, 0.22, 0.30, 1.0]
colors_share = ['#542788', '#b2abd2', '#f2f2f2', '#fdb863', '#b35806']
cmap_share = ListedColormap(colors_share)
norm_cmap_share = BoundaryNorm(boundaries_share, cmap_share.N)
ax_z.imshow(share, cmap=cmap_share, norm=norm_cmap_share, aspect='auto')

for i, v in enumerate(variants):
    for j, d in enumerate(DOMAIN_ORDER):
        sv = share[i, j]
        if sv < 0.10: c = 'white'
        elif sv < 0.18: c = '#3d2855'
        elif sv < 0.22: c = '#444'
        elif sv < 0.30: c = '#3d2855'
        else: c = 'white'
        ax_z.text(j, i, f'{sv*100:.0f}%',
                   ha='center', va='center',
                   fontsize=18, fontweight='bold', color=c)

ax_z.set_yticks(range(len(variants)))
ax_z.set_yticklabels(variants, fontsize=12, fontweight='bold')
ax_z.set_xticks(range(len(DOMAIN_ORDER)))
ax_z.set_xticklabels([f'{t}\n({d})' for t, d in zip(TASK_LABELS, DOMAIN_ORDER)],
                      fontsize=9.5)
ax_z.tick_params(axis='both', length=0)
for s in ax_z.spines.values(): s.set_visible(False)
ax_z.set_title("Where did the variant act biggest?\n(share of total change spent on each domain — rows sum to 100%)",
                fontsize=11.5, color='#444', pad=10)

# ─── Big titles ─────────────────────────────────────────────────────────
fig.text(0.05, 0.95,
         "How each variant differs from base — and where it acts biggest",
         fontsize=18, fontweight='bold', color='#222')
fig.text(0.05, 0.905,
         "Left: how big each move is.   Right: which domain the variant acts on most.",
         fontsize=11.5, color='#555', style='italic')

# ─── Legends ────────────────────────────────────────────────────────────
strip_x = 0.0; strip_w = 0.165
abs_legend_items = [
    ('#f0f0f0', '< 0.025\nbarely moved', '#222'),
    ('#c6dbef', '0.025–0.05\nsmall move', '#222'),
    ('#6baed6', '0.05–0.10\nmoderate', '#222'),
    ('#2171b5', '0.10–0.20\nbig move', 'white'),
    ('#08306b', '> 0.20\nhuge move', 'white'),
]
for k, (color, lbl, txt_color) in enumerate(abs_legend_items):
    cx = strip_x + k*(strip_w + 0.005)
    ax_legend_abs.add_patch(plt.Rectangle(
        (cx, 0.0), strip_w, 0.65,
        facecolor=color, edgecolor='#888', linewidth=0.6,
        transform=ax_legend_abs.transAxes, clip_on=False))
    ax_legend_abs.text(cx + strip_w/2, 0.32, lbl,
                       ha='center', va='center',
                       fontsize=8.5, color=txt_color, fontweight='bold',
                       transform=ax_legend_abs.transAxes, linespacing=1.2)
ax_legend_abs.text(0.0, 1.05,
                    "Smaller value = variant behaves more like base on this domain.",
                    ha='left', va='top', fontsize=9.5, color='#444',
                    transform=ax_legend_abs.transAxes)

share_legend_items = [
    ('#542788', '< 10%\nbarely acted', 'white'),
    ('#b2abd2', '10–18%\nsmall action', '#3d2855'),
    ('#f2f2f2', '18–22%\nbalanced', '#444'),
    ('#fdb863', '22–30%\nbig action', '#3d2855'),
    ('#b35806', '> 30%\nbiggest action', 'white'),
]
for k, (color, lbl, txt_color) in enumerate(share_legend_items):
    cx = strip_x + k*(strip_w + 0.005)
    ax_legend_z.add_patch(plt.Rectangle(
        (cx, 0.0), strip_w, 0.65,
        facecolor=color, edgecolor='#888', linewidth=0.6,
        transform=ax_legend_z.transAxes, clip_on=False))
    ax_legend_z.text(cx + strip_w/2, 0.32, lbl,
                     ha='center', va='center',
                     fontsize=8.5, color=txt_color, fontweight='bold',
                     transform=ax_legend_z.transAxes, linespacing=1.2)
ax_legend_z.text(0.0, 1.05,
                  "Each row sums to 100%.   Even split would be 20% per domain (5 domains).",
                  ha='left', va='top', fontsize=9.5, color='#444',
                  transform=ax_legend_z.transAxes)

# ─── Bottom line panel (right side) ────────────────────────────────────
ax_takeaway.text(0.0, 1.0, 'Bottom line',
                 fontsize=14, fontweight='bold', color='#222',
                 transform=ax_takeaway.transAxes, va='top')
ax_takeaway.text(
    0.0, 0.93,
    "These plots show how\nbig and where each\nvariant moved — not\nwhether the moves\nimproved performance.",
    fontsize=10, color='#333',
    transform=ax_takeaway.transAxes, va='top', linespacing=1.5,
)

flat_idx = np.unravel_index(norm.argmin(), norm.shape)
flat_idx2 = np.unravel_index(norm.argmax(), norm.shape)
bullets = [
    "• Most like base:",
    f"  {variants[flat_idx[0]]} on {DOMAIN_ORDER[flat_idx[1]]}",
    f"  ({norm[flat_idx]:.4f})",
    "",
    "• Biggest single move:",
    f"  {variants[flat_idx2[0]]} on {DOMAIN_ORDER[flat_idx2[1]]}",
    f"  ({norm[flat_idx2]:.4f})",
    "",
    "• Where each variant",
    "  acts biggest:",
]
for i, v in enumerate(variants):
    pk = share[i, :].argmax()
    pk_share = share[i, pk]
    bullets.append(f"  {v}: {pk_share*100:.0f}% on {DOMAIN_ORDER[pk]}")

y0 = 0.62
for i, line in enumerate(bullets):
    ax_takeaway.text(0.0, y0 - i*0.043, line,
                     fontsize=9, color='#222',
                     transform=ax_takeaway.transAxes, va='top',
                     family='DejaVu Sans Mono', linespacing=1.3)

plt.savefig('prototype_v5_clean.png', facecolor='white', dpi=180)
plt.close()
```


### 14.3 Figure: `prototype_cosine.png` (direction agreement, dual-view)

Shows raw and selective cosine matrices side-by-side, with cluster/outlier auto-detection in the bottom-line panel. Implementation reference for Phase 1 commit 1.9.

```python
"""
prototype_cosine.py — cosine similarity matrices (raw + selective).

Renders two cosine matrices side-by-side:
- Left: All differences (raw cosine = full direction agreement)
- Right: Probe-specific differences only (selective = mean-removed)

Includes auto-detection of cluster (≥3 variants pairwise cos > 0.90) and
outlier (1 variant with cos < 0.85 to all in cluster). Bottom line reports
both raw and selective cluster gaps.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
from itertools import combinations

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

with open('family_geometry_lm_eval_georesult.json') as f:
    gr = json.load(f)

variants = sorted(gr['variant_names'])
cm_raw = np.array([[gr['cosine_matrix'][a][b] for b in variants] for a in variants])
cm_sel = np.array([[gr['selective_cosine_matrix'][a][b] for b in variants] for a in variants])

# 5-bucket sequential palette for cosine: uncorrelated → near identical
boundaries = [-1.01, 0.30, 0.70, 0.85, 0.95, 1.01]
colors_list = ['#1f4e79', '#9ec5e8', '#f2f2f2', '#f5b9a8', '#c0392b']
labels_strip = [
    ('#1f4e79', '✗ uncorrelated\n< 0.30', 'white'),
    ('#9ec5e8', '↓ weak\n0.30–0.70', '#1f3a52'),
    ('#f2f2f2', '· moderate\n0.70–0.85', '#444444'),
    ('#f5b9a8', '↑ strong\n0.85–0.95', '#5a1f17'),
    ('#c0392b', '✓ near identical\n> 0.95', 'white'),
]
cmap = ListedColormap(colors_list)
norm_cmap = BoundaryNorm(boundaries, cmap.N)

fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(
    3, 3,
    width_ratios=[2.7, 2.7, 1.0],
    height_ratios=[5.5, 0.6, 0.7],
    hspace=0.30, wspace=0.20,
    left=0.06, right=0.98, top=0.85, bottom=0.05,
)
ax_raw = fig.add_subplot(gs[0, 0])
ax_sel = fig.add_subplot(gs[0, 1])
ax_takeaway = fig.add_subplot(gs[0:2, 2])
ax_legend = fig.add_subplot(gs[2, 0:2])
ax_takeaway.axis('off'); ax_legend.axis('off')

def draw_cos_matrix(ax, mat, title_top, title_bot):
    ax.imshow(mat, cmap=cmap, norm=norm_cmap)
    n = len(variants)
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if i == j:
                num_str = '—'; lbl = 'self'
                num_color = '#888'; txt_color = '#999'
            else:
                if v > 0.95: lbl, num_color, txt_color = '✓ near identical', 'white', 'white'
                elif v > 0.85: lbl, num_color, txt_color = '↑ strong', '#5a1f17', '#5a1f17'
                elif v > 0.70: lbl, num_color, txt_color = '· moderate', '#444', '#666'
                elif v > 0.30: lbl, num_color, txt_color = '↓ weak', '#1f3a52', '#1f3a52'
                else: lbl, num_color, txt_color = '✗ uncorrelated', 'white', 'white'
                num_str = f'{v:+.2f}'
            ax.text(j, i - 0.18, num_str, ha='center', va='center',
                    fontsize=13, fontweight='bold', color=num_color)
            ax.text(j, i + 0.22, lbl, ha='center', va='center',
                    fontsize=8.5, color=txt_color, style='italic')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(variants, fontsize=11, fontweight='bold')
    ax.set_yticklabels(variants, fontsize=11, fontweight='bold')
    ax.tick_params(axis='both', length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(f'{title_top}\n{title_bot}', fontsize=11, color='#444', pad=10)

draw_cos_matrix(ax_raw, cm_raw,
                "All differences",
                "raw cosine — full direction agreement")
draw_cos_matrix(ax_sel, cm_sel,
                "Probe-specific differences only",
                "selective cosine — uniform offset removed")

fig.text(0.06, 0.95, "Who pushes the base in the same direction?",
         fontsize=20, fontweight='bold', color='#222')
fig.text(0.06, 0.905,
         "Two variants compared by their per-probe deviations from base",
         fontsize=11.5, color='#555', style='italic')

# Legend strip
ax_legend.text(0.0, 1.05,
               "Each cell asks: do variants A and B agree on which probes drift more or less? "
               "+1 = perfect agreement; 0 = independent.",
               ha='left', va='top', fontsize=10, color='#444',
               transform=ax_legend.transAxes)
strip_x = 0.04; strip_w = 0.075
for k, (color, lbl, txt_color) in enumerate(labels_strip):
    cx = strip_x + k*(strip_w + 0.012)
    ax_legend.add_patch(plt.Rectangle(
        (cx, 0.05), strip_w, 0.55,
        facecolor=color, edgecolor='#888', linewidth=0.6,
        transform=ax_legend.transAxes, clip_on=False))
    ax_legend.text(cx + strip_w/2, 0.32, lbl,
                   ha='center', va='center',
                   fontsize=8, color=txt_color, fontweight='bold',
                   transform=ax_legend.transAxes, linespacing=1.2)

# Bottom line: data-driven cluster/outlier detection
def find_cluster_outlier(mat, t_cluster=0.90, t_outlier=0.85):
    n = len(variants)
    best = []
    for size in range(n, 1, -1):
        for combo in combinations(range(n), size):
            ok = all(mat[i, j] > t_cluster
                     for i in combo for j in combo if i != j)
            if ok: best = list(combo); break
        if best: break
    cluster = [variants[i] for i in best]
    outliers = []
    for i in range(n):
        if i not in best:
            cs = [mat[i, j] for j in best]
            if all(c < t_outlier for c in cs):
                outliers.append((variants[i], np.mean(cs)))
    return cluster, outliers

cluster_raw, outlier_raw = find_cluster_outlier(cm_raw)
cluster_sel, outlier_sel = find_cluster_outlier(cm_sel, 0.90, 0.85)

mean_cluster_raw = np.mean([cm_raw[i, j]
    for i in [variants.index(v) for v in cluster_raw]
    for j in [variants.index(v) for v in cluster_raw] if i != j])

ax_takeaway.text(0.0, 1.0, 'Bottom line',
                 fontsize=14, fontweight='bold', color='#222',
                 transform=ax_takeaway.transAxes, va='top')
ax_takeaway.text(
    0.0, 0.92,
    f"3 of 4 variants align\ndirectionally\n(cos ≈ {mean_cluster_raw:.2f}).\nOne stands apart.",
    fontsize=10.5, color='#333',
    transform=ax_takeaway.transAxes, va='top', linespacing=1.55,
)

bullets = []
bullets.append(f"• Aligned cluster:")
bullets.append(f"  {{ {', '.join(cluster_raw)} }}")
bullets.append("")
if outlier_raw:
    o, c = outlier_raw[0]
    bullets.append(f"• Outlier:")
    bullets.append(f"  {o}  (cos~{c:.2f})")
bullets.append("")
gap_raw = mean_cluster_raw - (outlier_raw[0][1] if outlier_raw else 0)
mean_cluster_sel = np.mean([cm_sel[i,j]
    for i in [variants.index(v) for v in cluster_sel]
    for j in [variants.index(v) for v in cluster_sel] if i != j]) if len(cluster_sel) >= 2 else 0
gap_sel = mean_cluster_sel - (outlier_sel[0][1] if outlier_sel else 0)
bullets.append(f"• Gap: {gap_raw:+.2f} raw")
bullets.append(f"        {gap_sel:+.2f} selective")
if gap_sel > gap_raw:
    bullets.append(f"  Gap widens after")
    bullets.append(f"  removing offset →")
    bullets.append(f"  real direction split.")

y0 = 0.65
for i, line in enumerate(bullets):
    ax_takeaway.text(0.0, y0 - i*0.045, line,
                     fontsize=9.5, color='#222',
                     transform=ax_takeaway.transAxes, va='top',
                     family='DejaVu Sans Mono', linespacing=1.3)

plt.savefig('prototype_cosine.png', facecolor='white', dpi=180)
plt.close()
```

### 14.4 Figure: `prototype_magnitude.png` (raw vs normalized magnitude bars)

Shows why per-token normalization is necessary by contrasting raw and normalized magnitude bars for the same data. The hatched portions in raw bars highlight that long-context probes dominate the raw signal. Implementation reference for Phase 1 commit 1.9 (`change_size_bars`).

```python
"""
prototype_magnitude.py — raw vs normalized magnitude comparison.

Two bar charts side-by-side:
- Left: raw ‖δ‖ (with hatched portion showing longbench contribution)
- Right: ‖δ‖ per √token (after normalization)

The left chart reveals that long-context probes dominate raw magnitude
(88-99% of ‖δ‖² across variants). The right chart shows the corrected
per-token drift, which can rank variants meaningfully.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

with open('family_geometry_lm_eval_georesult.json') as f:
    gr = json.load(f)

variants = sorted(gr['variant_names'])
domains_all = np.array(gr['probe_domains'])
T = np.array(gr['avg_tokens_per_probe'])
cv = {v: np.array(gr['change_vectors'][v]) for v in variants}

# Compute three numbers per variant
raw = {v: float(np.sqrt((cv[v]**2).sum())) for v in variants}
norm = {v: gr['magnitudes_normalized'][v] for v in variants}
mask_long = domains_all == 'long-context'
pct_long = {v: 100 * (cv[v][mask_long]**2).sum() / (cv[v]**2).sum() for v in variants}

# Order by normalized magnitude descending
order = sorted(variants, key=lambda v: -norm[v])

fig = plt.figure(figsize=(14, 6.2))
gs = fig.add_gridspec(
    2, 3,
    width_ratios=[2.7, 2.7, 1.0],
    height_ratios=[5.5, 0.5],
    hspace=0.35, wspace=0.20,
    left=0.07, right=0.98, top=0.83, bottom=0.10,
)
ax_raw = fig.add_subplot(gs[0, 0])
ax_norm = fig.add_subplot(gs[0, 1])
ax_takeaway = fig.add_subplot(gs[0, 2])
ax_legend = fig.add_subplot(gs[1, 0:2])
ax_takeaway.axis('off'); ax_legend.axis('off')

# Per-variant colors (consistent across figures)
COLORS = {'code':'#1f77b4','long':'#2ca02c','math':'#9467bd','yarn':'#d62728'}

# Left: raw bars with longbench % overlay
y_pos = np.arange(len(order))
for k, v in enumerate(order):
    ax_raw.barh(k, raw[v], color=COLORS[v], edgecolor='#333', linewidth=0.6)
    ax_raw.barh(k, raw[v]*pct_long[v]/100,
                color=COLORS[v], alpha=0.4, hatch='///',
                edgecolor='white', linewidth=0)
    ax_raw.text(raw[v] + 1.5, k, f'{raw[v]:.1f}',
                va='center', fontsize=11, fontweight='bold')
    ax_raw.text(raw[v]*pct_long[v]/200, k,
                f'longbench\n{pct_long[v]:.1f}%',
                va='center', ha='center', fontsize=8,
                color='white', fontweight='bold')

ax_raw.set_yticks(y_pos)
ax_raw.set_yticklabels(order, fontsize=12, fontweight='bold')
ax_raw.invert_yaxis()
ax_raw.set_xlim(0, max(raw.values())*1.18)
ax_raw.set_xlabel('‖δ‖ raw', fontsize=10)
ax_raw.set_title("Before normalization — long probes dominate",
                 fontsize=11, color='#444', pad=10)
for s in ['top','right']: ax_raw.spines[s].set_visible(False)
ax_raw.tick_params(axis='both', length=2)

# Right: normalized bars
for k, v in enumerate(order):
    ax_norm.barh(k, norm[v], color=COLORS[v], edgecolor='#333', linewidth=0.6)
    ax_norm.text(norm[v] + 0.002, k, f'{norm[v]:.4f}',
                 va='center', fontsize=11, fontweight='bold')

ax_norm.set_yticks(y_pos)
ax_norm.set_yticklabels(order, fontsize=12, fontweight='bold')
ax_norm.invert_yaxis()
ax_norm.set_xlim(0, max(norm.values())*1.25)
ax_norm.set_xlabel('‖δ‖ per √token', fontsize=10)
ax_norm.set_title("After normalization — comparable across domains",
                  fontsize=11, color='#444', pad=10)
for s in ['top','right']: ax_norm.spines[s].set_visible(False)
ax_norm.tick_params(axis='both', length=2)

# Big titles
fig.text(0.07, 0.94, "How far has each variant moved from base?",
         fontsize=20, fontweight='bold', color='#222')
fig.text(0.07, 0.895,
         "Hatched portion = share dominated by long-context probes "
         "(a length artifact, not real drift)",
         fontsize=11, color='#555', style='italic')

# How-to-read line
ax_legend.text(0.5, 0.5,
               "Longer raw bars don't mean larger real change — they mean longer probes. "
               "Per-token normalization (right) shows the actual per-token drift.",
               ha='center', va='center', fontsize=10, color='#444',
               transform=ax_legend.transAxes)

# Bottom line
ax_takeaway.text(0.0, 1.0, 'Bottom line',
                 fontsize=14, fontweight='bold', color='#222',
                 transform=ax_takeaway.transAxes, va='top')
ax_takeaway.text(0.0, 0.92,
    "Longbench probes are 100×\nlonger than other tasks.\n"
    "Without normalization,\nthey hide real differences.",
    fontsize=10.5, color='#333',
    transform=ax_takeaway.transAxes, va='top', linespacing=1.5)

ranking_norm = sorted(variants, key=lambda v: -norm[v])
lines = ["• Per-token ranking:"]
for i, v in enumerate(ranking_norm):
    lines.append(f"  {i+1}. {v}  {norm[v]:.4f}")
lines.append("")
lines.append("• Closest to base:")
lines.append(f"  {ranking_norm[-1]}")
lines.append("")
ranking_raw = sorted(variants, key=lambda v: -raw[v])
if ranking_norm != ranking_raw:
    lines.append("• Raw vs normalized")
    lines.append("  rankings differ.")

y0 = 0.62
for i, line in enumerate(lines):
    ax_takeaway.text(0.0, y0 - i*0.055, line,
                     fontsize=9.5, color='#222',
                     transform=ax_takeaway.transAxes, va='top',
                     family='DejaVu Sans Mono', linespacing=1.3)

plt.savefig('prototype_magnitude.png', facecolor='white', dpi=180)
plt.close()
```


### 14.5 Figure: `cli_mockup_v2.png` (terminal output rendered as PNG)

Renders the complete 5-layer CLI summary as a PNG image (for embedding in design docs and presentations). Useful as a non-runtime preview of what `lmdiff family` produces. Implementation reference for Phase 1 commit 1.7.

```python
"""
cli_mockup_v2.py — render the 5-layer terminal summary as a PNG.

Re-implements the terminal renderer logic but draws to PIL Image instead
of stdout. Used to preview/validate the terminal output design without
requiring an actual ANSI-capable terminal.
"""

from PIL import Image, ImageDraw, ImageFont
import json
import numpy as np

# ─── Load data ─────────────────────────────────────────────────────────
with open('family_geometry_lm_eval_georesult.json') as f:
    gr = json.load(f)

variants = sorted(gr['variant_names'])
domains_all = np.array(gr['probe_domains'])
T = np.array(gr['avg_tokens_per_probe'])
cv = {v: np.array(gr['change_vectors'][v]) for v in variants}
DOMAIN_ORDER = ['commonsense', 'reasoning', 'math', 'code', 'long-context']
TASK_LABELS = ['hellaswag', 'arc_chal', 'gsm8k', 'mmlu-cs', 'longbench']

norm = np.zeros((len(variants), len(DOMAIN_ORDER)))
for i, v in enumerate(variants):
    for j, d in enumerate(DOMAIN_ORDER):
        mask = domains_all == d
        norm[i, j] = np.sqrt((cv[v][mask]**2).sum() / (mask.sum() * T[mask].mean()))
share = (norm**2) / (norm**2).sum(axis=1, keepdims=True)
total_norm = {v: gr['magnitudes_normalized'][v] for v in variants}
cm = np.array([[gr['cosine_matrix'][a][b] for b in variants] for a in variants])

# Hardcoded accuracy from earlier experiment runs
acc_table = {
    'yarn': [0.55, 0.41, 0.04, 0.33, 0.00],
    'long': [0.61, 0.45, 0.00, 0.40, 0.00],
    'code': [0.53, 0.31, 0.00, 0.32, 0.00],
    'math': [0.48, 0.42, 0.01, 0.47, 0.00],
}
ARTIFACT_IDX = {2, 4}  # gsm8k, longbench artifacts

# ─── ANSI-equivalent palette for "terminal-like" PNG ───────────────────
PAL = dict(
    fg='#d4d4d4',     # default white
    dim='#888',       # dim gray
    red='#e06c75',    # red
    orange='#ff9c4f', # orange
    yellow='#dcdfa3', # yellow
    green='#98c379',  # green
    purple='#c678dd', # purple
)

# ─── Build line list (each line = list of (text, color, bold) tuples) ──
LINES = []
def line(*segs): LINES.append(list(segs))
def blank(): LINES.append([])
def seg(t, color=PAL['fg'], bold=False, dim=False):
    if dim: color = PAL['dim']
    return (t, color, bold)

# Header
SEP = '═' * 78
line(seg(SEP, bold=True))
line(seg('  Family experiment: Llama-2-7b vs 4 variants  (500 probes, 5 domains)', bold=True))
line(seg(SEP, bold=True))
blank()

# Layer 1: One-liner
line(seg('Each variant acts biggest on a different domain:', bold=True))
peaks = [(v, DOMAIN_ORDER[share[i].argmax()], share[i].max())
         for i, v in enumerate(variants)]
parts = []
for v, d, s in peaks:
    parts.append(seg('  '))
    parts.append(seg(v, bold=True))
    parts.append(seg(f' → {d} ({s*100:.0f}%)'))
LINES.append(parts)
blank()

# Layer 2: Headlines
line(seg('Headlines', bold=True))
flat_min = np.unravel_index(norm.argmin(), norm.shape)
flat_max = np.unravel_index(norm.argmax(), norm.shape)
line(seg('  Most like base : '),
     seg(f'{variants[flat_min[0]]}', PAL['green'], bold=True),
     seg(f' on {DOMAIN_ORDER[flat_min[1]]}'),
     seg(f'  (drift {norm[flat_min]:.4f})', dim=True))
line(seg('  Biggest single move : '),
     seg(f'{variants[flat_max[0]]}', PAL['red'], bold=True),
     seg(f' on {DOMAIN_ORDER[flat_max[1]]}'),
     seg(f'  (drift {norm[flat_max]:.4f}, '
         f'{share[flat_max]*100:.0f}% of {variants[flat_max[0]]}\'s budget)', dim=True))
line(seg('  Direction cluster : '),
     seg('{code, long, yarn}', bold=True),
     seg('  (cos ~0.95)', dim=True))
line(seg('  Direction outlier : '),
     seg('math', bold=True),
     seg('  (cos ~0.80 to cluster)', dim=True))
blank()

# Layer 3: Where (share) table
line(seg('Where each variant acts biggest', bold=True),
     seg('   share of total drift; rows sum to 100%', dim=True))
hdr = '         ' + ''.join(f'{l:>10}' for l in TASK_LABELS) + '   peak'
line(seg(hdr, dim=True))
for i, v in enumerate(variants):
    row = [seg(f'  {v:<6}', bold=True), seg(' ')]
    pk = share[i].argmax()
    for j in range(5):
        s = share[i, j]
        text = f'{s*100:>4.0f}%'.rjust(10)
        if j == pk: row.append(seg(text, PAL['orange'], bold=True))
        elif s >= 0.22: row.append(seg(text, PAL['yellow']))
        elif s >= 0.10: row.append(seg(text))
        else: row.append(seg(text, dim=True))
    row.append(seg(f'   {DOMAIN_ORDER[pk][:11]}', dim=True))
    LINES.append(row)
blank()

# Layer 3: Drift table
line(seg('How big is each move', bold=True),
     seg('   per-domain drift magnitude', dim=True))
hdr = '         ' + ''.join(f'{l:>10}' for l in TASK_LABELS) + '   total'
line(seg(hdr, dim=True))
for i, v in enumerate(variants):
    row = [seg(f'  {v:<6}', bold=True), seg(' ')]
    for j in range(5):
        val = norm[i, j]
        text = f'{val:.4f}'.rjust(10)
        if val >= 0.20: row.append(seg(text, PAL['red'], bold=True))
        elif val >= 0.10: row.append(seg(text, PAL['red']))
        elif val >= 0.05: row.append(seg(text))
        elif val >= 0.025: row.append(seg(text, dim=True))
        else: row.append(seg(text, PAL['green']))
    row.append(seg(f'   {total_norm[v]:.4f}', bold=True))
    LINES.append(row)
blank()

# Layer 3: Direction table
line(seg('Direction agreement', bold=True),
     seg('   cosine of δ vectors  (red = same direction; gray-purple = different)', dim=True))
hdr = '         ' + ''.join(f'{v:>9}' for v in variants)
line(seg(hdr, dim=True))
for i, va in enumerate(variants):
    row = [seg(f'  {va:<6}', bold=True), seg(' ')]
    for j, vb in enumerate(variants):
        if i == j: row.append(seg('       —'))
        else:
            val = cm[i, j]
            text = f'+{val:.2f}'.rjust(9)
            if val >= 0.95: row.append(seg(text, PAL['red'], bold=True))
            elif val >= 0.85: row.append(seg(text, PAL['red']))
            elif val >= 0.70: row.append(seg(text))
            else: row.append(seg(text, PAL['purple']))
    LINES.append(row)
blank()

# Layer 3: Accuracy table
line(seg('Per-task accuracy', bold=True))
hdr = '         ' + ''.join(f'{l:>10}' for l in TASK_LABELS)
line(seg(hdr, dim=True))
for v in variants:
    row = [seg(f'  {v:<6}', bold=True), seg(' ')]
    for j, a in enumerate(acc_table[v]):
        if j in ARTIFACT_IDX:
            text = f'{a:.2f}*'.rjust(10)
            row.append(seg(text, PAL['yellow']))
        else:
            text = f'{a:.2f} '.rjust(10)
            row.append(seg(text))
    LINES.append(row)
blank()

# Layer 4: Caveats
line(seg('Caveats', PAL['yellow'], bold=True))
line(seg('  '), seg('*', PAL['yellow'], bold=True),
     seg(' gsm8k & longbench accuracy ~0 likely a '),
     seg('max_new_tokens=16', bold=True),
     seg(' artifact,'))
line(seg('    not a capability finding. Re-run with --task-max-new-tokens to verify.'))
line(seg('  • Base accuracy not measured. Δaccuracy comparison skipped.'))
line(seg('  • Drift magnitude shows '), seg('where', bold=True),
     seg(' variants change, not whether changes help.'))
line(seg('    Cross-reference with accuracy to judge variant choice.'))
blank()

# Layer 5: Pointers
line(seg('See also', bold=True))
line(seg('  Full results JSON  ', dim=True),
     seg('runs/llama2-4variants/family_geometry_lm_eval.json'))
line(seg('  Geometry data      ', dim=True),
     seg('runs/llama2-4variants/family_geometry_lm_eval_georesult.json'))
line(seg('  Detail figures     ', dim=True),
     seg('lmdiff plot-geometry runs/llama2-4variants/'))
line(seg('  Metric definitions ', dim=True), seg('docs/metrics.pdf'))
blank()
line(seg(SEP, bold=True))

# ─── Render to PIL ─────────────────────────────────────────────────────
font_reg = ImageFont.truetype(
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 14)
font_bold = ImageFont.truetype(
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 14)
char_w = font_reg.getbbox('M')[2]
line_h = 21
W = char_w * 86
H = line_h * (len(LINES) + 2) + 20
img = Image.new('RGB', (W, H), '#1e1e1e')
draw = ImageDraw.Draw(img)
y = 10
for ln in LINES:
    x = 10
    for text, color, bold in ln:
        f = font_bold if bold else font_reg
        draw.text((x, y), text, fill=color, font=f)
        x += f.getbbox(text)[2]
    y += line_h
img.save('cli_mockup_v2.png')
```

### 14.6 Reproducibility checklist

For Phase 1 commit 1.7-1.10 implementation, validate against these reference outputs:

| Output | Source script | Validates |
|---|---|---|
| `prototype_v5_clean.png` | section 14.2 | drift_share dual-view figure (Phase 1 commit 1.8) |
| `prototype_cosine.png` | section 14.3 | direction_agreement figure (Phase 1 commit 1.9) |
| `prototype_magnitude.png` | section 14.4 | change_size_bars figure (Phase 1 commit 1.9) |
| `cli_mockup_v2.png` | section 14.5 | terminal renderer logic (Phase 1 commit 1.7) |

When implementing the renderer modules:
1. Run the corresponding reference script first → produces the expected output
2. Implement the lmdiff API to call out to the renderer
3. Diff: rendered output should be **byte-identical** modulo timestamp and version metadata
4. CI snapshot test: re-render on every PR, fail if diff exceeds tolerance

This prevents drift between the design intent (these scripts) and the implementation.

---

## 15. Metrics roadmap summary

### Stable in v1.0 (default-enabled)

**Output / behavioral** (v0.3.0–v0.7.0):
- BehavioralDistance (BD), TokenKL, ΔEntropy
- Drift (raw + RMS-normalized) *— per-domain normalization is `sqrt(mean(δ²))` over valid probes as of v0.4.1; the earlier `sqrt(Σδ²/ΣT)` was dimensionally inconsistent (Update 5)*
- Share-of-budget *— computed over measured domains only; `None` where the base could not measure a domain (v0.4.1)*
- Direction (raw cosine + selective cosine = Pearson r)
- Specialization z-score *— aggregates over measured domains only as of v0.4.2*
- Token-level cosine
- Perplexity Shift by Domain
- ECE Shift
- Confidence-Correctness Correlation Diff
- Semantic Entropy Diff
- EAS Diff
- Hallucination Rate
- Safety Regression Rate
- Consistency Score
- Crosslingual Consistency

*Every metric added from here needs a defined not-measurable state rather than a fallback value — see carry-over note 16 in Update 6 Z.9. Fallback values are how the √T problem started.*

**Representation / structural** (v0.8.0):
- CKA (linear + RBF)
- PWCCA
- SVCCA
- Effective Attention Rank Diff
- Steering Vector Direction Diff
- Intrinsic Dimensionality Shift (TwoNN + MLE)

**Trajectory / cloud** (v0.9.0):
- Tuned Lens Trajectory Diff
- Cloud MMD / Energy / C2ST

### Experimental in v1.0 (opt-in via `lmdiff.contrib`)

- Logit Lens Diff (Llama-only-reliable; doc-warned)
- Latent Functional Maps (hyperparameter-sensitive)
- Best-of-N Cloud Diff (sampling-budget-sensitive)

### Deferred to v2.0

- Attention Head Functional Role Drift
- Concept-level Probing Diff
- Feature Attribution Drift
- SAE Feature Overlap
- Knowledge Neuron Tracking
- Activation Patching / Causal Tracing
- Model Stitching
- Long-trajectory / agent metrics

Engine and Metric protocols leave hooks for all of these via reserved capability names.

---

## 16. Configuration roadmap summary

### Supported in v1.0

| Configuration type | Status | Notes |
|---|---|---|
| Model weights (HF) | ✓ | HFEngine default |
| LoRA / QLoRA / IA³ adapters | ✓ | `Config.adapter` |
| Quantization (INT8/INT4/GPTQ/AWQ) | ✓ | `Config.quantization` |
| Pruning | ✓ | Load already-pruned model |
| KV-cache compression | ✓ | `Config.kv_cache_compression` |
| System prompt | ✓ | `Config.system_prompt` |
| ICL / few-shot | ✓ | `Config.icl_examples` |
| Multi-turn context | ✓ | `Config.context` |
| Soft prompts | ✓ | `Config.soft_prompts` |
| Decoding (greedy/sample/beam/best-of-N/self-consistency) | ✓ | `Config.decode` |
| Steering vectors | ✓ | `Config.steering` (Phase 5) |
| Custom backend (non-HF) | ✓ | User implements MinimalEngine |
| Hosted APIs (OpenAI/Anthropic) | ✓ | APIEngine (Phase 6) |
| **Run configuration as a file** | ✓ | YAML round-trips the whole `Config` set; emitted with every report (Update 7) |
| **CoT (explicit spec)** | △ | Use `system_prompt` + `decode`; explicit CoT spec considered for cloud-distance scope |

### Deferred to v2.0

- Test-time training / online adaptation
- Agentic scaffolding (ReAct, reflection loops)
- Tool-use traces

---

## 17. Cross-phase work patterns

| Work | Pattern |
|---|---|
| Sphinx scaffolding | Starts Week 2 of Phase 1 (sidecar). Content accumulates per-PR. Polish in Phase 3. |
| Tests | Every commit ships tests. Verification scope is `pytest tests/` — the command CI runs, not a subset (L-034). Phase 7 adds external-driven test cases. |
| Examples backlog | Initial entries in Phase 3. Grow opportunistically. Phase 7 adds 3+ from external users. |
| API stabilization | Shape locked v0.3.0. Signatures tunable v0.3.x–v0.8.x via deprecation. Hard freeze v1.0. |
| Engine protocol expansion | Reserved names listed Phase 1; three required methods added in Phase 2 (§4.3); concrete optional methods in Phase 5 (hidden_states, attention_weights, steering) and Phase 6 (sampling_cloud). |
| Experimental metric promotion | A contrib metric can graduate to stable in a later release if validated; reverse possible if found unreliable. |
| **Design audit before implementation** | Any commit introducing or changing a metric, a schema, or a cross-cutting abstraction gets an audit document first — code paths traced with line references, numerical impact computed on existing fixtures, open questions enumerated with recommendations, and **no production code written**. Established for commit 4.0 (Update 4 X.3), refined for 4.1 (Update 6 Z.7), where it moved five latent bugs from post-merge to pre-merge. Empirical scripts in the audit cost seconds of CPU and repeatedly overturned design hypotheses that inspection had approved. |
| **Release checklist** | Every user-facing output path exercised on data representative of what the release changed. Not "tests pass" — `to_html()` was broken for the entirety of v0.4.1 on that release's own headline scenario, with CI green, because nothing in the process rendered it. |
| **Thresholds** | Any constant gating a claim is named, documented at its definition with its derivation, and re-derived after any change that rescales the quantity it gates. Bare literals make the post-change audit intractable (L-037). |
| **Lessons** | Recorded in `LESSONS.md` after each release, when citation URLs are stable. L-001 through L-037 as of v0.4.1; drafts for L-038 and L-039 pending v0.4.2. |

---

## 18. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| ~~Phase 1 too large (12 commits)~~ **Occurred** — needed 2 unplanned fat releases | — | Realized as v0.3.1 hotfix + v0.3.2. Retrospective in Update 3. |
| Configuration class fields underspecified | Medium | Phase 1 scoped 6-8 weeks; types defined upfront for all v0.7-target configs. Deprecation cycle accommodates adjustments. |
| Phase 4 breadth (10 new metrics) overruns | Medium-high | Cuttable scope: Hallucination/Safety/Consistency/Crosslingual could ship in a v0.7.x patch if Phase 4 stalls. |
| Phase 5 representation metrics numerical issues | Low-medium | Reference tests catch early; ship CKA + steering + ID alone if PWCCA/SVCCA falter. |
| Phase 6 cloud encoder choice locks in poor default | Medium | Multiple encoders shipped, no single default forced; documented tradeoffs. |
| External testing reveals deep design flaw | High-impact, low-likelihood | Phase 7 polish window 3-4 weeks specifically buffers this. Worst case: v1.0 slips by 2-4 weeks. |
| Multi-GPU `accelerate` integration quirks | Medium | Mock CI + manual smoke; if accelerate too brittle, fallback to simple multiprocessing (1 week additional). |
| ~~6-8 month timeline slips to 9+ months~~ **Occurred.** Revised to 10–13 months (§19) | — | Phase 4 + Phase 5 metric breadth remain the most cuttable; minimum viable path is "core metrics + multi-GPU + docs + LTS" without representation extensions. |
| A shipped metric turns out to measure the wrong thing | **Occurred** (√T normalization, Update 5) | Methodology decisions ship with published derivations, not just numbers. Design-audit-before-implement for any contested metric. Validate against external ground truth, never against a mockup derived from the same formula (L-033). |
| A correctness fix breaks a downstream presentation threshold | **Occurred** (v0.4.1→v0.4.2, L-037) | After any rescale or renormalization, re-derive every downstream threshold and re-render every output surface. Assertion suites do not cover thresholds on presentation. |
| Experimental metric users expect stable-grade reliability | Low-medium | Opt-in mechanism + warning at compute time + clear contrib namespace + documentation calling out known limitations. |

---

## 19. Time budget summary

Steady pace (~22 h/week from you, CC continuous):

| Phase | Budgeted | Actual | Cumulative |
|---|---|---|---|
| 0 | done | done | 0 |
| 1 | 6–8 (12 commits, larger scope incl. UX) | **~9** (v0.3.0 + 2 fat releases) | 9 |
| 2 | 3–4 | **~4 so far**, 8–12 projected (7 commits, multi-release) | 17–21 |
| 3 | 5–7 (concentrated; content from W2 onward) | — | 22–28 |
| 4 | 4–5 | — | 26–33 |
| 5 | 5–6 | — | 31–39 |
| 6 | 5–6 | — | 36–45 |
| 7 | 3–4 | — | 39–49 |
| 8 | 2–3 | — | **41–52 weeks** |

Plus **v0.5.0** (cross-cutting, between Phase 2 and 3): 2–3 weeks.

**~10–13 months total to v1.0** at steady pace — revised upward from the original 6–8 months.

Both completed phases overran. The causes are documented rather than smoothed over:

- **Phase 1** needed two unplanned fat releases (v0.3.1 hotfix, v0.3.2 engine reuse + share formula + figure polish) driven by real-usage findings that the plan could not have anticipated.
- **Phase 2** grew twice — to 6 commits when the √T methodology critique forced the validity framework in (Update 5), then to 7 when run-configuration serialization was adopted (Update 7). Commit 4.0 alone consumed ~2 weeks after four bugs appeared post-cutover (Update 4); commit 4.1 another ~1.5 weeks including a full design-audit cycle; and the v0.4.2 presentation sweep is unbudgeted work created by 4.1's own rescale.

The pattern is that **correctness work expands and feature work compresses**. Phases 4–6 introduce new metric families, each of which will face the same "is this measurement valid?" question that consumed Phase 2. Budget accordingly: assume every new metric costs a design-audit cycle on top of its implementation estimate.

Phase 3 weeks overlap heavily with Phase 1+2+4 due to W2 scaffolding start; calendar weeks are conservative.

---

## 20. Immediate next steps

*Rewritten 2026-05-13. Previous versions listed Phase 1 instruction batches, then the v0.4.1 ship sequence; both complete.*

**Shipped**

1. ✅ v0.4.1 published — measurement validity framework, Formula A, `min_valid_fraction` floor, schema v6. Calibration gate held at 1e-6 (51 + 89 = 140 assertions).
2. ✅ LESSONS.md L-033 through L-037 recorded.
3. ✅ Plan body realigned to shipped state (this pass) and Update 7 appended.

**In flight — v0.4.2**

4. Presentation-layer sweep. Ten output surfaces enumerated, four wrong after the v0.4.1 rescale: an HTML crash on `None` cells, superseded formula labels in three renderers, values shown for cells other surfaces mark `n/a` in three, and one **numeric** defect — the specialization z-score aggregating over excluded domains, which contaminates every other cell in the row.
5. Shared validity-filter helper, replacing what would otherwise be six copies of the same predicate.
6. `change_size` predicate fix — existence versus sufficiency (draft L-039). `normalization_effect` deprecated toward the v0.5.0 paper-tier retirement; it has been redundant with `change_size` since before v0.4.1.
7. Housekeeping from Update 6 Z.5: single-domain fallback formula, `tests/test_*.py` migration, `per_X` duplication audit.
8. Regression test rendering every surface from a `None`-bearing fixture, asserting no exception, shared formula constants, no cross-surface contradiction, and aggregation over measured cells only.

**Lab feedback (Update 5 Y.6)**

9. Phase 1 (pre-implementation): PR #17 shared as the design-review entry point, with a status comment pointing at the current doc on PR #19. Close with a superseded-by note once feedback lands.
10. Phase 2 (post-ship): demo on v0.4.1 numbers. **Must explain that `—` in the long-context column means the base could not measure that domain, not that nothing changed.** yarn scored 100 of 100 long-context probes successfully; the base scored 9. The natural misreading is the exact inverse of the truth (Update 6 Z.3).

**Next**

11. Commit 4.2 — run-config schema and emission (Update 7). Design audit first, per the pattern in §17. Does not depend on lab feedback.
12. Commit 4.3 — probe taxonomy. **Does** depend on lab feedback: the eight task types in §5.1 were derived theoretically before any external user saw the tool.
13. v0.5.0 scoping (Update 6 Z.4): v0.2.x removals, paper-tier figure retirement, `variant_only_metrics` populated with variant-vs-variant comparisons, colour separation of `variant_only` from `out_of_range`, `effective_context_length()` and the `degraded` annotation.

**Standing rules**

- Instructions for each commit are written only after the preceding one ships and its lessons are recorded. Updates 4 through 7 are all consequences of that discipline.
- Any commit touching a metric, a schema, or a cross-cutting abstraction gets a design audit before implementation (§17).
- `PHASE_PLAN_v6.md` itself belongs in `docs/internal/`. It has been untracked through seven updates, which is a single disk failure away from being a problem.


---

# UPDATE — 2026-04-25: Literature validation of Llama-2 case study findings

> **Status**: Appended after v6 was locked. Does not modify the plan above.
> **Purpose**: Document the literature-grounded support (and deliberate non-support) for the three core findings produced by lmdiff on the Llama-2 4-variant case. This material should land in `docs/concepts/llama2-case-study.md` (Phase 3 commit 3.3) and the positioning page `docs/concepts/positioning.md` (Phase 3 commit 3.7).

## U.1 Probe set identification (precise)

The Llama-2 4-variant case used the lm-eval-harness probe set:

```
lm_eval:hellaswag+arc_challenge+gsm8k+mmlu_college_computer_science+longbench_2wikimqa
```

100 probes per task, mapped to lmdiff domains:

| Domain | Probe source | Avg. tokens |
|---|---|---|
| commonsense | HellaSwag | ~52 |
| **reasoning** | **ARC-Challenge** | ~35 |
| math | GSM8K | ~58 |
| code | MMLU college_computer_science | ~43 |
| long-context | LongBench 2wikimqa | ~90+ |

ARC-Challenge is the AI2 Reasoning Challenge "Challenge" subset (Clark et al., 2018) — 4-option science MCQ, the difficult subset filtered to questions retrieval/co-occurrence baselines fail. Evaluated via lm-eval-harness's standard log-likelihood ranking.

Variants compared to base `meta-llama/Llama-2-7b-hf`:

| Variant | Source | Citation |
|---|---|---|
| **yarn** | `NousResearch/Yarn-Llama-2-7b-128k` | Peng et al. 2023 (arxiv 2309.00071), trained on PG19 |
| **long** | `togethercomputer/LLaMA-2-7B-32K` | Together AI blog 2023; uses Position Interpolation (Chen et al. 2023, arxiv 2306.15595); optimizer hyperparameters not publicly disclosed |
| **code** | `meta-llama/CodeLlama-7b-hf` | Rozière et al. 2024 (arxiv 2308.12950) |
| **math** | Llemma 7B + MetaMath SFT pipeline | Azerbayev et al. ICLR 2024 (arxiv 2310.10631) for Llemma; Yu et al. 2023 (arxiv 2309.12284) for MetaMath SFT |

## U.2 Empirical findings to validate

```
Per-domain drift magnitude (per √token):
  total           per-domain peak
  long: 0.0865    reasoning (ARC) = 0.3355 (66% of long's budget)
  yarn: 0.0795    commonsense (HellaSwag) = 0.0931 (51% of yarn's budget)
  math: 0.0545    math (GSM8K) = 0.1025 (35%)
  code: 0.0360    code (MMLU-CS) = 0.0626 (32%)
```

Three claims drove the validation effort:

- **U.2.a** — drift size ranking (`long > yarn > math > code`) reflects something interpretable about training
- **U.2.b** — `long → reasoning` peak has mechanism-level support
- **U.2.c** — `yarn → commonsense` peak has mechanism-level support

## U.3 Finding U.2.a — drift size cannot be predicted from training recipe

### U.3.1 Tokens-trained does not explain it

Continued-pretraining / SFT token volumes:

| Variant | Tokens trained on top of base |
|---|---|
| code (CodeLlama) | 500B |
| math (Llemma + MetaMath) | 200B + ~1B SFT |
| yarn | ~1.6B |
| long (Together) | ~1.5B |

Predicted drift order from token volume: `code > math >> yarn ≈ long`.
Observed: `long > yarn > math > code` — **reversed**.

### U.3.2 Effective optimizer path (lr × steps) does not explain it either

After correcting math's training recipe to the actual Llemma + MetaMath pipeline (lr=2e-5, 9258 SFT steps, on top of Llemma 7B which itself was 42K steps at lr=1e-4 from CodeLlama):

Cumulative `lr × steps` from base Llama-2-7b:

| Variant | Path | Cumulative lr × steps |
|---|---|---|
| math | Llama-2 → CodeLlama → Llemma → MetaMath SFT | ~42 |
| code | Llama-2 → CodeLlama | ~38 |
| long | Llama-2 → continued PT (1.5B tokens) | ~0.014 |
| yarn | Llama-2 → PG19 fine-tune (1.6B tokens) | ~0.012 |

Note: long's optimizer hyperparameters are not publicly disclosed by Together AI; the ~0.014 figure is a reasonable estimate based on community Llama-2 fine-tune recipes of that period (lr ~5e-6 to 2e-5, ~200–400 steps with batch ~512K–2M tokens). Citing this number requires the caveat "estimated from public dataset/token figures; optimizer parameters not officially published."

Predicted drift order from cumulative optimizer path: `math > code >> long ≈ yarn`.
Observed: `long > yarn > math > code` — **also reversed**.

### U.3.3 Why neither proxy works

The pattern `code` and `math` (highest training "intensity" by every proxy) showing **lowest** drift is structural, not noise. Two explanations:

1. **Training distribution overlap with probe distribution dominates over training intensity.** CodeLlama's 500B tokens were 85% GitHub code. From the perspective of NL probes (HellaSwag, ARC, MMLU, GSM8K), code-domain weight updates are largely orthogonal — the model on NL prompts walks back to nearly its base weight subspace. Math (Llemma + MetaMath) is two-stage with cancelling shifts: code training pulls weights toward a code subspace; MetaMath SFT pulls them back toward NL+math reasoning. From NL probe view, the two shifts partially cancel.

2. **Long-context fine-tuning preferentially modifies global parameters.** This is the focus of U.4 below — the literature establishes that long-context training requires changes to embedding + normalization layers, not just attention, so the per-probe distributional impact is broader than its small training budget would suggest.

### U.3.4 The actionable conclusion

**No single training-recipe proxy predicts behavioral drift ranking.** Token count is wrong. Cumulative optimizer path is wrong. A true predictor would need to capture the interaction between training-data distribution and probe-data distribution — a quantity that requires actually running the comparison.

This is lmdiff's positioning argument: training-recipe transparency is insufficient to predict downstream behavioral change. The behavioral distance must be measured directly. lmdiff is the tool that does this measurement and reports it in geometry-aware form (drift, share, direction) rather than scalar accuracy.

If recipe metadata were a sufficient predictor, lmdiff would be redundant. **It isn't, so it isn't.**

## U.4 Finding U.2.b — `long → reasoning` peak: mechanism-level support

### U.4.1 The HELM cross-check

The Together AI HELM v1.0 evaluation of `LLaMA-2-7B-32K` reports:

| Benchmark | base | long | Δ |
|---|---|---|---|
| MMLU | 0.435 | 0.435 | 0.000 |
| HellaSwag (EM) | 0.759 | 0.748 | −0.011 |
| OpenbookQA | 0.570 | 0.533 | −0.037 |
| AVG (16 HELM core scenarios) | 0.489 | 0.522 | +0.033 |

On reasoning-class benchmarks, **accuracy is essentially preserved**. Yet lmdiff measures ARC-Challenge drift = 0.3355 — long's largest single-domain drift, consuming 66% of its total drift budget.

These are **not contradictory**:
- HELM accuracy is **top-1 ranking** under MCQ — invariant to ranking-preserving distribution shifts
- lmdiff BD is **token-level cross-entropy difference** — sensitive to the full output distribution

A model can re-route its reasoning trajectory without changing which option scores highest. Accuracy preservation does not imply distribution preservation.

### U.4.2 Why long-context training has global impact: four mechanism-level papers

Long-context fine-tuning was historically described as "RoPE frequency adjustment" — a narrow, position-encoding-specific modification. The interpretability/mechanism literature paints a different picture: **long-context training affects parameters that influence every forward pass, regardless of prompt length**.

**Evidence 1: LongLoRA (Chen et al., ICLR 2024, arxiv 2309.12307)** — directly observes that LoRA on attention weights alone is insufficient for long-context adaptation:

> "LoRA for context extension works well **under the premise of trainable embedding and normalization**." (Table 2)

The paper demonstrates that even at LoRA rank 256, attention-only adaptation has a 3% perplexity gap to full fine-tune. The gap closes only when **embedding and normalization layers are made trainable**. Both layers influence every token in every prompt.

**Evidence 2: BFloat16 + RoPE breakdown (Yu et al. 2024, arxiv 2411.13476)** — finds that BF16 precision (standard in long-context training) breaks RoPE's relative-position property, particularly for the first token:

> "the combination of Rotary Position Embedding (RoPE) and BFloat16 precision breaks the relative positional encoding properties of RoPE"

This forces long-context training to globally re-calibrate position handling — not only at long range. The first-token degradation in particular impacts every prompt regardless of length.

**Evidence 3: LongRoPE2 critical-dimension analysis (Wang et al. 2025, arxiv 2502.20082)** — uses evolutionary search to identify "true critical RoPE dimensions" with disproportionate impact on long-context behavior:

> "leverages evolutionary search to identify the true critical RoPE dimensions and optimal rescaling factors"

These critical dimensions are not uniformly distributed across model layers; long-context training preferentially perturbs them, propagating effects through the model's depth even on short inputs.

**Evidence 4: Pause-Tuning attention redistribution (Yang et al. 2025, arxiv 2502.20405)** — confirms attention pattern shift in long-context fine-tuning is **global**, not localized to long-distance:

> "pause tokens induce **meaningful shifts in attention distribution**"

The U-shape primacy/recency bias change documented in Liu et al. 2024 ("Lost in the Middle") happens at all sequence lengths — short prompts also see attention redistribution.

### U.4.3 The synthesis

The four papers provide a converging argument: **long-context training's mechanism is not "RoPE frequency adjustment" but global parameter redistribution**. Embedding layers, normalization layers, critical RoPE dimensions, and attention pattern allocation all shift. These changes affect every forward pass on every prompt, including short prompts like ARC-Challenge (35 token average).

This explains lmdiff's counterintuitive observation:

> A variant explicitly trained for long context (`long`) shows its largest behavioral drift on the **shortest** probe set (35-token ARC). Without mechanism-level grounding, this looks like measurement noise. With mechanism-level grounding, it is the predicted result of long-context training affecting global parameters.

The lmdiff finding and the mechanism literature are **independent evidence streams converging on the same observation**.

### U.4.4 Why this finding has high positioning value

It demonstrates lmdiff providing diagnostic information that:
1. **Is not visible in accuracy** (HELM shows preservation)
2. **Is independently supported by mechanism research** (4 papers)
3. **Is counterintuitive without that research** (would be dismissed as noise)

This is the "lmdiff sees what accuracy can't" argument made concrete.

## U.5 Finding U.2.c — `yarn → commonsense` peak: literature-only-partial support

### U.5.1 The apparent contradiction

YaRN paper (Peng et al. 2023, arxiv 2309.00071) Table 2 states:

> "On the HuggingFace Open LLM Leaderboard (ARC, **HellaSwag**, MMLU, TruthfulQA), average degradation is **well below 1 point**" (vs. base Llama-2-7b)

YaRN's design philosophy is "modify only RoPE frequency calculation, do not touch model weights." From this, one would predict near-zero behavioral change on commonsense (HellaSwag).

But lmdiff observes drift = 0.0931 on HellaSwag for yarn — its largest single-domain drift, consuming 51% of yarn's budget. This is in apparent tension with YaRN's stated design.

### U.5.2 The reconciliation

Two reasons the apparent tension dissolves on close reading:

**Reason 1: YaRN does include 400+200 steps of fine-tuning on PG19.** Despite the stated design intent of "modify RoPE only", the actual training procedure does perform weight updates: 400 steps at lr=2e-5, batch 64, sequence 64K tokens, on PG19 alone (Peng et al., Section 4). The "RoPE-only" framing applies to the **method**, not the **training procedure** that brings about the working model.

**Reason 2: PG19 is structurally similar to HellaSwag.** PG19 is narrative fiction (Project Gutenberg books pre-1919). HellaSwag is "what happens next" multi-choice from narrative-style sources (stories, video transcripts, WikiHow). The two share narrative continuation as their core distributional pattern.

When yarn fine-tunes 1.6B PG19 tokens at lr=2e-5, the weight updates **preferentially strengthen the narrative-continuation directions in weight space**. HellaSwag probes these exact directions. The result is a measurable distribution shift that:
- Preserves top-1 ranking (HellaSwag accuracy 0.759 → 0.748, ~1pt drop)
- Does not preserve full distribution (lmdiff drift 0.0931)

### U.5.3 Why this is lmdiff-only

Unlike U.2.b, no mechanism paper directly predicted yarn → commonsense drift. The closest evidence is indirect:
- YaRN paper's own Table 2 shows the ~1pt HellaSwag drop, which is consistent with what we observe in accuracy
- But no paper measures the **distributional** shift, which is much larger than 1pt of accuracy suggests

This is a **lmdiff-original observation that survives literature cross-reference but is not pre-existing in the literature**. It is weaker support than U.2.b — but still supportable as: "the YaRN training procedure includes PG19 fine-tuning, which structurally overlaps with HellaSwag's narrative distribution; lmdiff observes the distributional consequence."

### U.5.4 Honest framing

Within the case study writeup (`docs/concepts/llama2-case-study.md`), this finding should be framed as:

> "lmdiff observes a 0.0931 drift on HellaSwag for the yarn variant, which is its largest single-domain shift. The YaRN paper's own benchmarks show only a ~1pt HellaSwag accuracy drop, consistent with this drift being **distribution-shape change rather than ranking change**. The PG19 training corpus shares narrative-continuation structure with HellaSwag, providing a plausible mechanism. To our knowledge, this distributional consequence has not been previously reported."

Marking this as "to our knowledge, not previously reported" is honest — it's an observation made possible by lmdiff that no benchmark or analysis paper had reason to surface.

## U.6 Implications for v6 Phase 1 deliverables

This update has three concrete implications for what Phase 1 (v0.3.0) ships:

1. **Findings system addition**: in addition to the 8 finding types listed in §4.5, consider adding `MechanismCorrespondenceFinding` (Phase 4 or later, opt-in) — when user provides training metadata for a variant and the observed drift pattern correlates with known long-context / instruction-tuning / quantization mechanism signatures, lmdiff surfaces a brief reference. This is post-v0.3.0; not blocking.

2. **`Config.metadata` extension**: leave a slot for `training_recipe_summary: Optional[str]` so users can record recipe-level information that lmdiff will not interpret automatically but will display in reports. Phase 1 commit 1.2 should include this slot.

3. **Llama-2 case study positioning**: the case study tutorial (Phase 3 commit 3.3) should structure the presentation as **two findings with mechanism support + one finding that survives literature cross-reference** — not as "three findings of equal evidential status." This is more accurate and more credible.

## U.7 Bibliography

In citation order from this update:

- Clark, P., et al. (2018). Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arxiv 1803.05457
- Rozière, B., et al. (2024). Code Llama: Open Foundation Models for Code. arxiv 2308.12950
- Peng, B., et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models. arxiv 2309.00071
- Chen, S., et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation. arxiv 2306.15595
- Together AI (2023). Preparing for the era of 32K context: Early learnings and explorations. https://www.together.ai/blog/llama-2-7b-32k
- Azerbayev, Z., et al. (2024). Llemma: An Open Language Model For Mathematics. ICLR 2024, arxiv 2310.10631
- Yu, L., et al. (2023). MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models. arxiv 2309.12284
- Chen, Y., et al. (2024). LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models. ICLR 2024, arxiv 2309.12307
- Yu, J., et al. (2024). When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training. arxiv 2411.13476
- Wang, S., et al. (2025). LongRoPE2: Near-Lossless LLM Context Window Scaling. arxiv 2502.20082
- Yang, T., et al. (2025). Pause-Tuning for Long-Context Comprehension. arxiv 2502.20405
- Liu, N., et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. arxiv 2307.03172


---

# UPDATE 2 — 2026-04-25: Master bibliography

> **Status**: Appended after v6 was locked. Does not modify the plan above.
> **Purpose**: Single, organized bibliography that consolidates citations from all sources used to build v6 (the original `modeldiff_vision.md`, the metric specification PDF, the literature validation Update 1, and the design rationale for individual phases). Each entry maps explicitly to the v6 sections that depend on it.
> **Audience**: Paper writers, doc writers (Phase 3 commits 3.2 and 3.7 in particular), and anyone evaluating where a claim in the plan comes from.
>
> Originally `modeldiff_vision.md` had a more complete reference list than what I carried into the v6 main text. This update fixes that omission. Future updates to v6 should add to this bibliography rather than create separate citation pools.

---

## B.1 Reading guide

The bibliography is organized by **role in v6**, not by alphabetical author. Each entry has:

- **Citation** in standard form (author, year, venue, arxiv id when available)
- **Used in** — pointer to the v6 sections / phases that depend on it
- **Why it's cited** — one sentence on the specific claim or design choice it supports

Some references are cited in multiple places. Those are listed once and the "Used in" field enumerates the locations.

---

## B.2 Behavioral distance foundations (cross-model KL / log-likelihood)

These references ground the BD metric (the core distance lmdiff computes) in established theory. Without them, BD looks like an ad-hoc choice; with them, BD is a known estimator.

### B.2.1 Takase et al. (2026)

**Citation**: Takase, S., et al. (2026). [Conditional log-likelihood representations for inter-model KL approximation.] arxiv (forthcoming/recent).

**Used in**:
- §1 Design principles (claim that BD is a principled estimator, not invented)
- §2 Phase summary (positions BD as established)
- §15 Metrics roadmap → BehavioralDistance entry
- §6 Phase 3 docs commit 3.7 — positioning page
- Update 1 §U.4 (mechanism evidence framing)

**Why cited**: Proves that log-likelihood vectors over a prompt set can approximate inter-model KL. This validates the lmdiff BD metric as theoretically grounded, not heuristic. Appears in vision doc §2.2 and §6.1.

### B.2.2 Amini, Vieira & Cotterell (2025)

**Citation**: Amini, A., Vieira, T., & Cotterell, R. (2025). [Variance and intractability in inter-LM KL estimation.] arxiv (recent).

**Used in**:
- §1 Design principles
- §15 BD entry
- Phase 4 commit 4.4 (semantic_entropy_diff variance handling)
- §6 Phase 3 commit 3.7 — positioning

**Why cited**: Shows that exact KL between arbitrary LMs is intractable and standard Monte Carlo has high variance. Motivates the design choice of per-token normalization and multi-sample averaging in lmdiff. Appears in vision doc §2.2 and §6.1.

### B.2.3 Binoculars — Hans et al. (2024)

**Citation**: Hans, A., et al. (2024). Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text. arxiv 2401.12070.

**Used in**:
- §6 Phase 3 commit 3.7 — positioning, "what we share with detection literature"
- §15 BD discussion of related distance metrics

**Why cited**: Uses cross-perplexity ratio between two models for AI text detection — same family of cross-model scoring as lmdiff, but narrow detection focus. Demonstrates that the cross-model perplexity comparison technique is well-established. Helps position lmdiff as "general comparison" not "detection."

### B.2.4 DetectGPT — Mitchell et al. (ICLR 2023)

**Citation**: Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., & Finn, C. (2023). DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature. ICLR 2023, arxiv 2301.11305.

**Used in**:
- §6 Phase 3 commit 3.7 — positioning
- §15 BD discussion

**Why cited**: Probability curvature under perturbation as cross-model scoring. Same family as Binoculars; narrow application but establishes the technique's validity.

### B.2.5 MAUVE — Pillutla et al. (NeurIPS 2021)

**Citation**: Pillutla, K., Swayamdipta, S., Zellers, R., Thickstun, J., Welleck, S., Choi, Y., & Harchaoui, Z. (2021). MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers. NeurIPS 2021, arxiv 2102.01454.

**Used in**:
- §9 Phase 6 commit 6.4–6.5 (cloud distance metrics — MMD/Energy/C2ST)
- §6 Phase 3 commit 3.7 — positioning ("what we share with sample-based distributional comparison")
- Update 1 §U.6 (recipe-blind comparison via samples)

**Why cited**: Sample-based distributional comparison using divergence frontiers — the closest precedent for lmdiff's Phase 6 cloud metrics. Black-box compatible but lacks per-prompt or per-domain diagnostics that lmdiff adds.

### B.2.6 MiniLLM — Gu et al. (ICLR 2024)

**Citation**: Gu, Y., Dong, L., Wei, F., & Huang, M. (2024). MiniLLM: Knowledge Distillation of Large Language Models. ICLR 2024, arxiv 2306.08543.

**Used in**:
- §15 BD discussion (forward vs reverse KL → mode covering vs mode collapse)
- §6 Phase 3 commit 3.7 — positioning, asymmetry score interpretation
- Phase 4 commit 4.3 (calibration metrics — distinct directions of "narrowing")

**Why cited**: Demonstrates the practical significance of KL directionality — Forward KL gives mode covering, Reverse KL gives mode collapse. lmdiff exposes this as the asymmetry score in BD. Vision doc §2.2 directly cites this for the asymmetry interpretation.

### B.2.7 Cross-Tokenizer Likelihood Scoring (2026)

**Citation**: [Author et al.] (2026). Cross-tokenizer likelihood for fair LM comparison. arxiv (recent — exact reference deferred until Phase 1 commit 1.4 implements BPB and confirms the citation).

**Used in**:
- Phase 1 commit 1.4 (GeoResult schema v5; BPB normalization)
- §15 BD discussion (`tokenizer_id` mechanism)
- §6 Phase 3 commit 3.7 — positioning

**Why cited**: Solves a critical practical blocker for cross-architecture BD computation. Justifies the lmdiff design where same-tokenizer pairs use per-token CE and cross-tokenizer pairs fall back to bytes-per-byte (BPB).

### B.2.8 RUT — Zhu et al. (2026)

**Citation**: Zhu, X., et al. (2026). [RUT: Black-box statistical testing for model equality.] arxiv (recent).

**Used in**:
- §6 Phase 3 commit 3.7 — positioning
- §9 Phase 6 cloud metrics (statistical-test framing for C2ST)
- §15 BD discussion (alternative for API-only)

**Why cited**: Black-box statistical testing of "are models equal?" — sister direction to cloud metrics in Phase 6. Answers existence question (same/different) but not "how/where they differ", which is lmdiff's value-add.

---

## B.3 Change geometry foundations (Configuration as vector / direction)

These references ground the Change Geometry framework — treating model modifications as vectors with magnitude and direction.

### B.3.1 Task Arithmetic — Ilharco et al. (ICLR 2023)

**Citation**: Ilharco, G., Ribeiro, M. T., Wortsman, M., Schmidt, L., Hajishirzi, H., & Farhadi, A. (2023). Editing Models with Task Arithmetic. ICLR 2023, arxiv 2212.04089.

**Used in**:
- §1 Design principle 3 (Configuration is the unit of comparison)
- §6 Phase 3 commit 3.7 — positioning (lmdiff is **behavior-space generalization** of Task Arithmetic)
- §15 Direction (cosine) entry
- §16 Configuration roadmap (vector-of-modifications mental model)

**Why cited**: First work to treat model changes as vectors (task vector τ = θ_finetuned − θ_pretrained). Lmdiff generalizes this from weight-space to behavior-space — vision doc §2.1 explicitly frames lmdiff this way. Differentiator: Task Arithmetic requires same architecture/init; lmdiff handles any configuration type.

---

## B.4 Representation similarity (CKA, PWCCA, SVCCA)

Phase 5 relies on these. Each metric has an originating paper plus follow-ups documenting strengths and weaknesses.

### B.4.1 CKA — Kornblith et al. (ICML 2019)

**Citation**: Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. ICML 2019, arxiv 1905.00414.

**Used in**:
- Phase 5 commit 5.2 (CKA implementation + reference test)
- §8 Phase 5 metric table (CKA entry)
- §15 Representation metrics
- §6 Phase 3 docs commit 3.5 (metric reference page for CKA)

**Why cited**: Original CKA paper. Reference test on toy data should reproduce numbers from this paper within 1% tolerance. The four invariance properties (orthogonal, isotropic scaling) come from this paper.

### B.4.2 CKA critique — Davari et al. (2022)

**Citation**: Davari, M., Asadi, N., Mudur, S., Aljundi, R., & Belilovsky, E. (2022). Reliability of CKA as a Similarity Measure in Deep Learning. arxiv 2210.16156.

**Used in**:
- Phase 5 commit 5.2 (must document CKA caveats)
- §8 Phase 5 metric table (CKA "do not use alone" caveat)
- §6 Phase 3 metric reference page for CKA (caveat section)
- §12.10 calibration case (when CKA could mislead)

**Why cited**: Shows CKA is sensitive to simple translations that preserve functional behavior. This is why lmdiff includes PWCCA and SVCCA as complements rather than CKA alone, and why the CKA reference page must include this caveat.

### B.4.3 PWCCA — Morcos, Raghu & Bengio (NeurIPS 2018)

**Citation**: Morcos, A., Raghu, M., & Bengio, S. (2018). Insights on Representational Similarity in Neural Networks with Canonical Correlation. NeurIPS 2018, arxiv 1806.05759.

**Used in**:
- Phase 5 commit 5.3 (PWCCA implementation + reference test)
- §15 Representation metrics
- §6 Phase 3 metric reference page for PWCCA

**Why cited**: Original PWCCA paper. Projection-weighted CCA is one of three complementary representation metrics in Phase 5. Reference test reproduces numbers from this paper within 1%.

### B.4.4 SVCCA — Raghu et al. (NeurIPS 2017)

**Citation**: Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017). SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability. NeurIPS 2017, arxiv 1706.05806.

**Used in**:
- Phase 5 commit 5.3 (SVCCA implementation + reference test)
- §15 Representation metrics
- §6 Phase 3 metric reference page for SVCCA

**Why cited**: Original SVCCA paper. SVD-truncated CCA is the third in CKA/PWCCA/SVCCA triplet. Reference test reproduces numbers within 1%.

### B.4.5 Latent Functional Maps (NeurIPS 2024)

**Citation**: Cannistraci, I., et al. (2024). Latent Functional Maps for Representation Stitching. NeurIPS 2024 (or related; exact author list to confirm during Phase 6 implementation).

**Used in**:
- Phase 6 commit 6.3 (`lmdiff.contrib.latent_functional_maps`)
- §9 Phase 6 experimental metrics table
- §15 Experimental metrics

**Why cited**: Spectral alignment of representation graphs — claimed to be more stable than CKA for stitching tasks but battle-tested for shorter time. Justifies the experimental/contrib status (not stable in v1.0).

---

## B.5 Trajectory & lens metrics

### B.5.1 Tuned Lens — Belrose et al. (2023)

**Citation**: Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., & Steinhardt, J. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. arxiv 2303.08112.

**Used in**:
- Phase 6 commit 6.1 (`tuned_lens_trajectory_diff`)
- §9 Phase 6 stable metrics table
- §15 Trajectory metrics
- §6 Phase 3 commit 3.7 — positioning ("Tuned Lens is single-model; lmdiff applies it cross-model")

**Why cited**: Original Tuned Lens. Trains an affine probe per layer to project hidden state to vocabulary. lmdiff's Tuned Lens Diff applies this per-layer to two models and compares trajectories. Critical: Belrose's setup is single-model analysis; lmdiff makes it a comparison metric, which is novel.

### B.5.2 Logit Lens — nostalgebraist (2020)

**Citation**: nostalgebraist (2020). Interpreting GPT: The Logit Lens. LessWrong / AI Alignment Forum (informal post; widely referenced).

**Used in**:
- Phase 6 commit 6.2 (`lmdiff.contrib.logit_lens_diff`)
- §9 Phase 6 experimental metrics
- §15 Experimental metrics
- §6 Phase 3 metric reference page (with caveats from later papers)

**Why cited**: Original logit lens technique. Used as the lightweight precursor to Tuned Lens. lmdiff includes logit lens diff as experimental/contrib because of known reliability issues on OPT/BLOOM (documented in Belrose et al. 2023).

---

## B.6 Calibration & uncertainty metrics

### B.6.1 Semantic Entropy — Kuhn et al. (Nature 2024)

**Citation**: Kuhn, L., Gal, Y., & Farquhar, S. (2024). Detecting hallucinations in large language models using semantic entropy. Nature 630, 625–630.

**Used in**:
- Phase 4 commit 4.4 (semantic_entropy_diff)
- §7 Phase 4 metric table
- §15 Output/behavioral metrics
- §6 Phase 3 metric reference page for semantic entropy

**Why cited**: Defines semantic entropy — sample k outputs, cluster by meaning, compute entropy over clusters. lmdiff's Semantic Entropy Diff applies this to two models. The Nature 2024 venue makes this a high-credibility primary source.

### B.6.2 LM-Polygraph (TACL 2025)

**Citation**: Vashurin, R., et al. (2025). LM-Polygraph: Uncertainty Estimation for Language Models. TACL 2025, arxiv 2406.15627 (or similar).

**Used in**:
- §6 Phase 3 commit 3.7 — positioning
- Phase 4 commit 4.3 (calibration metrics — explicit comparison to single-model UQ)

**Why cited**: Single-model uncertainty quantification. Positions lmdiff's calibration metrics as the comparison-aware version of UQ — different scope (cross-config) than LM-Polygraph (single-config).

---

## B.7 Mechanism evidence — long-context training (Update 1 §U.4)

These four papers support the finding that long-context training has global parameter impact, not just RoPE adjustment.

### B.7.1 LongLoRA — Chen et al. (ICLR 2024)

**Citation**: Chen, Y., Qian, S., Tang, H., Lai, X., Liu, Z., Han, S., & Jia, J. (2024). LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models. ICLR 2024, arxiv 2309.12307.

**Used in**:
- Update 1 §U.4.2 — primary evidence that long-context training requires embedding + normalization layer training
- §6 Phase 3 docs commit 3.3 (Llama-2 case study tutorial)
- §16 Configuration roadmap (justifies why drift can be global despite small training budget)

**Why cited**: Table 2 directly shows that LoRA on attention alone is insufficient for long-context — must also train embedding and normalization. This is the cleanest mechanism evidence that long-context training affects parameters that influence every forward pass.

### B.7.2 BFloat16 RoPE breakdown — Yu et al. (2024)

**Citation**: Yu, J., et al. (2024). When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training. arxiv 2411.13476.

**Used in**:
- Update 1 §U.4.2 — global position recalibration evidence
- §6 Phase 3 docs commit 3.3

**Why cited**: Shows BF16 precision (standard in long-context training) breaks RoPE's relative-position property, particularly first-token. Forces global recalibration during training. Explains why even short prompts see drift.

### B.7.3 LongRoPE2 — Wang et al. (2025)

**Citation**: Wang, S., et al. (2025). LongRoPE2: Near-Lossless LLM Context Window Scaling. arxiv 2502.20082.

**Used in**:
- Update 1 §U.4.2 — critical RoPE dimensions evidence
- §6 Phase 3 docs commit 3.3

**Why cited**: Identifies "true critical RoPE dimensions" using evolutionary search; shows long-context training perturbs these disproportionately. Supports finding that drift is concentrated, not uniform.

### B.7.4 Pause-Tuning — Yang et al. (2025)

**Citation**: Yang, T., et al. (2025). Pause-Tuning for Long-Context Comprehension: A Lightweight Approach to LLM Attention Recalibration. arxiv 2502.20405.

**Used in**:
- Update 1 §U.4.2 — global attention redistribution evidence
- §6 Phase 3 docs commit 3.3

**Why cited**: Shows attention pattern shift in long-context fine-tuning is global, not localized to long-distance. Including primacy/recency redistribution at all sequence lengths. Final piece of the four-paper convergent argument.

### B.7.5 Lost in the Middle — Liu et al. (2024)

**Citation**: Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). Lost in the Middle: How Language Models Use Long Contexts. TACL 2024, arxiv 2307.03172.

**Used in**:
- Update 1 §U.4.2 — establishes the U-shape primacy/recency bias that Pause-Tuning addresses
- §6 Phase 3 docs commit 3.3

**Why cited**: Documents the U-shape attention bias that motivates Pause-Tuning's approach. Strengthens the chain of evidence that long-context training has global attention effects, not just long-range effects.

---

## B.8 Long-context training mechanism context (background, not primary evidence)

### B.8.1 Position Interpolation — Chen et al. (2023)

**Citation**: Chen, S., Wong, S., Chen, L., & Tian, Y. (2023). Extending Context Window of Large Language Models via Positional Interpolation. arxiv 2306.15595.

**Used in**:
- Update 1 §U.1 — long variant source method citation
- §6 Phase 3 commit 3.3 (Llama-2 case study — long variant method description)

**Why cited**: Original Position Interpolation method that the Together AI long variant uses. Cites the method, not its specific Together AI implementation (which lacks public optimizer details).

### B.8.2 YaRN — Peng et al. (2023)

**Citation**: Peng, B., Quesnelle, J., Fan, H., & Shippole, E. (2023). YaRN: Efficient Context Window Extension of Large Language Models. arxiv 2309.00071.

**Used in**:
- Update 1 §U.1 — yarn variant source citation
- Update 1 §U.5 — yarn → commonsense finding analysis
- §6 Phase 3 commit 3.3

**Why cited**: Source paper for the yarn variant. Importantly, the paper's Table 2 reports near-zero accuracy degradation on HellaSwag, which lmdiff's Update 1 §U.5 cross-references against the observed BD drift to demonstrate "accuracy preservation ≠ distribution preservation."

### B.8.3 Together AI LLaMA-2-7B-32K (2023)

**Citation**: Together AI (2023). Preparing for the era of 32K context: Early learnings and explorations. https://www.together.ai/blog/llama-2-7b-32k

**Used in**:
- Update 1 §U.1 — long variant source (model checkpoint)
- Update 1 §U.3.2 — caveat that optimizer hyperparameters not publicly disclosed
- §6 Phase 3 commit 3.3

**Why cited**: The actual long variant (`togethercomputer/LLaMA-2-7B-32K`) source. Needs to be cited with the explicit caveat that optimizer hyperparameters are not in the blog post and so cannot be verified.

### B.8.4 Llama 2 Long — Xiong et al. (2023)

**Citation**: Xiong, W., et al. (2023). Effective Long-Context Scaling of Foundation Models. arxiv 2309.16039 (Meta's own long-context Llama-2 paper).

**Used in**:
- Update 1 §U.3.2 — context for what Together's recipe approximated
- §6 Phase 3 commit 3.3 (background reading for case study)

**Why cited**: Meta's own long-context recipe (400B tokens). Comparison anchor: Together used 1.5B tokens versus Meta's 400B; helps explain Together's smaller training budget.

---

## B.9 Variant source attributions

### B.9.1 CodeLlama — Rozière et al. (2024)

**Citation**: Rozière, B., et al. (2024). Code Llama: Open Foundation Models for Code. Meta AI tech report, arxiv 2308.12950.

**Used in**:
- Update 1 §U.1 — code variant source
- Update 1 §U.3.2 — code variant training recipe (lr=3e-4, 500B tokens)
- §6 Phase 3 commit 3.3

**Why cited**: Source paper for the code variant. Provides the optimizer recipe that drives Update 1's "training intensity" analysis showing code has highest training tokens but lowest drift.

### B.9.2 Llemma — Azerbayev et al. (ICLR 2024)

**Citation**: Azerbayev, Z., et al. (2024). Llemma: An Open Language Model For Mathematics. ICLR 2024, arxiv 2310.10631.

**Used in**:
- Update 1 §U.1 — math variant pipeline (intermediate stage)
- Update 1 §U.3.2 — math variant cumulative training path

**Why cited**: Llemma 7B is the intermediate model in math variant's pipeline (Llama-2 → CodeLlama → Llemma → MetaMath SFT). Without this citation, the math variant's true training history is incomplete.

### B.9.3 MetaMath — Yu et al. (2023)

**Citation**: Yu, L., et al. (2023). MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models. arxiv 2309.12284.

**Used in**:
- Update 1 §U.1 — math variant SFT stage
- Update 1 §U.3.1 — math variant token-volume figure

**Why cited**: Final SFT stage in math variant's pipeline. Provides the ~1B token SFT recipe (lr=2e-5, 9258 steps from the user-provided training script).

### B.9.4 ARC-Challenge — Clark et al. (2018)

**Citation**: Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., & Tafjord, O. (2018). Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arxiv 1803.05457.

**Used in**:
- Update 1 §U.1 — reasoning probe set is ARC-Challenge
- §6 Phase 3 commit 3.3 (case study probe set description)

**Why cited**: Source of the reasoning probes. Important because v6's narrative often refers to "reasoning" without specifying ARC-Challenge specifically; this citation grounds the term.

---

## B.10 Attention head & feature-level analysis (deferred to v2.0)

These are listed for completeness of design intent — they support metrics that are reserved-but-not-implemented in v1.0.

### B.10.1 Voita et al. (ACL 2019)

**Citation**: Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned. ACL 2019, arxiv 1905.09418.

**Used in**:
- §15 Deferred to v2.0 (Attention Head Functional Role Drift)
- §6 Phase 3 commit 3.7 — positioning, "what's deferred"

**Why cited**: Defines attention head role taxonomy (positional, syntactic, rare-token, etc.) that would be needed to implement Attention Head Functional Role Drift. The cite preserves the design intent for v2.0.

### B.10.2 Paulo & Belrose (2025)

**Citation**: Paulo, G., & Belrose, N. (2025). [Variability of SAE features across random seeds.] arxiv (recent).

**Used in**:
- §15 Deferred to v2.0 (SAE Feature Overlap)
- §6 Phase 3 commit 3.7 — positioning, "why we don't ship SAE in v1.0"

**Why cited**: Shows different random seeds yield different SAE feature sets, requiring careful matching. This is the reason SAE Feature Overlap is deferred — the matching problem isn't solved yet.

---

## B.11 Methods used as design references (not cited in main flow)

### B.11.1 lm-evaluation-harness — Gao et al.

**Citation**: Gao, L., et al. (2023+). The Language Model Evaluation Harness. github.com/EleutherAI/lm-evaluation-harness, software with periodic releases.

**Used in**:
- Phase 1 commits 1.1–1.4 (probe set integration)
- Phase 2 commit 2.3 (probe set sources)
- §13 CLI summary example (probe set name `lm_eval:hellaswag+arc_challenge+...`)

**Why cited**: lmdiff's probe loading is built on lm-evaluation-harness task definitions. Citing the tool acknowledges the engineering dependency.

### B.11.2 HELM — Liang et al. (2022)

**Citation**: Liang, P., et al. (2022). Holistic Evaluation of Language Models. arxiv 2211.09110.

**Used in**:
- Update 1 §U.4.1 (HELM cross-check on long variant)
- §6 Phase 3 commit 3.7 — positioning, "what HELM does and doesn't cover"

**Why cited**: HELM provides the accuracy data that Update 1 uses to demonstrate "accuracy preserved but BD large." Without HELM as comparison anchor, the lmdiff finding lacks the benchmark-side evidence that makes it striking.

### B.11.3 PG19 — Rae et al. (ICLR 2020)

**Citation**: Rae, J. W., Potapenko, A., Jayakumar, S. M., Hillier, C., & Lillicrap, T. P. (2020). Compressive Transformers for Long-Range Sequence Modelling. ICLR 2020, arxiv 1911.05507.

**Used in**:
- Update 1 §U.5 (yarn variant training corpus)

**Why cited**: PG19 is the corpus yarn variant fine-tunes on. Critical to the yarn → commonsense argument because PG19's narrative structure is what overlaps with HellaSwag's narrative MCQ format.

### B.11.4 RedPajama — Together Computer

**Citation**: Together Computer (2023). RedPajama: An Open Source Recipe to Reproduce LLaMA training dataset. github.com/togethercomputer/RedPajama-Data.

**Used in**:
- Update 1 §U.1 (long variant training data composition)

**Why cited**: long variant uses 25% RedPajama Book + 25% RedPajama ArXiv + 25% other RedPajama. Citing this grounds the "25% Book + 25% ArXiv" claim used in the long → reasoning analysis.

---

## B.12 What's *not* yet cited (intentional gaps)

For honesty, these are bibliography gaps that v6 should fill before reaching v1.0:

1. **Activation Patching primary references** — Meng et al. ROME (2022), Wang et al. interpretability circuits (2023). Currently only mentioned by name; should add full citations when Phase 6 / v2.0 plans for `patch_activations` capability mature.
2. **Steering vectors primary references** — Turner et al. activation addition (2023), Panickssery et al. CAA (2023), Zou et al. RepE (2023). Phase 5 commit 5.6 implementation should ground in these.
3. **TransformerLens** — Nanda & Bloom. Software dep that Phase 5/6 commits should formally cite if used.
4. **Two-NN intrinsic dimensionality** — Facco et al. (2017). Phase 5 commit 5.5 (`intrinsic_dimensionality_shift`) reference.
5. **C2ST primary reference** — Lopez-Paz & Oquab (2017). Phase 6 commit 6.5 reference.
6. **MMD primary reference** — Gretton et al. (2012). Phase 6 commit 6.5 reference.
7. **AnchorAttention** mentioned in §B.7.2 RoPE/BF16 paper context but not cited separately.

These gaps should be filled progressively as the corresponding commits are implemented. **No citation should be invented** — if a reference is uncertain, the corresponding metric's reference page should explicitly list "primary source TBD" rather than fabricate.

---

## B.13 Citation style policy (for Phase 3 docs work)

For consistency in the docs site (Phase 3 commits 3.2–3.7):

- **In running prose**: "Belrose et al. (2023)" or "Tuned Lens (Belrose et al., 2023)"
- **First mention per page**: include the venue if recognizable (ICLR / NeurIPS / Nature / TACL); otherwise just year and arxiv id
- **Reference page footers**: full citation in author-year-title-venue-arxiv form
- **Method recipe pages**: link directly to arxiv abstract URL when available
- **Recent (2025+) papers without venue yet**: cite as "arxiv YYYY.NNNNN" without forcing a venue assignment

Phase 3 commit 3.7 (positioning) should consolidate this bibliography into a `docs/source/about/citations.md` page or merge it into `docs/source/concepts/related-work.md`. The latter is preferred because it lets readers see the citation in context with what the paper does.


---

# UPDATE 3 — 2026-04-30: Phase 1 retrospective + v0.4.0 scope adjustment

This update is a retrospective on what Phase 1 (v0.3.0 / v0.3.1 / v0.3.2) actually shipped versus the original plan, and the resulting adjustments to Phase 2 (v0.4.0) scope. Written after v0.3.2 release to PyPI.

The structure mirrors Updates 1 and 2: append-only, sections numbered W.1 through W.7. Original §1–§20 and U.1–U.7 / B.1–B.13 stand as-is.

---

## W.1 What Phase 1 actually shipped (v0.3.0 → v0.3.2)

The Phase 1 plan (§4) specified 12 commits 1.1–1.12. Actual delivery used 7 PRs spanning 3 PyPI releases. All 12 commits' visible deliverables landed; the API surface and report system match §4 exactly. Several internal architectural decisions diverged from the plan:

| Commit | Plan | Actual |
|---|---|---|
| 1.1 (compare/family) | "wires up Engine + Config" | Wired API surface only — internally still calls v0.2.x `run_family_experiment` which uses `InferenceEngine`, not `HFEngine`. Architectural debt. |
| 1.2 (Config) | as planned | as planned |
| 1.3 (Engine + HFEngine + MinimalEngine + MockEngine) | as planned | as planned, but HFEngine.score originally used joint tokenization (`tokenize(prompt+continuation)`) — diverged from lm-eval-harness convention. Caught and fixed in v0.3.1. |
| 1.4 (schema v5) | "share_per_domain field added" | shipped, but with formula bug — used raw squared magnitudes instead of per-domain per-token normalized squared. Long-context probes (~9000 tokens) dominated 89-99% of every variant's share regardless of variant character. Caught from user feedback during 7-variant demo; fixed in v0.3.2 PR #11. |
| 1.5–1.10 (renderers, figures) | as planned | as planned |
| 1.11 (markdown / JSON polish) | as planned | as planned, with magnitude display bug — markdown / terminal showed raw L2 magnitude, not per-√token normalized. Caught from same demo; fixed in v0.3.2 PR #8. |
| 1.12 (release commit) | tag and ship v0.3.0 | shipped on schedule |

Two additional unplanned releases happened:

- **v0.3.1** — defensive fixes: `InferenceEngine.device` anchor (multi-7B-variant OOM root cause), `HFEngine.score` brought to lm-eval-harness convention as preparation for eventual cutover, `json_report` parent.mkdir.
- **v0.3.2** — fat release containing 5 PRs: `n_probes` per-task semantics for multi-task `lm_eval:` strings, figure layout for N>4 variants, hardcoded longbench caveat removal, magnitude display fix, progress bars + `device_map_summary`, engine reuse + variant release (the actual OOM fix), `share_per_domain` formula correction, figure spacing + legend overflow + Domain↔dataset panel.

Net effect: Phase 1 took ~4 weeks longer than planned (extending into v0.3.2 release in late April). The user-facing v0.3.2 release is what §4 envisioned for v0.3.0; v0.3.0/0.3.1 were learning steps that exposed gaps.

---

## W.2 The HFEngine path is shipped but not on the default route

This is the single largest piece of architectural debt remaining at end of Phase 1.

**What's in main**:
- `HFEngine` (PR #2 / commit 1.3) — fully implemented, lm-eval-harness convention as of v0.3.1, byte-equivalent to `InferenceEngine.score` within 1e-5 (verified by `tests/integration/test_engine_equivalence.py`)
- `compare()` / `family()` — accept `engine=` parameter; use the user's engine when passed
- Default path (when `engine=None`): builds `HFEngine` instances for `_check_capabilities` preflight, then **discards them and re-loads as `InferenceEngine`** for the geometry path

**What this means**:

1. v0.3.x users running `lmdiff.compare("gpt2", "distilgpt2")` actually run on v0.2.x InferenceEngine
2. HFEngine code path exists, is tested, but is exercised only by users who explicitly write `engine=HFEngine(...)`
3. The "Phase 1 ships HFEngine" promise is technically true (the class exists and works) but practically empty (default users don't use it)

**Why this happened**:

PR #3 instruction (commit 1.1) underspecified the cutover. CC honored the spec by wiring up the API surface and delegating to existing v0.2.x backend — fastest path to land the API. The full cutover (rewrite probe loop + metric computation on top of HFEngine.score) was implicitly deferred without being explicitly scheduled.

The v0.3.2 OOM fix (PR #10) did cut the redundant HFEngine preflight load (saves 8 dead loads per family() call) but kept the default path on InferenceEngine. So at v0.3.2, the architecture has two backends in parallel — InferenceEngine actually runs experiments, HFEngine runs only when explicitly requested.

**Resolution**: Phase 4 commit 4.0 (W.5).

---

## W.3 The bigger lesson — Engine implementation cross-checking

When implementing PR #2 (commit 1.3), CC tested HFEngine in isolation:
- Self-consistency: same input → same output across calls
- API conformance: matches Engine Protocol
- Smoke test: produces sensible logprob on real model

What was **not** tested: HFEngine.score's output cross-checked against InferenceEngine.score on the same `(prompt, continuation)` pair.

Had this been tested, the joint-tokenization bug would have been caught at PR #2 review. Instead it was caught one week later when planning the v0.3.1 cutover and CC audited the actual scoring code.

**Phase 4 commit 4.0 hard requirement**: any Engine implementation that becomes the default backend MUST have a cross-Engine equivalence integration test against the previous default. This catches semantic divergence at PR review, not at backend cutover.

This generalizes to all future backends (vLLM, TGI, hosted APIs in Phase 6+). New Engine should not become a default path until it has byte-level equivalence (or documented expected divergence) verified against a known-good reference.

---

## W.4 Per-domain per-token normalization is the canonical lmdiff metric — make it explicit in plan

The `share_per_domain` formula bug (PR #11) revealed a more fundamental issue: the plan §4.5 listed "share_per_domain — rows sum to 1.0" but didn't specify the formula. CC chose raw squared magnitude as the most obvious implementation — semantically correct but length-biased on heterogeneous probe sets.

The correct formula (now in v0.3.2):
```
pdn[v][d]                = sqrt( Σ_{i∈d} δ[v][i]² / Σ_{i∈d} T[i] )    # per-domain per-token normalized
share_per_domain[v][d]   = pdn[v][d]² / Σ_d' pdn[v][d']²              # length-neutral share
magnitudes_normalized[v] = sqrt( mean over d of pdn[v][d]² )          # equal-weight overall
```

This formula matches the v6 §13 calibration mockup exactly (long → reasoning 66%, yarn → commonsense 51% — verified post-fix).

**Update to Phase 1 §4 specifications**: §4.5 ("Schema v5 fields") should list the formulas, not just the field names. The same applies to any future schema changes — formula-level specification, not just field-level.

**Update to Phase 4 metric registry (§7)**: when adding new geometry metrics, the metric registry entry must specify both the field name and the computation formula explicitly.

---

## W.5 Phase 4 commit 4.0 (NEW): Backend cutover

The original Phase 4 plan (§7) was "metric registry expansion" — add new metric implementations using the existing Engine abstraction. This update inserts a new commit 4.0 **before** any metric work, and shifts all existing commits 4.1+ by one number.

**Commit 4.0 — Backend cutover** (rewrite of W.2 + practical implementation):

Goal: `compare()` / `family()` route through `HFEngine` directly. Internal probe loop, metric computation, BPB normalization, and GeoResult assembly are rewritten on top of `Engine` Protocol. v0.2.x `InferenceEngine` retained only for deprecated `ModelDiff` API path; removed in v0.5.0.

Subtasks:

1. **Pre-cutover audit**: trace exact call chain `compare()` → final GeoResult. Document every place `InferenceEngine` is instantiated, every direct attribute access (e.g. `engine._tokenizer`, `engine._model`), every place lm-eval probe loading happens. CC writes audit report; user reviews before cutover begins.

2. **`lmdiff/_pipeline.py`** (new): the cutover-phase data flow. Takes a base `HFEngine` + N variant `HFEngine`s + a probe set, produces a fully-populated `GeoResult` v5. Replaces `run_family_experiment` for the default API path.

3. **`Engine` Protocol extension** (if needed): if v0.2.x metric code reaches into engine internals (`._tokenizer`, etc.) that the protocol doesn't expose, surface them as protocol methods. Don't allow private-attribute reach into HFEngine from `_pipeline` — that's debt-trading.

4. **Calibration regression test** (mandatory): in `tests/integration/test_calibration_regression.py`, run Llama-2 4-variant on real models, assert byte-identical output to v0.3.x baseline (1e-6 tolerance for any field touching float math, exact match for everything else). Baseline is committed at `tests/fixtures/calibration_v030_baseline.json` — frozen ground truth.

5. **`run_family_experiment` deprecation**: emit `DeprecationWarning` at call site. Continues working with `InferenceEngine` for v0.4.x. Removed in v0.5.0.

6. **`InferenceEngine` deprecation**: same lifecycle.

7. **`ModelDiff` API removal**: v0.4.0 removes the v0.2.x deprecation shim entirely (warned since v0.3.0).

Estimated time: 1.5-2 weeks CC + 1 week reviewer (calibration regression review is non-trivial).

This commit is the gate for all subsequent Phase 4+ work that touches metric computation. Commits 4.1+ (metric registry expansion) build on top of `_pipeline.py`, not on `run_family_experiment`.

---

## W.6 Updated Phase 2 (v0.4.0) scope and ordering

The Phase 2 plan (§5) — probe taxonomy + 4 builtin task probe sets + YAML loader — stands as-is in subject matter. Order of work changes:

| New commit number | Original number | Subject |
|---|---|---|
| 4.0 (NEW) | — | Backend cutover (W.5) |
| 4.1 | 4.1 | Probe taxonomy (`task_type` field) |
| 4.2 | 4.2 | 4 builtin task probe sets |
| 4.3 | 4.3 | YAML probe set loader |
| 4.4 | — | v0.2.x deprecation removal (`ModelDiff`, `run_family_experiment`, `lmdiff.config.Config`) |

v0.4.0 ships all of 4.0 through 4.4 as a single minor release. Estimated total: 4-6 weeks CC + 2-3 weeks reviewer.

The 4 builtin task probe set selection (commit 4.2) — `safety_regression`, `hallucination`, `instruction_following`, `general_capability` — should be **revisited based on lab feedback after v0.3.2 release**. v6 plan §5 selection was theoretical; actual user demand may differ.

---

## W.7 Carry-over architectural notes for future phases

These observations are not actionable in v0.4.0 but should inform v0.5.0+:

1. **Engine implementations should be stateless on runtime parameters**. v0.3.2 made `InferenceEngine.score()` stateless (kwargs override). HFEngine in PR #2 was already stateless. Any future Engine (vLLMEngine, APIEngine) MUST follow the same pattern. This is a precondition for engine reuse and simplifies cache invalidation.

2. **`MODEL_SPECIFIC_COMPARATORS` extension hook is empty in v0.3.2**. First non-empty registration will likely come from a Phase 6 use case (custom Engine wrapping a fine-tuned proprietary model whose Config carries extra fields). When the first registration happens, formalize the convention (where to register, how to test, how to document).

3. **`look-ahead-by-one` engine release strategy is conservative — re-loads same model if it appears non-consecutively in iteration order**. For small models or fast disk this is fine. For 70B+ models on HDD, may want smarter caching. v0.5.0+ if it becomes a bottleneck.

4. **`lm_eval:` multi-task probe loading uses per-task `n_probes` semantics**. Flat probe sets (string name like `"v01"` or `ProbeSet` instance) keep "N total" semantics. This asymmetry is documented in `lmdiff.compare()` docstring. For Phase 2's probe taxonomy expansion, ensure new probe loading mechanisms (YAML, file path) document semantics explicitly — don't inherit ambiguity by default.

5. **The `pdn` (per-domain per-token normalized) field is the canonical first-class metric for multi-domain runs**. Renderers, findings, and external API consumers should read `magnitudes_per_domain_normalized` (added v0.3.2) rather than computing from raw `magnitudes`. Phase 4 metric registry should list `pdn` first when describing geometry-level metrics.

6. **Engine API stability promise** (clarification for v0.4.0+): the `Engine` Protocol method signatures are extensible (new optional kwargs allowed) until v1.0 LTS. v0.3.1 added `continuation_ids` to `score()` — that was OK because it's optional. Removing or renaming an existing kwarg before v1.0 is a breaking change requiring major version bump or deprecation cycle.

---

## W.8 What v6 plan §4 description should now read (mental update)

If §4 were rewritten today knowing what we know after v0.3.2, the relevant changes would be:

- §4.1: "compare() / family() wire up the API surface; **internal cutover to HFEngine deferred to commit 4.0 (W.5)**"
- §4.5: "share_per_domain field is computed as `pdn[v][d]² / Σ_d' pdn[v][d']²` where pdn = sqrt(Σ_{i∈d} δ[v][i]² / Σ_{i∈d} T[i])"
- §4.6: "Engine implementations that become the default path must have cross-Engine equivalence integration tests (W.3)"
- §4.7: "v0.3.2 ships an additional fat release containing engine reuse, share formula correction, and figure polish — not originally in §4 but necessary based on actual usage patterns"

The plan stands as the design document of intent; this Update preserves the as-built record without rewriting history.


---

# UPDATE 4 — 2026-05-11: v0.4.0 ship retrospective (Phase 2 commit 4.0)

This update is a retrospective on the v0.4.0 backend cutover. Written after PR #15 merged and v0.4.0 shipped to PyPI. Sections X.1 through X.9, append-only, parallel structure to Update 3.

The most important observation is that the audit-then-implement gate worked as intended for some classes of risk but missed others. Documenting both is the point of this update.

---

## X.1 What v0.4.0 actually shipped

Original Phase 2 plan (W.5 from Update 3) specified commit 4.0 as a single architectural piece: `compare()` and `family()` route through HFEngine via new `_pipeline.run_family_pipeline`. Mandatory pre-flight audit + calibration regression test as hard contract.

Actual delivery in PR #15:

| Commit | Subject | Source |
|---|---|---|
| `eb3f310` | Pre-flight audit doc (~420 lines) | Spec mandatory |
| `6703970` | Calibration baseline JSON committed | Spec mandatory |
| `448554f` | Audit decisions inlined | Process |
| `0b67f6c` | Engine Protocol additions (token_count, tokenizers_equivalent_to) | Audit recommendation |
| `50478c4` | `_pipeline.py` + `_api.py` rewire + 4 wiring-test patches | Spec Task 2-3 |
| `d8e1f8a` | run_family_experiment deprecation + calibration regression test + 13 pipeline unit tests + CHANGELOG + migration doc | Spec Task 4-7 |
| `f697a2f` | Fix 1 (top_k passthrough) + Fix 2 (prefix_text kwarg) + 6 new unit tests | Discovered post-cutover |
| `4b65fa7` | 7-variant calibration regression test fixture + L-030 lesson | Discovered post-cutover |
| `8fa6ee9` | Fix 3 (seed plumbing end-to-end) + L-031 + 6 plumbing tests | Discovered post-cutover |
| (later) | Fix 4 (lazy engine loading regression repair) + L-032 + 6 lazy-load tests | Discovered post-cutover |
| (release) | CHANGELOG date + version bump → PyPI publish | Process |

Original spec estimated 1.5-2 weeks CC + 1 week reviewer = ~3 weeks elapsed. Actual: ~2 weeks total elapsed, with ~1 week of cutover work and ~1 week of fix work post-cutover.

The fix work was not in the original spec. Spec scope assumed: audit catches everything, calibration regression test catches anything audit missed. Reality: calibration regression test only covered the 4-variant deterministic case; 7-variant test (added mid-PR after the first two fixes) was what surfaced the remaining bugs.

---

## X.2 The four post-cutover fixes (Fix 1 through Fix 4)

Each fix exposed a different class of failure in the original audit + spec.

### Fix 1: `top_k` passthrough

**Symptom**: User's 7-variant demo run showed `temp_1.5` share collapsed from v0.3.2's 34% reasoning to 5%.

**Root cause**: `HFEngine.generate` did not accept or pass through `top_k`. HuggingFace's `model.generate` defaults `top_k=50` when not specified — silently truncating sample distributions. v0.2.x `InferenceEngine` passed `top_k=0` (no filtering) explicitly; v0.4.0 did not.

**Why audit missed it**: Audit cataloged "decode params per call" as Protocol-gap but assumed it was just three params (`temperature`, `top_p`, `seed`). DecodeSpec actually has more fields, including `top_k`. Spec for HFEngine.generate signature was based on memory of v0.2.x InferenceEngine without exhaustive cross-check.

**Why calibration missed it**: 4-variant calibration uses only greedy decoding. No sampling variant means `top_k` plumbing untested.

**Lesson**: Cross-engine equivalence tests must cover every value of every config field, not just the canonical happy-path. (Codified as L-030.)

### Fix 2: `prefix_text` tokenization preservation

**Symptom**: User's 7-variant demo run showed `system_prompt` share on commonsense jumped from v0.3.2's 60% to 94%.

**Root cause**: v0.2.x InferenceEngine used split tokenization for system_prompt prefix: `tokenize(prefix, add_special=True) + tokenize(probe, add_special=False)`. v0.4.0 `_assemble_prompt` used joint tokenization: `tokenize(prefix + probe, add_special=True)`. For Llama SentencePiece at boundaries like `"\n→Y"`, the two are not always byte-identical.

**Why audit missed it**: Audit 4 (BPB / cross-tokenizer) noted SentencePiece boundary issue but rated it "rare for clean newline boundaries". Underestimate.

**Why calibration missed it**: 4-variant calibration uses only weight-level variants (yarn/long/code/math) — none have system_prompt or context fields. The prompt-assembly code path that diverges is exercised only by Configs with non-None `system_prompt`, `context`, or `icl_examples`.

**Lesson**: Tokenization equivalence must be tested at every interface where prompt assembly happens, not assumed to hold by inspection. (Same L-030 lesson.)

### Fix 3: `seed` plumbing end-to-end

**Symptom**: 7-variant calibration test produced different probe counts (497 vs 500) across back-to-back GPU runs of the same code on the same input.

**Root cause**: Three compounding facts:
1. `family(seed=...)` kwarg accepted but never passed to `run_family_pipeline`. Public API docstring read "Reserved for future randomized metrics; v0.3.0 ignores it."
2. `DecodeSpec.seed` default `None` — so `temp_1.5` Config had no explicit seed
3. `HFEngine.generate` only called `torch.manual_seed` when given explicit seed; with `None`, RNG state inherited from prior 5 variants' work (cumulative model loads, tokenizer canary checks, 1000+ forward passes)

Result: `temp_1.5`'s `model.generate` produced slightly different sampled outputs each run; some hit EOS immediately → empty completions → NaN → global filter dropped them → final probe count wobbled.

**Why audit missed it**: Audit recognized "DecodeSpec.seed" as a runtime parameter the new pipeline must thread. Didn't recognize that `family(seed=...)` itself was a documented-as-no-op kwarg that needed wiring through.

**Why calibration (4-variant) missed it**: No sampling variant → no RNG dependence → deterministic regardless of `manual_seed` call.

**Lesson**: Public kwargs that are "reserved for future" tend to mask bugs. Either implement or remove — don't ship API surface that's documented to not work. (Codified as L-031.)

### Fix 4: Lazy engine loading regression

**Symptom**: 7-variant Llama-2 family completed but threw a CPU spillover warning during the run, with throughput dragged into the ground. Originally hidden behind the (then-passing) calibration test.

**Root cause**: v0.4.0 cutover refactor accidentally regressed v0.3.2's lazy-load + look-ahead-by-one release behavior (L-029). `_api.family()` was building all unique-model variant engines into an `engines: list` *before* `run_family_pipeline` started — so 6 × 14 GB BF16 engines sat in memory simultaneously even though `_pipeline` had look-ahead release machinery ready to free them.

**Why audit missed it**: The audit didn't model engine lifetime as a contract. Engine construction was treated as "where InferenceEngine gets built" without tracking when it gets freed. The L-029 lazy-loading invariant from v0.3.2 was implicit, not codified.

**Why calibration missed it**: 4-variant calibration on real Llama-2 used ~56 GB VRAM total (4 engines × 14 GB). 2 × A6000 (96 GB) accommodates that without CPU spill, so the regression didn't trigger calibration failure. 7-variant (98 GB raw) triggered the spillover that exposed it.

**Lesson**: Architectural invariants (lazy loading, look-ahead release, peak resident engine count) need explicit invariant-style tests. Loose statements like "the pipeline manages engines correctly" are not enforceable. (Codified as L-032.)

---

## X.3 What the audit-then-implement gate actually caught

The pre-flight audit gate caught real issues before implementation, justifying the spec's choice to make it mandatory:

- **Engine Protocol gaps**: 5 attribute accesses on `InferenceEngine` outside Protocol (all in `geometry.py`) — caught and categorized. Two real gaps required new Protocol methods (`token_count`, `tokenizers_equivalent_to`); the other three were eliminated by inline rewrite.
- **`run_family_experiment` external caller surface**: Documented before deprecation, avoiding silent breakage.
- **Scope sizing**: Audit confirmed full cutover in one commit was feasible (no Protocol gap so large that splitting was needed). Spec's "scope-down option" (compare-only first) was held in reserve and not needed.
- **Calibration baseline as frozen ground truth**: Audit's "user runs baseline on GPU, commit JSON" step prevented post-hoc rationalization ("oh, the new path produces these numbers, that's the new contract").

What the audit did NOT catch:

- DecodeSpec field completeness (Fix 1)
- Tokenization boundary divergence severity (Fix 2 — was acknowledged then rated rare)
- Public kwarg plumbing for `family(seed=...)` (Fix 3 — not modeled at all)
- Engine lifetime invariant from L-029 (Fix 4 — not codified, implicit)

Pattern: audit was thorough on what existed in the codebase (Protocol surface, call chain, attribute access). Audit was incomplete on what should exist but didn't (e.g. a documented invariant that wasn't enforced anywhere) and on edge cases beyond the canonical happy-path.

---

## X.4 What "byte-identical" calibration regression test actually proved

The 4-variant calibration regression test passed 11/11 — and was insufficient as the cutover gate.

What it proved:
- Engine score implementation byte-identical to InferenceEngine on the 4 calibration variants
- Per-domain normalized magnitude computation byte-identical
- Share-per-domain computation byte-identical
- Cosine matrix computation byte-identical
- Schema serialization byte-identical
- Engine reuse + variant release path byte-identical

What it did not prove:
- Sampling decode equivalent (no sampling variant in test)
- system_prompt / context handling equivalent (no such variants in test)
- `family(seed=...)` actually does anything (test didn't exercise it)
- Engine lifecycle scales beyond 4 variants without OOM (4 × 14 GB fits in 96 GB)
- Cross-run reproducibility of any variant (test ran cutover once, compared to fixed baseline)

The 7-variant calibration regression test (added mid-PR as part of Fix 1+2) covered some of these gaps but introduced its own incompleteness (fixture content audit revealed it may not have included `change_vectors` originally — this would have meant the byte-equivalence assertion was silently never running).

**Lesson** (refines L-030, partially codified there): "Calibration regression test" is too generic a contract name. The test fixture must be exhaustively documented and audited; specifically: (a) every field referenced by an assertion must be in the fixture, and (b) the fixture must cover every code path the new implementation introduces.

---

## X.5 Process learnings

### When CC pushes back on user, listen carefully

After the 9-failure pytest report, my (the human's) first reading was "the cutover is wrong, 3 probes are missing." CC's reading was "the baseline fixture says 497, the cutover produced 500 — the cutover is *more inclusive*, this is good news." Both readings were defensible from the pytest output text.

Resolution required CC to verify the pytest output direction more carefully and report back. CC did this and confirmed the user's reading was wrong: cutover ran #1 produced 497, #2 produced 500. Same code, same inputs, different probe counts → non-deterministic RNG.

If I had asserted my reading and pushed CC to "fix the test", the real bug (Fix 3) would have been missed.

**Lesson**: When CC and human read the same evidence differently, the resolution is verification, not authority. CC's habit of saying "I read the failure as X — please confirm" before proposing fixes is the right pattern.

### Audit-stop-implement-stop pattern is worth the friction

Spec mandated CC stop after audit for user review, before any implementation code. CC followed this. Mid-implementation when CC discovered Fix 1+2 from the user's 7-variant demo data, CC again stopped, reported, and proposed fixes — didn't silently expand cutover scope to include them.

This stop-and-report cycle made it possible to:
- Adjust scope (Fix 4 lazy loading was added as a follow-up commit rather than a separate PR)
- Catch CC mis-reasoning (the "v0.4.0 is strictly more inclusive" claim was pushed back on)
- Document each fix's discovery context (Fix 1+2 from user demo; Fix 3 from 7-variant calibration test; Fix 4 from CPU spill warning)

**Lesson**: Single-PR with multiple commits + explicit stop points is better than many-PR for architectural cutovers. The PR description ends up as a complete record of "what we found and decided" instead of git-archaeology across PRs.

### Calibration baseline must be generated *before* implementation, not after

Spec required this and CC followed it. User ran the 4-variant baseline on GPU, committed JSON, then CC implemented. This made calibration test a true contract (cutover output must match committed baseline) rather than a self-consistency check (cutover output must match cutover output).

The 7-variant fixture introduced mid-PR was less rigorous about this — it was extracted from `demo_032_rerendered` (the user's earlier demo run), not generated fresh on GPU with the v0.3.2 release tag in a controlled run. This is part of why fixture content audit was needed.

**Lesson**: Calibration baselines for cutovers should be generated from the previous version under controlled conditions (tagged release, documented env vars, single-run), not extracted from any existing artifact.

---

## X.6 Update to W.5 (Phase 4 commit 4.0 spec, after the fact)

The Phase 4 commit 4.0 spec (from W.5) was generally correct as a process. Adjustments based on actual delivery:

| Spec said | Actual lesson |
|---|---|
| "Calibration regression test asserts byte-identical on 4-variant case" | Also need a multi-variant test covering every code path the new implementation introduces (sampling, prefix-text, ICL, etc.) |
| "Engine Protocol extensions only when needed" | Same, but exhaustive Protocol-vs-Config-field audit before signing off ("does every DecodeSpec field map to a kwarg or method on Engine?") |
| "User generates calibration baseline JSON on GPU box" | Add: "User regenerates baseline if the implementation introduces new code paths the original baseline doesn't exercise" |
| "Tasks 5+6 (calibration + tests)" | Add Task 8: "Multi-variant calibration including sampling + prefix variants" as mandatory pre-merge |

The spec did not include lazy-load lifecycle as an invariant to preserve. Future cutover specs should explicitly enumerate v0.3.x invariants from L-024 through L-032 and require the cutover audit to confirm each one is preserved or explicitly broken.

---

## X.7 Phase 2 (v0.4.0+) remaining work

After commit 4.0 ship, Phase 2 has commits 4.1-4.4 still scheduled per W.6:

- **4.1**: Probe taxonomy (`task_type` field). Adds 8 task types: `safety_regression`, `hallucination`, `instruction_following`, `general_capability`, `code_generation`, `math_reasoning`, `commonsense_reasoning`, `long_context`. Schema additive, no bump.
- **4.2**: 4 builtin task probe sets — selection revisitable based on lab feedback. Original v6 §5 selection was theoretical.
- **4.3**: YAML probe set loader.
- **4.4**: v0.2.x deprecation removal (`ModelDiff`, `lmdiff.config.Config`, `run_family_experiment`, `InferenceEngine`, `ChangeGeometry`). Schedule pushed to v0.5.0 due to L-028 minimal-hotfix discipline.

Estimated total for 4.1-4.3 alone: 2-3 weeks CC + 1 week reviewer. Each commit is smaller than 4.0 (no architectural cutover), so audit phase should be shorter or skippable.

4.4 (v0.2.x removal) is now a v0.5.0 work, not v0.4.x. Reason: `run_family_experiment` is documented public API and shipped through v0.4.0 with `DeprecationWarning`. v0.4.x users may be incrementally migrating; one full minor release cycle of warnings is the right buffer.

---

## X.8 New carry-over architectural notes

Adding to W.7's list:

7. **All public kwargs must be functional from declaration**. `family(seed=...)` and `compare(seed=...)` shipped through v0.3.x as documented-no-op kwargs. This kind of "API surface exists but doesn't work" is worse than "API surface doesn't exist" because users discover it post-deployment in production.

8. **Engine lifecycle is a Protocol-level contract**. `Engine.close()` exists in the Protocol but lifecycle expectations (when does pipeline call close, who owns the lifetime, what's the peak resident count) need to be explicitly stated alongside the Protocol. Add to engine.py docstring in next pass.

9. **Cross-cutting invariants need explicit tests, not just code**. Lazy-load + look-ahead-by-one release (L-029) was a real invariant in v0.3.2 that got accidentally regressed in v0.4.0 because no test asserted "at most 2 engines resident at any time". Phase 2 commits 4.1-4.3 should add similar invariant-style tests for each new contract introduced.

10. **`MockEngine` should be a first-class API conformance test**. CC's plumbing tests for Fix 3 used MockEngine successfully. As `Engine` Protocol grows (Phase 4+ may add hidden_states, attention_weights, etc.), MockEngine implementations of new methods become the spec for "what does Protocol-correctness mean here". v0.4.0 MockEngine is essentially complete; future Engine implementations should be tested against the same MockEngine reference behavior.

---

## X.9 Mental update to v6 plan §4 specifications

Sections to add to §4 specifications based on Fix 1-4 lessons:

- **§4.3 (Engine implementations)**: every public method should accept every value of every DecodeSpec / Config field that affects its behavior. Audit method signature against Config schema during implementation, not after.
- **§4.5 (Schema fields)**: extend W.4 — when implementing a derived field, audit which code paths produce it (with which inputs) and ensure they're all in the cutover calibration test.
- **§4.6 (Cross-engine equivalence)**: extend W.3 — equivalence test must cover at minimum: (a) every Engine method, (b) every value-set of every parameter that affects behavior, (c) every public kwarg of the API that the engine consumes.
- **§4 new section §4.8 (Engine lifecycle)**: codify the invariants: `Engine.close()` releases all model state; peak resident engines in a single `family()` call is `1 + 1` (base + active variant) under default config; pipeline owns the lifecycle of engines it constructs, callers own engines they pass in.


---

# UPDATE 5 — 2026-05-11: Methodology critique + Phase 2 reorganization

This update follows lab feedback "我们额外的 normalization（根号 T）似乎没什么根据" and the subsequent audit chain. Written same day as Update 4, but separate update because the conclusions reach into methodology design rather than ship retrospective. Sections Y.1 through Y.10, append-only.

The key outcome is **a new commit 4.1 inserted before existing Phase 2 work**: normalization + measurement validity framework. All originally-planned 4.1 through 4.4 shift up by one number. Phase 2 grows from 5 to 6 commits.

---

## Y.1 The √T critique and what it actually exposed

Lab feedback was three Chinese characters of critique: 我们额外的 normalization（根号 T）似乎没什么根据 ("our extra √T normalization seems to have no basis"). What followed was an iterative debug chain across the same conversation, with user push-back at three critical points correcting earlier wrong answers.

Timeline:

1. **First wrong defense**: Claude framed √T as a clean derivation from the Oyama et al. (2025, ACL) "log-likelihood vector" theorem: 2·KL ≈ ||q||²/N → √(2·KL) ≈ ||q||/√N. This *would* justify √T normalization — *if* the lmdiff δ vector represented total log-likelihood differences (scales with T).

2. **User push-back #1**: "我们不是已经除过一次 token 数了吗" ("haven't we already divided by token count once?"). This was the critical correction. CC audit confirmed: lmdiff's δ is already per-token CE difference (T-invariant by construction from `score()` return value). The Oyama framework requires δ to be total log-likelihood (scales with T) — unit mismatch invalidates the derivation.

3. **Reframed as Option C (token-weighted RMS)**: Claude then proposed `sqrt(Σ T_i δ_i² / Σ T_i)` as the dimensionally clean alternative, framing it as "Oyama-aligned with correct unit handling."

4. **User push-back #2**: "token-weighted RMS 还会 long context 主导吗" ("would token-weighted RMS still let long-context dominate?"). CC audit on demo data confirmed: Formula C makes long-context share dominate at 50–99% for 6 of 7 variants, destroying the specialization narratives (CodeLlama → code, llemma → math, chat → reasoning, system_prompt → commonsense).

5. **Second wrong defense**: Claude then proposed Path B (Formula C + specialization layer dividing by all-variant geomean baseline), framing dimensional cleanness as the fix.

6. **User push-back #3**: "long context 下 base 崩溃" ("base model collapses on long context"). This was the methodology-altering observation. Long-context σ inflation is not real per-token-drift signal — it is base-model catastrophic failure noise. Llama-2-7B (4K RoPE training) on a 9000-token probe produces garbage attention patterns and inflated CE for both base and variant. Variant-base δ is large but represents **measurement invalidity**, not specialization.

This third push-back **invalidated all three preceding formulations** (current Formula B, Formula C alone, Formula C + specialization layer). All were trying to find the right normalization for a measurement that is invalid at the probe level for out-of-context prompts.

**Lesson**: When defending a metric, the first question is "does the underlying measurement support the question being asked?" not "what's the right normalization?" The √T critique surfaced via lab framing as a normalization issue, but the deeper problem was measurement validity. Normalization tricks cannot fix invalid measurements.

(Codified as L-033 in LESSONS.md, see Y.8 below.)

---

## Y.2 The current Formula B's actual semantics

For the record, since this matters for the v0.4.x → v0.5.0 transition:

Current `pdn[v][d] = sqrt(Σ_{i∈d} δ_i² / Σ_{i∈d} T_i)` with δ ∈ nats/token computes:

```
pdn_d = σ_d / √T̄_d
```

where σ_d is the per-token CE diff RMS across probes in domain d, and T̄_d is the average probe token count in domain d.

This is dimensionally inconsistent (nats/token^1.5). It has no statistical derivation. Its empirical effect is to suppress long-context domains by factor √T̄_d, which **incidentally** mitigates the long-context dominance in raw `||δ_d||²` shares — but the mitigation works on the wrong mechanism (token-budget division), not on the actual cause (long-context probes are invalid measurements for short-context base models).

The Formula B numbers (long → reasoning 66%, yarn → commonsense 51%, system_prompt → commonsense 60%, CodeLlama → code 32%) were the published v0.4.0 showcase numbers and the v6 §13 calibration mockup numbers. They are **self-consistent** (mockup derived from same formula as implementation) but not externally validated against a theoretical or empirical ground truth.

---

## Y.3 The measurement validity reframe

Per user observation: long-context σ inflation is not per-token signal compounding over many tokens (CC's initial interpretation). It is the base model failing to handle prompts beyond its training context window. Specifically:

- Llama-2-7B has 4K RoPE context window
- `longbench_2wikimqa` probes are 9000–10000 tokens
- Beyond 4K: RoPE position embeddings extrapolate to untrained ranges, attention patterns degrade catastrophically, per-token CE inflates substantially
- This affects base AND variants that share the same context window
- δ = ce_base - ce_variant is large in absolute value but reflects two-sided failure noise, not a real differential signal

Evidence that this is the right interpretation (not "per-token compounding"):

1. ALL variants share comparable long-context σ inflation (Yarn 7.62, but also CodeLlama, llemma, chat etc.) — if signal were specialization, variants should differ in long-context σ
2. CodeLlama and llemma share Llama-2's 4K context window, so neither has "real" long-context capability — yet both show σ_long >> σ_other
3. Yarn-128K and LLaMA-32K actually handle long-context, but their σ_long against 4K base is dominated by base failure, not variant capability

**Implication**: any measurement framework that surfaces long-context σ as "share" — whether B, C, or B + specialization layer — leaks base failure noise into the user-facing showcase.

The correct solution is upstream of normalization: detect invalid measurements at the probe level and exclude them from aggregation.

---

## Y.4 New commit 4.1: Normalization + measurement validity framework

Inserted as **new commit 4.1 in Phase 2**, between completed 4.0 (backend cutover, v0.4.0) and existing 4.1+ (probe taxonomy). All originally-numbered 4.1 through 4.4 shift up by one. Ships as **v0.4.1**.

**Commit 4.1 (NEW): `feat(geometry): measurement validity framework + normalization correction`**

Components:

1. **Per-probe validity check**:
   ```python
   # In _pipeline._delta_for_variant or new validity layer
   def _check_probe_validity(probe, base_engine, variant_engine):
       T_i = base_engine.token_count(probe.full_text)
       base_max = getattr(base_engine.config, "max_position_embeddings", None)
       variant_max = getattr(variant_engine.config, "max_position_embeddings", None)
       
       if base_max is not None and T_i > base_max:
           return Invalidity(reason="exceeds_base_context", limit=base_max)
       if variant_max is not None and T_i > variant_max:
           return Invalidity(reason="exceeds_variant_context", limit=variant_max)
       return Valid()
   ```

2. **Domain status taxonomy**:
   ```python
   domain_status[d]:
     - "full":           all n_d probes valid
     - "partial":        some probes valid, some invalid (cross-variant pattern matters)
     - "variant_only":   invalid for base, valid for at least one variant — 
                         showable via variant-only metric (v0.5.0+)
     - "out_of_range":   invalid for everyone
   ```

3. **Share calculation excludes invalid domains**:
   ```python
   share_per_domain[v][d] = pdn[v][d]² / Σ_{d': status[d'] in {"full", "partial"}} pdn[v][d']²
   # For status="out_of_range", share_per_domain[v][d] = None / NaN
   ```

4. **Drop √T̄ over-correction in pdn formula**: After validity framework excludes out-of-context domains, the dimensional bug becomes harmless to remove. New formula:
   ```python
   pdn[v][d] = sqrt(mean(δ_i² for valid i in d))   # = ||δ_d|| / √n_d_valid
   ```
   Equivalent to token-weighted RMS when T_i ≈ T̄_d within domain (which is generally true after dropping out-of-context probes).

5. **Schema additions** (GeoResult v6 → v7):
   ```python
   GeoResult.probe_validity: dict[probe_id, ValidityRecord]
   GeoResult.domain_status: dict[domain, DomainStatus]
   GeoResult.share_per_domain[v][d]: float | None   # None for out_of_range
   ```
   Backward compat: v6 georesults load with all probes assumed valid (no validity field) → DeprecationWarning + auto-flag long-context domain as `partial` based on probe length heuristic.

6. **Visualization changes**:
   - `figs/drift_share_dual.png` — out-of-range domains rendered as hatched bar with "—" label or dashed border + tooltip "out of base context (4K)"
   - Variant-specific out-of-range cells (where one variant handles the prompt but another doesn't) use distinct hatching from "all out of range"
   - Legend adds NA + partial markers

7. **Documentation**:
   - New `docs/normalization.md` deriving the pdn formula from first principles (token-level CE diff under no-i.i.d. assumption), with explicit "Why not √T̄?" subsection citing the v0.3.2 PR #11 history
   - Acknowledge that v0.4.0 published numbers used dimensionally-inconsistent formula; v0.4.1+ numbers differ for variants with long-context probes
   - Citation: Oyama et al. (2025) ACL Long Paper, log-likelihood vector framework — establish that lmdiff δ is *per-token* analog of their q vector (not total)

8. **Tests**:
   - Validity unit tests: short prompt + long base passes; long prompt + short base flagged; partial domain (mix of long and short probes) handled
   - Visualization tests: NA / partial render correctly
   - Calibration regression: 4-variant test still passes (calibration probes are within 4K); 7-variant regression test updated to expect long-context domain marked partial/out-of-range
   - Reference v0.4.0 baseline JSON is **NOT** the ground truth anymore — v0.4.1 produces dimensionally-corrected numbers; new baseline regenerated

**Estimated**: 1 week CC audit/design + 1 week CC implementation + 1 GPU validation pass + 3-5 days lab review of new showcase numbers. Total 3 weeks.

**Ships as v0.4.1**. Breaking change to share_per_domain numerics (announce in release notes).

---

## Y.5 Phase 2 commit renumbering and version timeline

Original Phase 2 (Update 3 W.6) had commits 4.0 through 4.4 all shipping as v0.4.0 single release. Update 4 (X.7) noted 4.4 (v0.2.x removal) actually pushed to v0.5.0.

Now Update 5 reorganizes Phase 2 as multi-release:

| Commit | Subject | Version | Status |
|---|---|---|---|
| 4.0 | Backend cutover (HFEngine pipeline) | v0.4.0 | ✓ shipped 2026-05-11 |
| **4.1 (NEW)** | **Validity framework + pdn formula correction** | **v0.4.1** | **scheduled** |
| 4.2 (was 4.1) | Probe taxonomy (task_type field) | v0.4.2 | scheduled |
| 4.3 (was 4.2) | 4 builtin task probe sets | v0.4.3 | scheduled |
| 4.4 (was 4.3) | YAML probe set loader | v0.4.4 | scheduled |
| 4.5 (was 4.4) | Task-type-aware metric registry | v0.4.5 | scheduled |

Estimated overall Phase 2 timeline: 8-12 weeks total. Each commit ships as its own minor patch instead of a single fat v0.4.0 release. Rationale:

- v0.4.0 already shipped; we cannot retroactively add to it
- Validity framework (new 4.1) is significant enough to warrant its own release rather than bundling with probe taxonomy
- v0.4.1 ship gives lab a corrected showcase to react to before downstream probe-taxonomy work commits to a specific normalization
- Subsequent commits remain bundle-able if scope stays modest

v0.5.0 (v0.2.x removal: `ModelDiff`, `lmdiff.config.Config`, `run_family_experiment`, `InferenceEngine`, `ChangeGeometry`) still scheduled per X.7 Update 4 — independent of normalization investigation.

---

## Y.6 Lab feedback collection plan

The new commit 4.1 hinges on validity framework being designed *with* lab input, not delivered to lab as a fait accompli. Two collection points:

**Pre-implementation (1 week)**:
- Share `docs/internal/v041_validity_design.md` (audit + design doc, not code) with lab
- Specific questions for lab:
  1. Does the "long-context probes are invalid measurements" framing match lab's intuition about model behavior beyond context window?
  2. For variant-only valid domains (Yarn-128K handling 9000-token probe while base does not): is hard-exclude (v0.4.1) acceptable, or is variant-only metric (v0.5.0+) a priority?
  3. Are there other measurement-validity issues the lab is aware of? (e.g. instruction-following probes on base models — do non-instruct variants meaningfully "drift" on instruction-following, or is the measurement invalid in the same way?)
  4. The corrected v0.4.1 showcase numbers will differ from current v0.4.0 demo: chat → reasoning, CodeLlama → code, etc. will be *cleaner* (less long-context noise dilution). Acceptable to re-present updated showcase to lab as v0.4.1 ships?

**Post-implementation (1 week after v0.4.1 to PyPI)**:
- Lab demo with corrected numbers + validity framework UI
- Collect reactions to:
  - Out-of-range visual rendering — does it confuse users or clarify?
  - Whether NA cells for partial domains are sufficient, or whether users want explanatory inline copy
- Optional: cross-check on lab's other base models (Mistral-7B 8K, Llama-3-8B 8K) to confirm framework generalizes

This is the **first time** lmdiff explicitly puts methodology design out to lab review before implementation. Set precedent for future contested metrics (specialization metrics in v0.5.0+, representation metrics in Phase 5, trajectory metrics in Phase 6).

---

## Y.7 Carry-over architectural notes (added to W.7 / X.8)

Extending the carry-over list:

11. **Measurement validity must be a first-class GeoResult field, not implicit in metric values**. v0.3.x and v0.4.0 silently included out-of-context probes in metric calculations because no machinery existed to mark them invalid. Future metrics (representation distance, semantic entropy, etc.) will face the same issue — encode validity at the data-structure level so all metrics can respect it uniformly.

12. **Methodology decisions deserve published derivations alongside published numbers**. Lab critique on √T was right partly because √T had no derivation in docs. Future metric choices (specialization layers, representation metrics, calibration metrics in Phase 4) should ship with `docs/methodology/<metric>.md` containing the derivation, assumptions, and design alternatives considered. This is the publication-grade evidence that converts "trust us" into "here is why".

13. **External feedback channels need explicit hooks in the release cycle**. Lab caught the √T issue post-shipment because no pre-ship feedback hook existed. Add to Phase 7 (release polish): for each minor release, a designated "feedback collection window" between PyPI publish and the next implementation commits, with a structured ask. (See Y.6 for v0.4.1's instance of this pattern.)

14. **Self-consistent error pattern**: when mockup numbers and implementation numbers come from the same formula, downstream "validation" tests only catch implementation-vs-spec divergence, not spec-vs-truth divergence. v6 §13 calibration mockup was derived from Formula B; v0.3.2 implementation was Formula B; tests passed; nobody noticed the formula was wrong. Anti-pattern: never validate a metric against a mockup generated from the same code path it's supposed to validate. Validate against external derivation, theoretical expectation, or empirical data with predetermined success criteria. (This is the substance of L-033.)

---

## Y.8 LESSONS.md L-033 specification

To be added to LESSONS.md after Phase 2 commit 4.1 implementation kicks off (so the lesson cites the actual fix as concrete instance):

```
L-033 — Self-consistent ad-hoc fixes evade validation

Context: v0.3.2 PR #11 introduced the share_per_domain formula 
sqrt(Σδ²/ΣT) as an ad-hoc fix for "long-context domain over-dominates 
share" observed in pre-PR#11 raw L2 magnitude form. The formula has 
no derivation. v6 plan §13 calibration mockup numbers were derived 
from the same formula. v0.4.0 implementation produced numbers matching 
the mockup. Calibration regression tests passed. Lab demo showcase 
numbers (long → reasoning 66%, etc.) were published.

Failure mode: when mockup and implementation come from the same 
formula, validation only confirms implementation-matches-spec, 
not spec-matches-truth. The √T-correction was a happy accident 
that incidentally suppressed long-context dominance by dividing 
out the dimensional inflation — but for the wrong mechanism. The 
real cause (long-context probes are invalid measurements when 
base model context window is smaller than probe length) was 
invisible to the formula and to the testing regime.

Lab critique 三个字: "没什么根据" ("no basis") triggered audit that 
revealed: (a) dimensional inconsistency (δ already per-token, √T 
over-corrects), (b) self-consistent error in mockup-implementation 
pair, (c) deeper measurement validity issue underneath. Three rounds 
of user push-back during this conversation were necessary to reach 
the correct framing — each round reframing the previous wrong 
defense.

Generalizable rule: validate metrics against external ground truth, 
not against mockups generated from the same formula. Specifically:
- Theoretical derivation from published frameworks (Oyama, BPB, 
  perplexity lit) with explicit unit/assumption checks
- Empirical data with predetermined success criteria established 
  before formula design
- Cross-implementation comparison (e.g. lm-eval-harness equivalent 
  scoring path) when possible

When defending a metric in response to external critique, the first 
action is to read the implementation and trace units, not to search 
the literature for a justification. (Three Claude defenses in this 
conversation were wrong before the user's push-back forced reading 
the actual code.)

Source: 
- v0.3.2 PR #11 (lmdiff, share_per_domain formula)
- Lab feedback 2026-05-11 ("√T 没什么根据")
- Audit chain documented in v6 plan Update 5 Y.1-Y.4
- Fix: Phase 2 commit 4.1 (v0.4.1)
```

---

## Y.9 What this changes about the Phase 4 + Phase 5+ roadmap

Phase 4 (Output, behavioral, calibration metrics — §7) is unaffected. New metrics introduced there (output-quality metrics like rouge_diff, BLEU-diff, behavioral-fidelity probes, calibration ECE-diff) are computed at the probe level, not aggregated by domain — they inherit validity framework from commit 4.1 automatically.

Phase 5 (Representation + steering — §8) needs explicit validity treatment for representation distance metrics:
- Hidden state distance per-layer per-probe: same validity rules apply
- Steering tensor extraction: must verify token positions used for extraction are within valid context range
- This is straightforward extension of commit 4.1 framework, no design change

Phase 6 (Trajectory + cloud distance — §9) representation-level analyses need particular care for long-context: trajectory analysis spanning many checkpoints with mixed-context-window training stages may produce comparison artifacts that look like trajectory shifts but are actually validity transitions. Add as Phase 6 design concern.

Methodology paper (long-term, post-v0.5.0): the validity framework + corrected formula is a natural section. If lmdiff ever produces a publication, the v6 plan W.8 / X.8 / Y.7 carry-over architectural notes form the publication-grade evidence base — they record what was tried, what was rejected, and why current choices stand.

---

## Y.10 Process retrospective: three wrong Claude defenses

For the long-term record, this conversation had three distinct wrong defenses by Claude before reaching the right framing:

1. **Wrong defense #1** (Oyama-framework derivation for √T): based on wrong assumption about δ unit. Corrected by user push-back #1 (验证 actual code).

2. **Wrong defense #2** (Token-weighted RMS as the clean fix): based on assumption that long-context σ inflation is real signal that token-weighting handles correctly. Corrected by user push-back #2 (CC audit on demo data showed long-context still dominates).

3. **Wrong defense #3** (Path B: Formula C + specialization layer): based on assumption that all per-token δ values represent valid measurements. Corrected by user push-back #3 (base model collapses on long-context, signal is invalid not just biased).

Three wrong defenses, three corrections, before reaching the validity framework. This is unusually many — typical methodology critiques converge in one or two rounds of push-back. The root cause was Claude's tendency to **search for a normalization fix when the underlying question was measurement validity**, not normalization.

The user's three push-backs were each:
- specific and short (a sentence each, no lengthy preamble)
- evidence-grounded (referenced actual code behavior or actual data behavior)
- not deferential (pushed past Claude's first response despite Claude's defense being phrased confidently)

The pattern Claude needs to internalize: when defending a contested metric or method, the **first step** is always to read the implementation and trace units before searching the literature for justification. Literature comes second, derivation third. Code first. (L-033 codifies this.)

For future methodology discussions in this project, Claude should explicitly state at the start "I will read the relevant implementation before responding" rather than defaulting to literature search. The user's epistemic burden was unnecessarily large in this round.


---

# UPDATE 6 — 2026-05-11: v0.4.1 implementation record + v0.5.0 scoping

Written during the v0.4.1 ship cycle, after fixture regeneration and before PyPI publish. Sections Z.1 through Z.9, append-only.

Update 5 specified the v0.4.1 design. This update records what actually happened when it was built, one design change that the real data forced, and the v0.5.0 scope that accumulated as a result.

**Unlike Updates 1–5, this update also revises the plan body.** Sections 2, 13, 19, and 20 were edited in place. Rationale in Z.6.

---

## Z.1 The `min_valid_fraction` floor — a design change the fixtures forced

Update 5 Y.4 specified four domain states with this tie-break: a (variant, domain) pair is `variant_only` only when no probe is valid for both engines, and `out_of_range` only when no probe is valid for any engine. Everything else is `partial`.

The regenerated fixtures showed this rule producing a result nobody wanted. Of 100 `longbench_2wikimqa` probes, exactly **9** fall inside Llama-2-7B's 4096-token window — the left tail of a distribution running from 1202 to 19161 tokens. Because `n_both = 9 > 0` for every variant, every (variant, long-context) pair classified as `partial`, and `partial` contributes a share.

The resulting shares were computed from nine probes and sat alongside domains computed from a hundred:

| variant | long-context share under `partial` |
|---|---|
| temp_1.5 | **27.6%** |
| chat | 18.4% |
| yarn | 8.7% |
| code | 3.4% |
| long | 2.8% |
| system_prompt | 2.3% |
| math | 0.3% |

A 27.6% headline share resting on nine probes is the same failure v0.4.1 exists to prevent, in a quieter form. The nine survivors are also **systematically biased**: they are the shortest long-context probes, and a 1202-token passage does not test long-context capability. Small sample was the lesser problem; unrepresentative sample was the larger one.

**Resolution.** `compute_domain_status` gained a `min_valid_fraction` parameter, default 0.5. When the fraction of probes valid for both engines falls below the floor, the pair degrades: to `variant_only` if the variant alone clears the floor, otherwise to `out_of_range`. Either way `share_per_domain` and `pdn` become `None`.

`min_valid_fraction=0.0` reproduces pre-floor semantics exactly, verified by replaying `compute_domain_status` over the committed fixtures' own `probe_validity` records and matching the previous `domain_status` cell for cell.

**Effect on the fixtures:**

| variant | long-context status | valid-for-both / variant-only |
|---|---|---|
| yarn | `variant_only` | 9 / 91 |
| long | `variant_only` | 9 / 91 |
| code | `variant_only` | 9 / 80 |
| math | `out_of_range` | 9 / 0 |
| chat | `out_of_range` | 9 / 0 |
| temp_1.5 | `out_of_range` | 9 / 0 |
| system_prompt | `out_of_range` | 9 / 0 |

CodeLlama landing on `variant_only` was not predicted during review: its 16384-token window covers 80 of the 100 probes, clearing the floor on its own. The prediction said `out_of_range`. CC caught the error by computing from the fixture rather than reasoning from the review comment.

**The floor value is arbitrary.** So is the alternative. Not having a floor is equivalent to setting it at `1/n`, which is both arbitrary and invisible. 0.5 is defensible as "a majority of probes must be measurable" and is exposed as a parameter rather than buried as a constant. `docs/methodology/normalization.md` says exactly this.

**Side effect worth noting.** Before the floor, no fixture cell was `variant_only` or `out_of_range`, so the entire `None` code path — serialization, v6 loader, hatched rendering, the None-aware peak lookup added mid-implementation — had zero integration coverage. The GPU calibration gate exercised none of it. The floor made both branches reachable and closed a gap that was exactly the L-030 shape: the calibration set failing to span the code paths the release introduces.

---

## Z.2 Final v0.4.1 numbers

Both fixtures, 409 valid probes (100 each for code / commonsense / math / reasoning, 9 for long-context):

| variant | code | commonsense | long-context | math | reasoning |
|---|---|---|---|---|---|
| yarn | 7.5% | **78.7%** | — | 11.0% | 2.8% |
| long | 18.3% | 7.7% | — | 27.2% | **46.9%** |
| code | **52.1%** | 13.4% | — | 28.6% | 5.9% |
| math | 27.4% | 16.5% | — | **42.6%** | 13.4% |
| chat | 29.5% | 23.0% | — | **32.0%** | 15.5% |
| temp_1.5 | 23.8% | **30.8%** | — | 18.7% | 26.7% |
| system_prompt | 10.5% | **62.0%** | — | 14.5% | 12.9% |

Against the v0.4.0 published numbers:

| variant | v0.4.0 biggest | v0.4.1 biggest | reading |
|---|---|---|---|
| code (CodeLlama) | code 32% | **code 52.1%** | sharpened — the domain the model was trained for |
| math (llemma) | math 35% | **math 42.6%** | sharpened, same reason |
| system_prompt | commonsense 60% | commonsense 62.0% | stable |
| yarn | commonsense 51% | commonsense 78.7% | sharpened |
| long | reasoning 66% | reasoning 46.9% | same domain, less concentrated |
| chat | reasoning 30% | math 32.0% | **changed** — see below |
| temp_1.5 | reasoning 34% | commonsense 30.8% | **changed** — sampling variance |

The two strongest signals are CodeLlama → code and llemma → math, and both are the domain the model was explicitly trained for. Both strengthened once the phantom long-context share stopped diluting them. This is the clearest evidence the v0.4.1 numbers are better than the v0.4.0 ones.

**Two variants are undifferentiated and must not be described as specialized.**

| variant | top three | gap |
|---|---|---|
| chat | math 32.0%, code 29.5%, commonsense 23.0% | **2.5pp** |
| temp_1.5 | commonsense 30.8%, reasoning 26.7%, code 23.8% | **4.1pp** |

Neither gap supports a ranking. For `chat`, the honest reading is "no single dominant domain, consistent with instruction tuning touching reasoning and math together" — not "chat specializes in math." For `temp_1.5` the case is stronger still: it is the only sample-decode variant, the calibration test allows it a 2pp share tolerance for exactly that reason, and a 4.1pp lead sits at roughly twice that tolerance. Its peak could flip on a reseed. "temp_1.5 specializes in commonsense" is as unsupported as the chat claim.

The other five variants have gaps of 15.3–67.7pp and are safe to describe as specialized. Any presentation that reports `argmax(share)` without a spread check will misrepresent chat and temp_1.5.

**The reasoning column falls for six of seven variants, and that is arithmetic, not signal.** Renormalizing over four domains instead of five costs the largest pre-floor share the most in absolute terms, and `reasoning` was the largest term for several variants. `code` gains everywhere for the mirror-image reason. Do not read this as "v0.4.1 shows variants drifting less on reasoning" — the per-domain magnitudes tell that story, the shares do not.

The 4-variant and 7-variant fixtures produce **identical values** for the four shared variants — max difference `0.000e+00` on `change_vectors`, `share`, and `pdn` — which cross-validates the pipeline across two independent GPU runs.

---

## Z.3 What a `—` in the long-context column means

This will be misread, and the misreading is the exact opposite of the truth.

`yarn` shows `—` for long-context. That does **not** mean Yarn-128K is unremarkable on long context. Yarn scored **100 of 100** long-context probes successfully. So did LLaMA-2-32K. CodeLlama scored 89. The base — Llama-2-7B, 4096 tokens — scored 9.

δ is a difference. It needs both sides. When the base cannot score a probe, there is no baseline to difference against, however well the variant performed. `variant_only` names precisely this situation: the variant measured it, the base could not, so no comparison exists.

Two states currently share the `—` and the same hatching:

- **`variant_only`** (yarn, long, code): the variant produced valid scores on the majority of probes. The data exists. It has nowhere to go in v0.4.1 because there is no base-relative frame for it. This is a promise, not a dead end.
- **`out_of_range`** (math, chat, temp_1.5, system_prompt): these variants share the base's 4096-token window. Nobody measured anything.

Update 5 Y.4 decision Q9.4 gave both the same `xxxx` hatch, on the reasoning that v0.4.1 has no variant-only data to display. That was right for v0.4.1 and is wrong for v0.5.0 — see Z.5.

**The underlying issue is a probe-set/base mismatch**, not an lmdiff defect. `longbench_2wikimqa` averages ~9000 tokens; any 4K-context base is the wrong base for it. Running the same family against a long-context base (Llama-3.1-8B at 128K, say) lights the column up with no code change. This belongs in Phase 2 commit 4.3 (builtin probe sets), which should decide whether a probe set declares a `required_context_window` and whether `family()` warns at startup when the base cannot satisfy it.

---

## Z.4 v0.5.0 scope

v0.5.0 sits between Phase 2 and Phase 3 and is not itself a phase. Four workstreams, the first three tightly coupled:

**1. `variant_only_metrics` population.** The field ships as a `None` stub in v0.4.1 (Q9.3). The 91 long-context probes that yarn, long, and CodeLlama all scored successfully are currently discarded. They cannot support base-relative metrics, but they can support **variant-versus-variant** ones:

```
variant_only_metrics["yarn"]["long-context"] = {
    "vs_long": <distance to LLaMA-2-32K over the 100 shared probes>,
    "vs_code": <distance to CodeLlama over the 89 overlapping probes>,
    "n_probes": 100,
}
```

This is where "yarn is a long-context model" becomes visible — through comparison against other long-context models rather than through a fabricated number against a 4K base.

Open design questions: which distance (cosine on δ requires a base, so probably raw per-token CE difference between the two variants); how to handle partial overlap when two variants have different context windows; whether this appears in the main share table or a separate section.

**2. Visual separation of `variant_only` from `out_of_range`.** Once (1) gives `variant_only` cells actual content, they should stop looking identical to empty ones. **Use colour, not hatching** — hatching is already carrying the partial/invalid distinction and adding a third pattern would be unreadable at cell size. A muted accent fill for `variant_only` (data exists, not base-comparable) against neutral grey for `out_of_range` (nothing measured) reads correctly at a glance and survives greyscale printing as a lightness difference. The legend gains a fourth entry.

**3. `effective_context_length()` and the `degraded` state.** Sliding-window models (Mistral-7B and successors) report `max_position_embeddings = 32768` while each layer attends over only 4096 tokens; long-range information survives through multi-layer indirection with quality that decays over distance. v0.4.1 trusts `max_position_embeddings` (Q9.9).

**A correction to how Update 5 framed this.** The audit treated sliding-window decay as noise to be filtered, in the same category as Llama-2's out-of-window collapse. That is wrong, and the distinction matters for the v0.5.0 design:

- Llama-2 beyond 4096 extrapolates RoPE into a position range it never trained on. Attention patterns fail. The output is a numerical artifact, not the model's behaviour. There is nothing to measure.
- Mistral at 20000 tokens is performing normal inference with attenuated long-range information. The output *is* the model's behaviour. Attenuation is a property of the model, and a variant that changes the attention mechanism has changed its behaviour in a way lmdiff should report — this is configuration-is-the-unit applied to attention.

So `degraded` is **an annotation, not an exclusion**. The share is computed and displayed as normal; the flag tells the reader that part of the difference in this cell originates in attention mechanics rather than weights. A design that filtered `degraded` cells out would be discarding real signal.

**4. v0.2.x removals.** `ModelDiff`, `run_family_experiment`, `InferenceEngine`, `ChangeGeometry`, `lmdiff.config.Config`. All have carried `DeprecationWarning` since v0.4.0, giving users a full minor cycle of notice (Update 4 X.7).

**5. Loader hygiene.** Replace the seven enumerated version gates in
`geo_result_from_json_dict` with range comparisons (Z.5). Small, and it
belongs in a release already touching the schema. Its current failure
mode is silence: a forgotten gate drops a field rather than raising.

Estimated 2–3 weeks. Items 1–3 are one coherent theme — "make variant-only measurement first-class" — and should ship together.

---

## Z.5 Deferred items and where they went

For traceability, everything consciously pushed out of v0.4.1:

| Item | Deferred to | Reason |
|---|---|---|
| Visual separation of `variant_only` / `out_of_range` | v0.5.0 | Pointless until `variant_only` cells carry data |
| `variant_only_metrics` population | v0.5.0 | Requires a variant-vs-variant distance design |
| `effective_context_length()` + `degraded` | v0.5.0 | Needs a fifth state and share-display rules |
| `min_valid_fraction` on the public API | **superseded — lands in v0.4.3** | Deferred on the grounds that no user had asked. The run-config schema is now asking: AA.3 requires every value-affecting parameter to be written explicitly, and this is the parameter whose default most recently changed meaning (implicit `1/n` → explicit `0.5`), which is what decides whether long-context reads 27.6 % or `—`. See `v043_runconfig_design.md` §10.6 |
| Single-domain fallback still using the old formula | **closed in v0.4.2** | Was Formula B's `‖δ‖/√(n·mean_tokens)`, so single-domain runs sat on a different scale from every multi-domain run. Survived the Q9.10 correction because no shipped experiment is single-domain — which is also why it was safe to defer |
| `tests/test_*.py` → `tests/unit/` migration | **closed in v0.4.2** | 27 files moved; verified as a pure move (1287 collected before and after). The L-034 trap is measurably closed — `unit` + `integration` now sum to the whole tree, so no test lives outside one of the two |
| Method/field duplication audit across other `per_X` patterns | **closed in v0.4.2** | Two instances. The pdn field/method pair — agreement now pinned by test rather than collapsed, since the method serves the v0.2.x `ChangeGeometry` path (see the v0.5.0 row below). And `change_size`/`normalization_effect`, near-duplicates since before v0.4.1; the latter is deprecated |
| Replace the loader's enumerated version gates with range comparisons | v0.5.0 | `geo_result_from_json_dict` has **seven** gates of the form `sv in ("5", "6", "7")`, each meaning "this version and every later one". Every schema bump must edit all seven, and **omitting one does not raise — it silently drops the field that gate protects.** The 6 → 7 bump hit exactly this: `domain_status` deserialized as `{}` while every version-pin test passed, because those tests assert the version string rather than the payload. Replace with `int(sv) >= n`. **Until that lands, a schema bump must edit every gate and add a payload-level regression test, not only a version-string one.** Same shape as L-037 — a change pushing values past a constant defined elsewhere, with the assertions aimed at the wrong layer |
| `lmdiff validate-result --base-model` CLI helper | unscheduled | Would let legacy saves be re-classified with the correct context window |
| Probe sets declaring `required_context_window` | Phase 2 commit 4.3 | Properly a probe-taxonomy concern |

---

## Z.6 Why the plan body was edited this time

Updates 1–5 were append-only. That rule made sense while the body still described unbuilt work: the body was intent, the updates were record, and the gap between them was the story.

It stopped making sense once the body began stating things that are **false about shipped software**. Three sections crossed that line:

- **§2 phase summary** listed Phase 1 as "Pending" after three releases had shipped from it, and mapped Phase 3 to v0.5.0 after v0.5.0 had been claimed for other work.
- **§13 CLI summary example** displays numbers no version of lmdiff produces. Read cold, it looks like a spec the implementation is violating.
- **§19 time budget** and **§20 immediate next steps** described a schedule and a task list that were both complete.

The distinction now applied: **intent stays, claims get corrected.** §13's layout specification, layer structure, and rendering rules are intent — they were implemented as written and remain authoritative. Its numbers are a claim about output, and that claim is now false, so it carries a correction header pointing at the current fixture. The section is not rewritten, because Update 5 Y.1 depends on those exact numbers as evidence of the self-consistent-error pattern; deleting them would erase the record of how the mistake was found.

§13 also earns a permanent lesson. Numbers in a planning document age badly and silently — nobody diffs a plan against a fixture. Future plan revisions should show shapes and layouts, and reference `tests/fixtures/` for values.

---

## Z.7 Implementation record for commit 4.1

Eleven commits on `feat/v0.4.1-validity`, PR #19.

| # | Subject |
|---|---|
| 1 | `max_context_length()` on the Engine Protocol + HFEngine / MinimalEngine / MockEngine |
| 2 | `ProbeValidity` / `EngineValidity` / `compute_domain_status` |
| 3 | Per-probe validity wiring in `_delta_for_variant` |
| 4 | Formula A + None-aware share |
| 5 | Schema v6 + v5 loader preserving saved values |
| 6 | Hatched rendering for invalid cells |
| 7 | `normalization.md`, migration guide, CHANGELOG, L-033 |
| 8 | Fixture regeneration scripts + spec rename |
| — | None-aware peak lookup in markdown / terminal (review follow-up) |
| — | Schema version sync across the full test tree (CI catch) |
| — | `magnitudes_per_task_normalized()` switched to Formula A |
| — | Case-fold CLI help assertions (typer 0.27 metavar rendering) |
| — | `min_valid_fraction` floor (Z.1) |

Unit tests: 933 at the v0.4.0 baseline → **1001**.

**Five latent bugs caught before merge**, against four caught after merge in v0.4.0:

| Found by | What |
|---|---|
| Follow-up grep | `_findings.py` peak lookup crashed on `None` |
| Follow-up grep | `markdown.py` / `terminal.py` `max(..., key=...)` crashed on `None` |
| Test failure trace | `magnitudes_per_task_normalized()` still computed the old formula — two figures and the report table would have disagreed with `drift_share_dual.png` |
| **CI** | Six schema-version assertions in `tests/test_*.py`, invisible to `pytest tests/unit/` |
| **Regenerated fixture** | `partial` firing on 9/100 (Z.1) |

The last two are the interesting ones. Neither was reachable by inspection — one required running the full test tree, the other required real numbers from a GPU run. Design review and unit tests could not have found either.

**The method/field bug deserves emphasis.** `magnitudes_per_domain_normalized` (a field) and `magnitudes_per_task_normalized()` (a method) had independent implementations of the same concept, inherited from v0.2.x when "task" and "domain" were used interchangeably. Commit 4 updated the field. The method kept computing the old formula, and it feeds `normalized_magnitude.png`, `specialization.png`, the specialization z-score, and the report tables. Shipping that would have put two different numbers for the same metric in the same release — `drift_share_dual.png` disagreeing with `specialization.png`. The fix makes the method delegate to the field so no future formula change can diverge again.

---

## Z.8 Lessons for LESSONS.md

To be added after PyPI publish, when citation URLs stabilise.

**L-034 — Verification scope must equal CI scope.** CC ran `pytest tests/unit/` (502 tests) as its per-commit gate and reported green seven times. CI runs `pytest tests/` (984) and found six failures in top-level `tests/test_*.py` files that predate the `tests/unit/` directory. Roughly half the suite went unverified for the whole implementation. Both the instruction and the execution assumed `tests/unit/` was the canonical location. Before claiming tests pass, run the command CI runs.

**L-035 — Similar names hide duplicate implementations.** `magnitudes_per_domain_normalized` (field) and `magnitudes_per_task_normalized()` (method) each implemented the per-domain formula. A grep for `magnitudes_per_domain` found one; a grep for the computation shape (`sqrt`, `sum`, `avg_tokens`) would have found both. When changing a formula, search by what the code does, not by what it is called — then remove the duplication so the next change has one edit point.

**L-036 — A partial-validity state needs a floor.** v0.4.1's four-state classification let `partial` fire on 9 valid probes out of 100, producing a 27.6% share for one variant computed from those nine. The state was correct by its own definition and wrong by intent. Any "some data is valid" classification needs a threshold below which it degrades to "not measurable", and the threshold must be a documented parameter rather than an implicit `1/n`. Small samples were the lesser issue; the survivors being the unrepresentative tail of the distribution was the larger one.

**L-037 — A correctness fix can push values across thresholds elsewhere in the system.** Adding the `min_valid_fraction` floor was strictly a correction: it stopped a nine-probe subset from contributing a share. But dropping the long-context column renormalized every remaining share upward, and `chat` moved from 29.9% to 32.0% — across the 30% gate on `SpecializationPeakFinding`. That gate had been silently protecting the variant with the second-weakest specialization in the set. The fix therefore *manufactured* a claim that "chat specializes in math," on a 2.5pp lead, in the release whose entire thesis is not overstating what the data supports.

Nothing detected this. The 140 calibration assertions passed, `pytest tests/` passed, and the numeric change was correct at every step. It surfaced only when the showcase figure was re-rendered and read by eye.

The generalizable rule: after any change that rescales or renormalizes a reported quantity, enumerate every threshold downstream of it — display bins, finding gates, warning triggers, colour scales — and re-derive rather than assume. In this release the same change invalidated the figure's magnitude bins (a 10× rescale left 93% of cells in the top bin) and the specialization gate, and neither had a test that could have caught it, because both are thresholds on presentation rather than on values.

**L-033 recurred inside this cycle, one layer up.** Update 5 Y.4 specified the `partial` tie-break, the implementation matched the specification, and the unit tests passed — but the code agreeing with the design document proved nothing, because the design document is what defined the code. This is the same self-consistent-pair failure as the v0.3.2 mockup-and-formula case, relocated from "mockup vs implementation" to "design doc vs implementation." It took real fixture data to break the loop, exactly as it did the first time. Whichever of L-034 through L-037 lands first should cross-reference this: agreement between any two artifacts derived from the same source is not validation, regardless of how far apart in the process they sit.

---

## Z.9 Carry-over architectural notes

Extending the lists in W.7, X.8, and Y.7:

15. **Any classification with a "partial" state needs an explicit threshold.** Generalises L-036. This will recur: representation metrics (Phase 5) will have probes where hidden states are extractable for some layers only; trajectory metrics (Phase 6) will have checkpoints where some metrics are computable. Each needs a documented floor, not an implicit one.

16. **"Cannot measure" is a first-class result, not an error.** The `—` cells are informative output: they say the base cannot assess this domain, which is a fact about the experiment worth reporting. Every metric family added from here should have a defined not-measurable state rather than a fallback value. Fallback values are how the √T problem started.

17. **Prefer colour over pattern for a second categorical axis in figures.** Hatching carries validity in v0.4.1. Adding a third pattern for a second axis (Z.4 item 2) would be unreadable at cell size. Colour and pattern are independent channels; use one per axis. Both must survive greyscale — colour as a lightness difference, pattern as itself.

18. **Probe sets and base models have a compatibility relation that nothing currently checks.** A 9000-token probe set against a 4096-token base yields nine usable probes. `family()` should be able to warn about this at startup rather than after a GPU run. Belongs in Phase 2 commit 4.3 alongside `required_context_window` on probe-set definitions.

19. **Two independent fixtures scoring identically is worth the cost.** The 4-variant and 7-variant runs produce identical numbers for their four shared variants. That agreement is a free cross-validation of the pipeline across separate GPU runs, and it caught nothing this time only because nothing was wrong. Keep the overlap when designing future calibration sets.

20. **`argmax` over a near-uniform distribution is not a finding.** Two of seven v0.4.1 variants have top-two gaps under 5pp (Z.2), one of them a sample-decode variant whose own test tolerance is 2pp. v0.4.1 addresses this with a 5pp margin gate on `SpecializationPeakFinding` plus an explicit `UndifferentiatedFinding` so a flat variant reports "no dominant domain" rather than falling silent. Every ranked output the project adds from here needs the same treatment — a maximum without a spread check is not a result.

21. **Presentation thresholds are untested code.** Display bins, finding gates, colour cutoffs, warning triggers. v0.4.1's formula change invalidated two of them simultaneously — the figure's magnitude ladder (93% of cells in the top bin) and the specialization gate (L-037) — and the full 1001-test suite plus 140 calibration assertions caught neither, because both are thresholds on how values are shown rather than on the values themselves. They surfaced by looking at a rendered PNG. Either give them tests that assert distribution shape rather than exact numbers, or derive them from the same constants the values come from so a unit change propagates automatically. v0.4.1 did the latter for the bin ladder — one tuple now feeds the `BoundaryNorm`, the legend text, and the white-text cutoff, which had previously been three independent copies of the same number.

---

# UPDATE 7 — 2026-05-13: Run configuration as a serializable artifact

External suggestion, adopted. Sections AA.1 through AA.8, append-only.

The proposal: allow a YAML configuration file as an intermediate form between the Python API and the CLI, and attach that YAML to every generated report — including the versions of lmdiff and its dependencies, so a result can be pinned. Both entry points produce it; either can consume it.

---

## AA.1 What this actually solves

lmdiff currently has two entry points expressing the same experiment, and a report that is a container for numbers rather than a record of an experiment. Reading `CodeLlama → code 52.1%` in a report, there is no way to recover the call that produced it: which variants, which probe set, what `n_probes`, which seed, whether `min_valid_fraction` was defaulted or overridden.

Attaching the configuration makes a report **self-contained**. This is the same commitment v0.4.1 made about measurement, applied to provenance: v0.4.1 made the report state honestly what could not be measured; this makes it state honestly how the measurement was performed.

The dependency pinning matters more than it first appears. Between v0.3.2 and v0.4.1 the semantics of `share_per_domain` changed twice — once by the PR #11 formula (Update 5 Y.2) and once by the validity framework plus Formula A. "Why do I get different numbers than you" has had no mechanical answer until now. `lmdiff: 0.4.2` in the artifact is that answer.

---

## AA.2 The controlling constraint: the emitted YAML is runnable

The obvious design — a terse input format plus a complete state snapshot on output — produces an output artifact that **cannot be fed back in**. It would mix executable configuration with runtime facts (devices used, wall time, probes excluded), and a loader would not know which keys to honour and which to ignore.

That fails the actual use case. Someone reading a report and wanting to reproduce it, or to vary one thing, will copy the attached YAML, edit two lines, and run it. That path has to work.

**So there is one schema, and the emitted artifact is a valid input.** Runtime facts live in a clearly delimited block that the loader ignores:

```yaml
lmdiff_schema: 1

# ---- executable configuration ----
base: meta-llama/Llama-2-7b-hf
variants:
  yarn: NousResearch/Yarn-Llama-2-7b-128k
  code: codellama/CodeLlama-7b-hf
  temp_1.5:
    model: meta-llama/Llama-2-7b-hf
    decode: {strategy: sample, temperature: 1.5}
probes: lm_eval:hellaswag+arc_challenge+gsm8k
n_probes: 100
seed: 42
min_valid_fraction: 0.5

# ---- provenance: read-only, ignored on load ----
provenance:
  lmdiff: 0.4.2
  transformers: 4.51.0
  torch: 2.6.0
  run_id: 2026-05-13T09:14:22Z
  duration_s: 2514
  devices: [NVIDIA RTX A6000, NVIDIA RTX A6000]
  probes_excluded: 91
  domain_status_summary:
    long-context: {variant_only: 3, out_of_range: 4}
```

Copy, add a variant, run. The `provenance` block is dropped on load.

---

## AA.3 Two consequences of AA.2

**Defaults must be expanded on emission.** If `min_valid_fraction` is omitted because it was defaulted, and a later version changes that default, the same YAML silently produces different numbers on a different version. Every parameter that affects a computed value is written explicitly, defaulted or not. This project has already changed a numeric default's effective meaning twice inside two minor versions; pinning by omission does not work.

Corollary: emitted YAML is verbose and that is correct. Hand-written input YAML may omit anything with a default.

**Version mismatch warns, does not block.** A loader running `provenance.lmdiff: 0.4.2` under 0.5.0 should say so — numbers may differ — and proceed. Re-running an old configuration on a new version is a legitimate and common intent; silently doing it is not.

---

## AA.4 The serialization is one-directional, not a third API

`Config → YAML → Config` must round-trip to identity. YAML is a serialization of the existing configuration object, not an independent surface.

Stated because the failure mode is known and this codebase has hit it twice: `magnitudes_per_task_normalized()` diverging from `magnitudes_per_domain_normalized` (L-035), and the drift-magnitude bin edges living in both the `BoundaryNorm` and the legend literals. A YAML schema maintained separately from `Config` grows fields the Python and CLI paths do not have, and then drifts. The round-trip identity test is the guard, and it belongs in the first commit.

---

## AA.5 The escape hatch, added now rather than at 1.0

Python will eventually accept things YAML cannot express — an in-memory HuggingFace model instance rather than a resolvable identifier. That expansion is post-1.0 work and is not being designed here. But **the field that declares a configuration unrunnable is added now**:

```yaml
reproducible: false
non_serializable:
  - path: variants.custom_ft
    reason: in-memory model object, not a resolvable identifier
```

This lives in the executable section, not `provenance`, because it determines whether the file can be run at all. A loader encountering `reproducible: false` raises and names the offending path rather than pretending to execute.

Cost now: a few lines. Cost later: a schema version bump, and the loader-compatibility burden of one of those is fresh evidence — v0.4.1 went from schema 5 to 6 and carried a full preserving-loader path for the old version.

---

## AA.6 Emission ships before execution

Two separable capabilities, deliberately sequenced:

**Emission** is purely additive. Reports gain an attached artifact; nothing existing changes behaviour. It exercises the schema against every configuration the project actually produces, and puts it in front of readers who will find its gaps.

**Execution** needs a loader, validation, actionable error messages, path resolution, and answers to questions like what a model identifier means when it names a local path on a machine that is not this one.

Shipping emission first lets the schema be calibrated by real reports before anything depends on parsing it. The round-trip identity test from AA.4 belongs with emission, since it needs both directions internally even while only one is public.

---

## AA.7 Placement in the plan

This is Phase 2 work and **precedes commit 4.2 (probe taxonomy)**. Taxonomy introduces a `task_type` field, which the schema must carry; defining the schema first and adding a field to it is cheaper than retrofitting.

Revised Phase 2 ordering:

| Commit | Subject | Version |
|---|---|---|
| 4.0 | Backend cutover | v0.4.0 ✅ |
| 4.1 | Validity framework + pdn correction | v0.4.1 ✅ |
| — | Presentation-layer sweep + housekeeping | v0.4.2 (in flight) |
| **4.2 (NEW)** | **Run-config schema + emission** | **v0.4.3** |
| 4.3 (was 4.2) | Probe taxonomy (`task_type`) | v0.4.4 |
| 4.4 (was 4.3) | Builtin task probe sets | v0.4.5 |
| 4.5 (was 4.4) | YAML probe set loader | v0.4.6 |
| 4.6 (was 4.5) | Task-type-aware metric registry | v0.4.7 |

Run-config **execution** slots after 4.3 or later — its schema needs `task_type` settled first.

Note the relationship to commit 4.5, the YAML probe-set loader from §5.3. That one serialises a probe set; this one serialises an entire experiment and will reference probe sets by identifier. They should share a YAML dialect and loader conventions, and 4.5's design should be revisited once this schema exists rather than proceeding from §5.3 as written.

---

## AA.8 Open questions for the design audit

Not decided here. The commit gets the same audit-then-implement treatment as 4.1 (Update 5 Y.4, and the process record in Update 6 Z.7 for why that was worth it).

1. **Where does the artifact attach?** Sidecar file next to the report, embedded block inside the HTML/markdown, a field in the GeoResult JSON, or several? The JSON already carries a `metadata` dict; overlap needs resolving rather than duplicating.

2. **What identifies a probe set?** `lm_eval:hellaswag+arc_challenge` is a string today. If commit 4.5 introduces user-defined YAML probe sets, a run config referencing one must pin it — by path, by content hash, or by inlining.

3. **How much of `provenance` is worth recording?** Dependency versions are clearly worth it. Devices, wall time, and probe-exclusion counts are diagnostics that duplicate what the GeoResult already holds. Err toward less; the block is not the report.

4. **Does the CLI gain a `--config` flag, or a subcommand?** Affects nothing until execution ships, but the surface should be decided alongside the schema.

5. **Schema versioning.** `lmdiff_schema: 1` is independent of the GeoResult schema version (currently 6) and of the package version. Three version numbers is one too many if they can be collapsed; it is one too few if they cannot.