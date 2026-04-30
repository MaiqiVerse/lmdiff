# Changelog

## [0.3.2] - 2026-04-30

### Fixed
- **`share_per_domain` and overall `magnitudes_normalized` recomputed using per-domain per-token normalization** — the v0.3.0–v0.3.2 (pre-fix) formulas were length-weighted: `share_per_domain[v][d] = ‖δ_{v|d}‖² / Σ_d' ‖δ_{v|d'}‖²` and `magnitudes_normalized[v] = ‖δ_v‖ / sqrt(n_probes·mean_T)`. On a 5-domain run including longbench (~9000 tokens vs ~30 for MCQ), the long-context domain dominated 90–99 % of every variant's share even when its per-token drift was modest, and the overall normalized magnitude was biased toward the long-prompt domain. The corrected formulas match what the figure renderers already compute per-domain (and what the v6 §13 calibration mockup documents):
    - `magnitudes_per_domain_normalized[v][d] = sqrt( Σ_{i∈d} δ[v][i]² / Σ_{i∈d} T[i] )` — per-token RMS within each domain (new top-level field).
    - `share_per_domain[v][d] = pdn[v][d]² / Σ_d' pdn[v][d']²` — relative per-token energy across domains.
    - `magnitudes_normalized[v] = sqrt( mean over d of pdn[v][d]² )` — per-domain RMS, each domain weighted equally.

  v0.3.0–v0.3.2 GeoResult JSONs (which lack the new `magnitudes_per_domain_normalized` field) **auto-recompute on load** via `lmdiff.load_result` / `geo_result_from_json_dict`, emitting a single `DeprecationWarning` per file. Re-save with `result.save(path)` to upgrade. Long-context-heavy probe sets see substantially different shares — this is the corrected behavior, matching the v6 §13 documented calibration. Raw `magnitudes` (untouched name, untouched semantics) remains the unmodified L2 norm for users who want the length-weighted view.
- **OOM in multi-variant `family()` runs from duplicate model loads** — the same 7-variant Llama-2 demo that surfaced the per-task `n_probes` bug also OOMed on 2 × 48 GiB A6000s. Two compounding causes:
    1. ``_api.compare()`` / ``_api.family()`` eagerly built one ``HFEngine`` per Config (1 base + N variants) for capability checking, but the geometry path immediately threw them away and re-loaded each model as ``InferenceEngine``. Every weights file went through the GPU twice. Fix: skip the eager HFEngine preflight when ``engine=None`` (the default path); the real engine that runs inference does its own contract enforcement.
    2. Variants whose Config differed from base only in runtime-only fields (e.g. ``temp_1.5`` with a different ``decode``, ``system_prompt`` with a different prompt) loaded a full extra copy of the same weights instead of sharing base's loaded engine. Fix: variants sharing ``model`` and differing only in fields from ``RUNTIME_ONLY_FIELDS`` (``name``, ``system_prompt``, ``icl_examples``, ``context``, ``decode``, ``tokenizer_id_override``, ``capabilities_required``, ``training_recipe_summary``) now share a single loaded ``InferenceEngine``. Variants with weight-affecting differences (``adapter`` / ``quantization`` / ``pruning`` / ``soft_prompts`` / ``kv_cache_compression`` / ``steering``) still get their own engine. See the audit comment at the top of ``lmdiff/_config.py``.
- **Variant engines now released aggressively in ``ChangeGeometry.analyze``** — after each variant's ``_delta_for_variant`` completes, we look ahead by one. If the next variant in iteration order doesn't reuse this engine (different anchor), the engine is freed immediately (drop ref + ``gc.collect`` + ``torch.cuda.empty_cache``). If a later variant ends up needing the same engine again, we re-load the weights from disk. Memory savings beats reload time — re-loading 7B weights from disk is ~2 seconds; running OOM costs hours. The base engine is the one exception: held for the entire ``family()`` lifetime. Combined with the engine-reuse fix above, peak active engine count drops from 7 to 2 (base + 1 active variant) for the user's 7-variant pattern, and total weight loads drop from 16 (8 zombie HFEngines + 8 InferenceEngines) to 6 (one per unique runtime-incompatible Config).
- **`n_probes=` is now per-task for multi-task `lm_eval:` strings** — surfaced during the v0.3.0 7-variant Llama-2 demo. With `probes="lm_eval:hellaswag+arc_challenge+gsm8k+mmlu_college_computer_science+longbench_2wikimqa", n_probes=100`, all 100 probes came from the first task (`hellaswag`/`commonsense`), making a multi-domain comparison impossible. Root cause: `_coerce_to_probe_set` called `from_lm_eval(t)` per task without `limit=`, concatenated all task probes, then `compare()` sliced the merged set with `[:n_probes]` — head-of-list slicing of a concatenation only keeps the first task. The 5-task spec above now loads 500 probes (100 per task), matching the v0.2.x calibration convention. Flat probe sets (`"v01"`, `ProbeSet` instances) keep the v0.3.0 "total" semantics. New additive `GeoResult.metadata` keys: `n_probes_per_task`, `task_breakdown`, `tasks` (no schema bump). Asymmetry is documented in `compare()` / `family()`.
- **`render_direction` layout now scales with N variants** — the v0.3.0 layout was hardcoded to 14×7 inches with per-cell text labels positioned for ≤4 variants. At N=7 the x-tick labels overlapped (`system_prompt` running into `temp_1.5`) and per-cell sub-labels collided with the cosine number. Figure size now scales (~1.6"/cell, capped at 10"), x-ticks rotate 30° when names are long or N>5, and the per-cell sub-label is dropped at N>4 (the cell color band already encodes the bucket). The 4-variant path is unchanged.
- **`render_change_size` narrative is data-driven** — the v0.3.0 figure unconditionally claimed "Longbench probes are 100× longer than other tasks" in the bottom-line panel and "Hatched portion = share dominated by long-context probes" in the subtitle, even when `result.probe_domains` contained zero `long-context` probes (the v0.3.0 demo's `commonsense`-only run hit this). The narrative is now gated on `mask_long.any()`. When long-context probes are present, the existing wording renders with the actual `long_context_domain` parameter substituted (the hardcoded `"longbench"` string is gone). Otherwise a generic "raw vs per-token" caveat replaces it.
- **"How big is each move" report section now shows per-√token-normalized magnitude** in markdown / terminal / HTML. The previous "total" column was an RMS-of-per-domain raw value — length-weighted and not comparable across runs. The right pane of `change_size_bars.png` already showed the normalized number; the table column is now consistent with it. Per-domain raw cells are kept (they line up with the figure's hatched left pane). Column header changes from `total` to `‖δ‖/√tok`.

### Changed
- ``InferenceEngine.score`` and ``InferenceEngine.generate`` now accept ``system_prompt=`` / ``context=`` (and ``generate`` also ``decode=``) as keyword-only kwargs that override the engine's stored config for the call. Required for the engine-reuse path: one engine instance can serve multiple Configs that differ only in runtime params, with the per-call config supplied at the call. Defaults are ``None`` (use ``self.config``), preserving backward compatibility for existing direct callers.

### Added
- ``Config.is_runtime_only_modification_of(other)`` — predicate for whether two Configs can share a loaded engine. ``True`` iff (a) they share ``model``, and (b) every weight-affecting field is identical. Documented field-by-field rationale in ``lmdiff/_config.py``.
- ``lmdiff._config.RUNTIME_ONLY_FIELDS`` — frozenset of Config field names that are safe to differ across two configs sharing one engine.
- ``lmdiff._config.MODEL_SPECIFIC_COMPARATORS`` — extension hook (``dict[str, Callable]`` keyed by model id) for downstream models that need custom reuse semantics. Empty by default; pure default behaviour unless explicitly populated.
- ``ChangeGeometry.analyze(engine_groups=...)`` — optional ``dict[str, str]`` mapping variant name to anchor name. Variants sharing an anchor share a loaded engine; the look-ahead-by-one release rule fires when an anchor is no longer needed. Built automatically by ``_api.compare()`` / ``_api.family()``; set to ``None`` (default) for legacy per-variant loading.
- ``GeoResult.magnitudes_per_domain_normalized`` — per-variant per-domain per-token RMS magnitude (`pdn`). Surfaces the value figure renderers were already computing as a top-level field; the corrected basis for `share_per_domain` and overall `magnitudes_normalized`. Empty when `probe_domains` or `avg_tokens_per_probe` is empty. Auto-recomputed on load when missing from the JSON. Additive — no schema bump (still v5).
- ``[lmdiff] loading weights: <model_id>`` line printed before each model load — visible in run logs alongside the transformers ``Loading checkpoint shards`` progress bar. Lets you count actual loads in a multi-variant run at a glance.
- ``LMDIFF_DEBUG_ENGINE_LIFECYCLE=1`` env var — emits structured ``[lmdiff lifecycle] InferenceEngine.init`` / ``engine_reuse`` / ``engine_release`` lines for diagnosing memory patterns in family runs. Off by default.

### Notes
- The variant name ``"__base__"`` is now reserved by lmdiff as the base-engine sentinel. ``family()`` raises ``ValueError`` if any variant is named ``"__base__"`` — extremely unlikely in real usage but documented for completeness.
- All 5 fixes are additive or rendering-only; no API breakage and no schema bump. Existing JSON results load and re-render correctly with the new code.
- Re-rendered the v0.3.0 demo's `family_geometry.json` through the v0.3.2 renderers (in `runs/v032-rerendered/`) to verify all four figure / report fixes. The mono-domain demo data can't show the multi-domain story — the user's GPU re-run with the new per-task `n_probes` semantics is what surfaces the real five-domain split.
- Tests: 5 new `TestNProbesLmEvalSemantics` cases (per-task limit, single-task = total, no double-truncation, flat-set unchanged, metadata propagation through `family()`); all monkey-patch `from_lm_eval` so lm-eval-harness is not a test dependency. 283 unit tests pass (was 278 → +5).

## [0.3.1] - 2026-05-01

### Fixed
- **`result.save("nested/dir/r.json")` no longer raises `FileNotFoundError`** — `lmdiff.report.json_report.write_json` and the module-level `render` function now `mkdir(parents=True, exist_ok=True)` before writing, matching the behavior of the HTML / markdown / figures channels. Reported during the v0.3.0 demo.
- **`InferenceEngine.device` is now anchored to `model.get_input_embeddings().weight.device` after load**, fixing `RuntimeError: Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on cpu` when running 4+ large variants in sequence. With `device_map="auto"` and memory pressure, accelerate would silently shard the embedding layer to CPU while `self.device` stayed `"cuda"`. Affects users of the deprecated `ModelDiff` and direct users of `InferenceEngine`. The v0.3.0 `compare()` / `family()` API path is also affected because v0.3.0 internals still wrap `InferenceEngine` (full backend cutover deferred to v0.4.0; see "Changed" below).

### Changed
- **`HFEngine.score()` tokenization now follows the lm-eval-harness convention** for byte-level equivalence with the v0.2.x `InferenceEngine.score()`:
  ```
  full_ids = tokenizer("", add_special_tokens=True)
             + tokenize(prompt, add_special_tokens=False)
             + tokenize(continuation, add_special_tokens=False)
  ```
  Previously HFEngine tokenized `prompt` and `prompt + continuation` jointly and took the diff for continuation tokens. SentencePiece tokenizers (Llama family) merge across the prompt/continuation boundary, producing different per-token logprob breakdowns. The new convention asks the tokenizer for its empty-prefix special tokens (Llama → `[BOS]`, GPT-2 → `[]`) and concatenates separately-tokenized prompt and continuation. **This is a behaviour change for direct users of `HFEngine.score()`** — per-token logprobs may shift slightly compared to v0.3.0.
- **`HFEngine.score()` signature gains `continuation_ids: list[int] | None = None`** — pre-tokenized continuation IDs are used as-is, skipping retokenization. Recommended for self-scoring (when the same engine that generated the continuation is now scoring it), avoiding decode→retokenize round-trip drift. Mirrors `InferenceEngine.score(continuation_ids=...)`. `continuation` and `continuation_ids` are mutually exclusive; both / neither raise `ValueError`.

### Notes
- API surface unchanged from v0.3.0; the `HFEngine.score()` signature gain is additive (the prior positional `score(prompt, continuation)` call shape is preserved).
- New equivalence test (`tests/integration/test_engine_equivalence.py`, marked `slow`) pins HFEngine ↔ InferenceEngine equivalence on CPU using `hf-internal-testing/tiny-random-gpt2`. Runs in ~7s. After this, the v0.4.0 backend cutover (route `compare()` / `family()` through `HFEngine` directly, deprecate `run_family_experiment`) can land with byte-identity against the v0.3.0 calibration baseline.
- Tests: 798 fast passed (was 794 in v0.3.0 → +4 fast); +6 slow tests for the defensive fixes and the equivalence pins. No regressions.

## [0.3.0] - 2026-04-30

### Added
- **Top-level `compare()` and `family()` functions** — the new public API. Replaces the v0.2.x `ModelDiff` class. Both accept either model-id strings (coerced to `Config(model=...)`) or pre-built `Config` instances; build engines internally and clean up in a `try/finally`. See `docs/migration/v02-to-v03.md`.
- **`Config` class** as the unit of comparison — frozen, hashable, validated at construction. Packages model + adapter + quantization + context + decoding + steering through 8 typed sub-specs (`AdapterSpec`, `QuantSpec`, `PruneSpec`, `ICLExample`, `Message`, `KVCacheSpec`, `DecodeSpec`, `SteeringSpec`).
- **`Engine` Protocol** for backend integration (`runtime_checkable`, PEP 544). Canonical implementation `HFEngine` (HF Transformers, lazy torch import, capability registry). `MinimalEngine` is the copy-paste template for custom backends; `MockEngine` is the test fixture used throughout `tests/unit/`. Reserved capability registry forward-compatible with v0.7+ representation, v2.0+ patching.
- **`GeoResult.findings`** — 8 data-driven `Finding` types (`MostLikeBaseFinding`, `BiggestMoveFinding`, `DirectionClusterFinding`, `DirectionOutlierFinding`, `SpecializationPeakFinding`, `AccuracyArtifactFinding`, `TokenizerMismatchFinding`, `BaseAccuracyMissingFinding`) extracted from any comparison. Single source of truth across renderers.
- **5-layer terminal renderer** (`result.print()`) with ANSI colors, adaptive width (< 80 cols collapses to per-variant blocks), `NO_COLOR` / non-tty / `force_color` precedence rules.
- **3 application-tier figures** (`result.figures(out_dir)`): drift+share dual-view heatmap, direction agreement (raw + selective cosine matrices), raw-vs-normalized magnitude bars. Following the v6 plan §12.1 5-rule template; outputs visually identical to the §14 reference scripts on the calibration data.
- **Self-contained HTML report** (`result.to_html("report.html")`) — single ~1 MB file with base64-embedded figures, light/dark theme toggle (defaults to OS preference), print stylesheet. `embed_images=False` switches to a small HTML + sibling `figs/` directory.
- **Markdown renderer** (`result.to_markdown("report.md")`) — GitHub-flavored, mirrors terminal 5-layer structure, bolds row peaks in tables, `⚠`-prefixed blockquote caveats, optional figure links via `figures_dir=` kwarg.
- **GeoResult schema v5** — adds `share_per_domain` field (per-variant per-domain energy share, rows sum to 1.0). v4 schemas load with `DeprecationWarning` and synthesise the new field on the fly so the in-memory result is always v5-shaped.
- **`lmdiff.load_result(path)`** for symmetry with `result.save(path)` — round-trips schema v1-v5 JSON.
- **CLI: `lmdiff family-experiment` + `lmdiff plot-geometry`** subcommands wrap the v0.3.0 entry points; comprehensive migration docs at `docs/migration/v02-to-v03.md`.

### Deprecated
- `lmdiff.ModelDiff` (use `lmdiff.compare()`).
- `lmdiff.config.Config` (use top-level `lmdiff.Config` from `lmdiff._config`).
- v0.2.x kwargs `prompts=` (now `probes=`), `n_samples=` (now `n_probes=`), dict-style `decode={...}` (now `DecodeSpec(...)`).
- All emit `DeprecationWarning` and continue working in v0.3.x. **Removed in v0.4.0.**

### Architecture
- Lazy import preserved end-to-end: `import lmdiff` and `import lmdiff.report` do NOT load torch / transformers / matplotlib. Each heavy dependency loads on first use only.
- Engine capability negotiation runs before any inference: a metric requiring a capability the backend does not declare raises `CapabilityError` immediately, not after a 7B model load.
- All renderers (terminal / markdown / HTML / JSON / figures) share `result.findings` for narrative content. Cross-renderer consistency — every `Finding.summary` appears verbatim in every channel — is a test invariant.
- `lmdiff/_LAZY` PEP-562 `__getattr__` mechanism lets callers do `from lmdiff import compare` without paying torch import latency.

### Notes
- Tests: 790 passing (was 482 in v0.2.4) across config, engine, geometry, findings, report, and 4 viz modules. 7 skipped (lazy-import vacuous-skip guards), 59 deselected (slow / GPU markers).
- v0.3.0 freezes the new public API surface. Phase 2 (v0.4.0) ships probe taxonomy + 4 builtin task probe sets + YAML loader + the v0.2.x shim removal.
- Calibration validated on the Llama-2 4-variant family experiment: terminal output structurally matches v6 §13, application figures reproduce the v6 §14 reference outputs to four decimal places.

## [0.2.4] - 2026-04-24

### Fixed
- **`v01.json` is now shipped in the wheel and sdist.** v0.2.2 and v0.2.3 published wheels that omitted `lmdiff/probes/v01.json` because `[tool.setuptools.packages.find]` ships Python modules only, not data files. Every CLI default (`--probes v01` across `compare`, `radar`, `run-task`, `geometry`) failed with `BadParameter` immediately after `pip install lmdiff-kit`. Added `[tool.setuptools.package-data] lmdiff = ["probes/*.json"]` so the built-in probe set rides inside both artifacts. Source-install users were unaffected.
- **`import lmdiff.cli` no longer pulls `torch` and `transformers` at module-load time.** The typer console-script entrypoint (`lmdiff = lmdiff.cli:app`) was loading `lmdiff/__init__.py` → `from lmdiff.diff import ModelDiff` → `engine.py` → `torch`, so `lmdiff --help` and `lmdiff list-metrics` paid full torch-import latency (and hard-crashed on torch-less environments). `lmdiff/__init__.py` now uses :pep:`562` `__getattr__` to lazy-resolve the re-exports — public API (`from lmdiff import ModelDiff`) is unchanged, but torch / transformers are only imported when you actually reach for a symbol that needs them.

### Notes
- No code changes to metrics, geometry, or viz. Tests pass unchanged (+0).
- If you're upgrading from v0.2.2 / v0.2.3: this release is strongly recommended, nothing to migrate.

## [0.2.3] - 2026-04-24

### Added
- `GeoResult.magnitudes_specialization_zscore()` — per-variant row-wise z-scored per-domain normalized magnitudes. Recovers training-objective specialization signatures masked by absolute magnitude ranking (L-023).
- `lmdiff.experiments.family.TASK_TO_DOMAIN` and `DEFAULT_DOMAIN_ORDER` (exported from top-level).
- Summary JSON now includes `delta_specialization_zscore_by_variant` field (variant → task_name → z-score).
- `lmdiff plot-geometry` CLI now produces a 7-figure paper-grade set by default: cosine heatmaps (raw + selective), per-task normalized magnitude, specialization z-score (the paper main figure), PCA scatter (raw + normalized), raw-vs-normalized bar comparison.
- `lmdiff/viz/` restructured into per-plotter modules (`cosine`, `normalized_magnitude`, `specialization`, `pca`, `normalization_effect`, `family_figures`) with shared `_style.py` (`VARIANT_COLORS`, `VARIANT_MARKERS`, `DEFAULT_DPI`, `BASE_MARKER`).
- CLI flags `--figures`, `--variant-order`, `--domain-order`, `--dpi` on `plot-geometry`.
- `lmdiff.experiments.family.TASK_MAX_NEW_TOKENS` — per-task generation length defaults (`gsm8k=256`, `longbench_*=128`; MCQ tasks at 16).
- `lmdiff.experiments.family.resolve_max_new_tokens(task, default, overrides)` helper.
- CLI flag `--task-max-new-tokens KEY=VAL,...` on `family-experiment`.

### Fixed
- Summary JSON `delta_magnitude_by_variant_normalized` was written by querying a domain-keyed dict with task-name keys, producing all-zero values in v0.2.2. Now correctly populated via `TASK_TO_DOMAIN` reverse-lookup. The raw GeoResult JSON (`*_georesult.json`) was never affected.
- `gsm8k` and `longbench_*` accuracy no longer clamp to artifactual 0.0: v0.2.2 used a uniform `max_new_tokens=16` across all tasks, truncating generative output before a scoreable answer could be produced. MCQ tasks (hellaswag/arc/mmlu_*) behavior unchanged.

### Deprecated
- `lmdiff.experiments.family.plot_family_geometry` emits `DeprecationWarning` and is superseded by `lmdiff.viz.plot_family_figures`. Will be removed in v0.4.0. Radar PNGs from `run_family_experiment` are unaffected.

### Notes
- No GeoResult schema bump (still v4). Summary JSON wire format stable: `delta_specialization_zscore_by_variant` field added; existing fields unchanged in structure.
- Validated against the 4-variant Llama-2 family experiment: specialization fingerprint matches independently to two decimals (yarn hellaswag +1.42, long arc +1.89, code mmlu +1.33, math gsm8k +1.59); normalization effect bars confirm yarn longbench contribution at 98.9% of raw ‖δ‖².

## [0.2.2] - 2026-04-23

### Added
- **`lmdiff.experiments.family`** — library entry point for the family-experiment workflow:
  - `run_family_experiment(base, variants, tasks, ...) -> FamilyExperimentResult` bundles the GeoResult + per-task δ magnitudes (raw and per-token-normalized) + per-variant per-task accuracies + output paths + per-phase timings.
  - `plot_family_geometry(geo_or_path, output_dir)` re-renders the figure suite (direction heatmap / selective heatmap / PCA scatter / domain bar with v4 per-token-normalized fallback) from a GeoResult instance or JSON path.
  - `DEFAULT_TASKS` and `FamilyExperimentResult` exported at the package root.
- **`lmdiff family-experiment` CLI** — repeatable `--variant name=model_id` flag (one per variant), `--tasks` comma-separated (defaults to `DEFAULT_TASKS`), full passthrough for `--limit-per-task` / `--max-new-tokens` / `--seed` / `--dtype` / `--skip-accuracy` / `--output-prefix` / `--no-radars`. Replaces ad-hoc invocation of `scripts/run_family_geometry_lm_eval.py`.
- **`lmdiff plot-geometry` CLI** — render the figure suite from a previously written GeoResult JSON. `--no-index` to skip the HTML preview.
- **`GeoResult.avg_tokens_per_probe`** — per-probe token count from the base tokenizer, length == `n_probes` after the NaN filter.
- **`GeoResult.magnitudes_normalized`** — bulk per-token-normalized magnitude: `raw / sqrt(n_probes × mean_tokens)`. Comparable across probe sets with very different prompt lengths.
- **`GeoResult.magnitudes_per_task_normalized()`** — per-variant per-task per-token-normalized magnitude (requires both `probe_domains` and `avg_tokens_per_probe`).
- **`GeoResult.pca_map(use_normalized=True)`** — new flag; when enabled and `avg_tokens_per_probe` is populated, scales each probe entry `δ_v[i]` by `1/sqrt(token_count_for_probe_i)` before SVD. Default flipped to `True` (was raw-only).
- Family-experiment summary JSON gains `delta_magnitude_by_variant_normalized` and `magnitudes_total_normalized` fields when v4 token data is present.

### Changed
- GeoResult JSON `schema_version` bumped `"3"` → `"4"`. Reader accepts v1, v2, v3, v4.
- Default `pca_map(use_normalized=True)`. Existing v1/v2/v3 GeoResults keep producing raw PCA (graceful fallback when `avg_tokens_per_probe` is empty).
- `scripts/plot_family_geometry.py` — `domain_bar` now defaults to `magnitudes_per_task_normalized()` when v4 token data is present, falls back to raw `domain_heatmap()` on v3 / older.
- `scripts/run_family_geometry_lm_eval.py` and `scripts/plot_family_geometry.py` reduced to thin argparse wrappers around the new library API. Backward-compatible CLIs (the legacy script keeps its comma-separated `--variants` flag).

### Notes
- Backward compatible with v0.2.x public API. `magnitudes`, `cosine_matrix`, `selective_cosine_matrix`, `change_vectors`, `probe_domains`, `constant_fractions` all unchanged.
- Per-token normalization motivated by L-022: in lm-eval task mixes with very heterogeneous prompt lengths (e.g. ~30-token MCQ vs ~9000-token long-context QA), raw `‖δ‖` is dominated by the longest-prompt task and obscures per-token CE differences across the rest.
- `[lm-eval]` extra continues to pull `jieba` / `fuzzywuzzy` / `rouge` so longbench tasks load without an additional step.

## [0.2.1] - 2026-04-21

### Added
- `GeoResult.pca_map(n_components)` — PCA projection of change vectors via numpy SVD; returns `PCAResult` with coords + explained variance ratio.
- `GeoResult.domain_heatmap()` — per-variant per-domain ‖δ‖ magnitude, partitioned from `probe_domains`.
- `GeoResult.complementarity(v1, v2, threshold)` — two-variant overlap / unique domain analysis.
- `GeoResult.cluster(method, distance_metric)` — hierarchical clustering of variants via scipy; returns `ClusterResult` with linkage matrix.
- `GeoResult.probe_domains` — new field storing per-probe domain labels (populated automatically by `ChangeGeometry.analyze` when given a `ProbeSet`).
- `lmdiff/viz/direction_heatmap.py` — N×N cosine similarity heatmap.
- `lmdiff/viz/pca_scatter.py` — 2D PCA scatter of variants.
- `lmdiff/viz/domain_bar.py` — grouped bar chart of per-variant per-domain ‖δ‖.
- `scripts/plot_family_geometry.py` — offline figure generator from GeoResult JSON.
- scipy added to `[viz]` optional dependency.

### Changed
- GeoResult JSON `schema_version` bumped "2" → "3" (adds `probe_domains`). Reader accepts v1/v2/v3 payloads.

### Notes
- Backward compatible with v0.2.x public API. Existing `magnitudes`, `cosine_matrix`, `selective_cosine_matrix`, and `constant_fractions` unchanged.
- `domain_heatmap()`, `complementarity()`, and the `domain_bar` plot all require populated `probe_domains`. For a v1/v2 JSON or a list-of-strings probe set, these methods raise `ValueError` with an instruction to regenerate.
- Optional-dep-dependent tests (scipy clustering, matplotlib rendering) now use `pytest.skipif`; CI installs `[dev,viz]` so they run. See LESSONS L-021.

## [0.2.0] - 2026-04-21

### Added
- `from_lm_eval(task_name, ...)` adapter: load lm-evaluation-harness tasks as ProbeSets with full metadata (task_name, native_metric, output_type, choices, correct_index, aliases).
- `KNOWN_TASK_DOMAINS` registry covering 30 lm-eval tasks across 9 domains (commonsense, reasoning, math, knowledge, code, reading, language, long-context, safety).
- `F1` evaluator (SQuAD-style token-overlap F1) with multi-target support via `probe.metadata["aliases"]`.
- `Gsm8kNumberMatch` evaluator for number extraction + exact match on gsm8k-style tasks.
- `loglikelihood_accuracy(probes, engine, normalize=True)`: lm-eval-style acc / acc_norm for multiple-choice probes via per-choice CE scoring.
- `lmdiff/viz/radar.py`: matplotlib-based polar radar chart rendering (part of `[viz]` optional dep).
- `scripts/run_family_geometry_lm_eval.py`: experiment script producing δ-magnitude radar + accuracy radar across default 5 tasks.
- `scripts/discover_lm_eval_tasks.py`: read-only tool for extending `KNOWN_TASK_DOMAINS`.
- `[lm-eval]` optional dependency group (`pip install lmdiff-kit[lm-eval]`).

### Changed
- `KNOWN_TASK_DOMAINS`: `naturalqs` → `nq_open` (lm-eval 0.4.x canonical name).
- `from_lm_eval`: multiple-choice probes now populate `metadata["choices"]` and `metadata["correct_index"]` in addition to the primary target text.

### Notes
- Default code axis for the accuracy radar is `mmlu_college_computer_science` (capability MCQ), not HumanEval — see LESSONS L-019 for the rationale. HumanEval/MBPP are in the registry with `requires_execution=True` and are available for δ-magnitude-only experiments.
- Sweep code that touches many lm-eval tasks must use `except BaseException:` — some task configs call `sys.exit()` on missing system deps (L-020).

## [0.1.2] - 2026-04-20

### Added
- `ChangeGeometry` (geometry.py): δ-vector framework for one base × N variants.
- Step 1.5 selective decomposition: `delta_means`, `selective_magnitudes`, `selective_cosine_matrix`, `constant_fractions`.
- `examples/family_geometry_partial.json`, `examples/family_geometry_extended.json`.

## [0.1.1] - 2026-04-19

### Fixed
- Probe-slice alignment in TokenKL/TokenEntropy when configs differ in prefix (L-009).
- `tokenizers_equivalent` now correctly identifies slow/fast tokenizer variants as equivalent (L-011).
- `CapabilityRadar` no longer double-generates under sampling decode (L-010).

## [0.1.0] - 2026-04-16

### Initial release
- BehavioralDistance, TokenEntropy, TokenKL, CapabilityRadar.
- Config abstraction, InferenceEngine, ProbeSet.
- CLI + JSON reports + terminal reports.
