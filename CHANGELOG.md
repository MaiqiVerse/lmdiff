# Changelog

## [0.3.0] - 2026-04-21

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
