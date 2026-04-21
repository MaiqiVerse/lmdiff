# Changelog

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
