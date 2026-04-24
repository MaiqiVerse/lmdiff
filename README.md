# lmdiff

> Measures **how** and **where** two LLM configurations differ — not just whether one scores higher.

Compare language model **configurations** — not just weights, but weights + context + decoding + adapter + agent — via behavioral distance and multi-level diagnostics.

## Why lmdiff?

`lm-eval-harness` tells you "model A scores 3 points higher than model B on MMLU." That's a scalar.

lmdiff tells you *where* those 3 points came from: which capabilities shifted, how far the output distribution moved, and whether two different modifications (e.g. a fine-tune vs. a context change) push behavior in the same direction or in opposite directions.

A **Configuration** is `model + context + decoding + adapter + agent scaffold`, not just model weights. Same checkpoint with a different system prompt is a different config — and lmdiff can quantify the difference.

## Install

```bash
pip install lmdiff-kit

# With lm-eval-harness task loader (hellaswag, arc, gsm8k, mmlu, ...)
pip install "lmdiff-kit[lm-eval]"

# With matplotlib radar plots
pip install "lmdiff-kit[viz]"

# Both
pip install "lmdiff-kit[lm-eval,viz]"
```

The import name is `lmdiff`; the PyPI distribution is `lmdiff-kit` (name disambiguation on PyPI).

### Development install

```bash
mamba create -n lmdiff python=3.12
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e .
```

`cu130` is for RTX 5090 / Blackwell. Pick the CUDA version that matches your GPU.

## Command line

```bash
# Metric-level comparison (BD, token entropy, token KL)
lmdiff compare gpt2 distilgpt2 --probes v01

# Same, but JSON output to file
lmdiff compare gpt2 distilgpt2 --probes v01 --json --output result.json

# Per-domain capability radar (accuracy + BD per domain)
lmdiff radar gpt2 distilgpt2 --probes v01

# Single-model task evaluation
lmdiff run-task gpt2 --probes v01 --evaluator contains_answer

# List available metrics
lmdiff list-metrics
```

### Quick start: family experiment

End-to-end ChangeGeometry + per-task accuracy + radar PNGs over an
lm-eval task mix (requires `pip install "lmdiff-kit[lm-eval,viz]"`):

```bash
lmdiff family-experiment \
    --base meta-llama/Llama-2-7b-hf \
    --variant yarn=NousResearch/Yarn-Llama-2-7b-128k \
    --variant code=codellama/CodeLlama-7b-hf \
    --tasks hellaswag,arc_challenge,gsm8k \
    --task-max-new-tokens gsm8k=256,longbench_2wikimqa=128 \
    --output-dir runs/llama2-family

# Render the 7-figure paper-grade set from a GeoResult JSON
lmdiff plot-geometry runs/llama2-family/family_geometry_lm_eval_georesult.json \
    --output-dir runs/llama2-family/figures \
    --variant-order yarn,long,code,math
```

`--variant` is repeatable; defaults to the 5-task mix used in the
Llama-2 example below when `--tasks` is omitted. `plot-geometry` produces
7 numbered PNGs by default — cosine heatmaps (raw + selective), per-task
normalized magnitude, **specialization z-score (the paper main figure)**,
PCA scatter (raw + normalized), and a raw-vs-normalized bar comparison.
Use `--figures specialization,cosine_selective` to render a subset.

Both subcommands wrap `lmdiff.experiments.family.run_family_experiment`
and `lmdiff.viz.plot_family_figures`, also callable directly from Python.

> **Note on accuracy clamping (v0.2.2 artifact, fixed in v0.2.3):**
> generative tasks like `gsm8k` (chain-of-thought) and `longbench_2wikimqa`
> need 128–256 tokens of generation, not the MCQ default of 16. Pass
> `--task-max-new-tokens gsm8k=256,longbench_2wikimqa=128` (or rely on
> `TASK_MAX_NEW_TOKENS` defaults) or accuracy will silently clamp to 0.0.
> See `LESSONS.md` L-024.

## Python API

```python
from lmdiff import Config, ModelDiff, ProbeSet
from lmdiff.report.terminal import print_report, print_radar

probes = ProbeSet.from_json("lmdiff/probes/v01.json")
md = ModelDiff(
    Config(model="gpt2"),
    Config(model="distilgpt2"),
    probes,
)

# Metric-level comparison
report = md.run(level="output", max_new_tokens=16)
print_report(report)

# Per-domain capability radar
radar_result = md.run_radar(probes=probes, max_new_tokens=16)
print_radar(radar_result)
```

## Example: Llama-2-7B family comparison

One base model, seven variants, 90 completion-style probes across math/knowledge/code:

| Variant | Modification | BD | KL | ΔEntropy |
|---|---|---|---|---|
| 7B + temp=1.5 | Decoding only | 0.59 | 0.00 | +0.00 |
| CodeLlama-7B | Domain fine-tune | 0.79 | — | — |
| Llama-2-13B | Scale up | 0.85 | 0.17 | −0.06 |
| YaRN-128k | RoPE scaling | 0.99 | 0.35 | +0.05 |
| Llama-2-7B-32K | Continued pretrain | 1.07 | 0.71 | +0.41 |
| 7B + system prompt | Prefix context | 1.09 | 1.62 | −0.11 |
| Llama-2-7B-chat | RLHF | 1.15 | 1.14 | −0.41 |

BD = Behavioral Distance (nats). KL = symmetric TokenKL. ΔEntropy = entropy(variant) − entropy(base). CodeLlama has a different vocabulary; KL/Entropy require matching tokenizers.

**What this table shows:**

A single system prompt causes more distributional shift (BD=1.09) than scaling to 13B parameters (BD=0.85). Temperature=1.5 changes generation behavior (BD=0.59) but leaves the underlying distribution identical (KL=0, Entropy=0) — it only affects sampling, not the model's beliefs. YaRN and 32K both extend context length, but do it differently: YaRN shifts the distribution without increasing uncertainty (Entropy≈0), while 32K's continued pretraining substantially increases uncertainty (Entropy=+0.41).

These are the kinds of insights that accuracy benchmarks cannot surface.

## What gets measured

Three output-level metrics:

- **BehavioralDistance** — symmetric, self-entropy-baseline-subtracted cross-entropy distance. BPB-normalized when tokenizers differ.
- **TokenEntropy** — mean per-token next-token entropy delta, A vs B.
- **TokenKL** — symmetric KL divergence over full vocab.

**CapabilityRadar** adds per-domain accuracy + BD breakdown across math/knowledge/code (or any multi-domain probe set).

**ChangeGeometry** (v0.2.0, extended in v0.2.1) compares one base model against *N* variants simultaneously. For each variant it builds a change vector δ by probe, then exposes magnitudes, a full pairwise cosine matrix, and a selective (mean-subtracted, Pearson) cosine matrix that separates "uniform behavioral shift" from "selective behavioral shift". v0.2.1 adds `pca_map()`, `domain_heatmap()`, `complementarity()`, and scipy-backed `cluster()` for further decomposition.

```python
from lmdiff import ChangeGeometry, Config, ProbeSet
geo = ChangeGeometry(
    base=Config(model="meta-llama/Llama-2-7b-hf"),
    variants={
        "yarn": Config(model="NousResearch/Yarn-Llama-2-7b-128k", name="yarn"),
        "code": Config(model="codellama/CodeLlama-7b-hf", name="code"),
    },
    prompts=probes,
).analyze(max_new_tokens=16)
```

**lm-eval-harness tasks** (v0.2.0, `[lm-eval]` extra) load directly into ProbeSets:

```python
from lmdiff.probes.adapters import from_lm_eval
probes = from_lm_eval("hellaswag", limit=100, seed=42)  # or arc_challenge, gsm8k, ...
```

All return structured results with per-probe breakdowns in `.details`.

## Configuration abstraction

A `Config` is more than a model name:

```python
Config(
    model="gpt2",
    system_prompt="You are concise.",
    context=[{"role": "user", "content": "..."}],
    decode={"strategy": "sample", "temperature": 0.7},
    name="gpt2-concise",
)
```

Same weights + different context/decoding = different config = measurable behavioral difference.

## JSON output

All results serialize to deterministic JSON with `schema_version` for forward compatibility:

```python
from lmdiff.report.json_report import to_json, write_json
write_json(report, "output.json")
```

## Status

Phase 2 shipped — published to PyPI as `lmdiff-kit` v0.2.3. Now working: everything from v0.1.x plus **ChangeGeometry** (N-variant δ-vector geometry with PCA / domain heatmap / complementarity / hierarchical clustering, plus per-token normalized magnitudes and **specialization z-score fingerprints** for recovering training-objective signatures), **lm-eval-harness adapter** (30+ task registry), `loglikelihood_accuracy` (acc_norm-style MCQ scoring), `F1` and `Gsm8kNumberMatch` evaluators, the `lmdiff family-experiment` / `lmdiff plot-geometry` CLIs (and matching `lmdiff.experiments.family` library API), per-task generation-length overrides via `TASK_MAX_NEW_TOKENS`, and a paper-grade 7-figure suite under the `[viz]` extra (cosine heatmaps, normalized magnitude, specialization, PCA scatter, normalization effect).

Not yet: representation / trajectory / causal metrics, HTML / LaTeX reports, HumanEval-style executional tasks (sandboxing deferred — δ-magnitude-only usage is already available). See `CLAUDE.md` for the full roadmap.

## Development

```bash
pytest                                    # fast tests (mocks only)
pytest -m slow -o "addopts="              # includes gpt2/distilgpt2 E2E
```

Architecture rules, implementation order, and coding conventions live in `CLAUDE.md`.

## License

MIT — see [LICENSE](LICENSE).

## Citation

Paper forthcoming.
