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
import lmdiff
from lmdiff import Config, DecodeSpec
from lmdiff.report.terminal import print_geometry

# Pairwise comparison (v0.3.0)
result = lmdiff.compare(
    "gpt2",            # str → Config(model="gpt2")
    "distilgpt2",
    probes="v01",      # bundled 90-probe set; pass a ProbeSet for custom
    n_probes=90,
    max_new_tokens=16,
)
print_geometry(result)

# One-vs-N family comparison
family = lmdiff.family(
    Config(model="meta-llama/Llama-2-7b-hf"),
    {
        "yarn": "NousResearch/Yarn-Llama-2-7b-128k",
        "sampled": Config(
            model="meta-llama/Llama-2-7b-hf",
            decode=DecodeSpec(strategy="sample", temperature=0.7),
        ),
    },
    probes="v01",
    n_probes=100,
)
```

> **Migrating from v0.2.x?** `lmdiff.ModelDiff` and `lmdiff.config.Config`
> still work but emit `DeprecationWarning` and will be removed in v0.4.0.
> See [`docs/migration/v02-to-v03.md`](docs/migration/v02-to-v03.md) for
> the mapping table.

## Metrics: what each one means

lmdiff reports several metrics at three levels. The rule of thumb:

- **Output-level** metrics (`BD`, `KL`, `ΔEntropy`) answer: *how different is variant A from base on a single probe set?*
- **Capability-level** metrics (`CapabilityRadar`) answer: *which skills improved or degraded?*
- **Geometry-level** metrics (`ChangeGeometry`) answer: *do two or more variants drift from base in the same direction, and on which domains?*

### Output-level metrics (pairwise: base vs one variant)

| Metric | Units | What it measures |
|---|---|---|
| **BehavioralDistance (BD)** | nats or bits-per-byte | How surprised each model is by the other's output, symmetrically. `BD = 0` means behaviorally identical; `BD > 1` means one model finds the other's text roughly as surprising as a different language. BPB-normalized when tokenizers differ. |
| **TokenKL** | nats | Symmetric KL divergence over the full next-token vocabulary, averaged over positions. `KL = 0` means the models agree on every token's distribution. Requires matching tokenizers. |
| **ΔEntropy** | nats | Mean per-token entropy of variant minus base. Positive = variant is more uncertain (often: more creative, or less confident). Negative = variant is more confident (often: RLHF'd, distilled, or narrow fine-tune). |

**Reading them together:** `BD` high + `KL` zero means behavior differs but weights don't (e.g. temperature change). `BD` high + `KL` high + `ΔEntropy` ≈ 0 means weights shifted but confidence didn't (e.g. scale-up). `BD` high + `KL` high + `ΔEntropy` large means the model's whole confidence profile changed (e.g. RLHF).

### Capability-level: CapabilityRadar

Breaks BD and accuracy down by domain (math, code, commonsense, ...). Surfaces "variant is better overall but worse on math" patterns that a single BD scalar hides.

### Geometry-level: ChangeGeometry (multi-variant)

For each variant *v*, the **change vector** **δ_v** has one entry per probe, measuring how much the variant's preferred continuation is more natural to itself than to base. Geometry metrics compare these vectors across variants.

| Metric | Range | What it answers |
|---|---|---|
| **Magnitude** `‖δ_v‖` | ≥ 0 | How much variant *v* deviates from base overall. Largest magnitude = most globally changed variant. |
| **Per-task normalized magnitude** `‖δ_{v,d}‖ / √(n_d · T̄_d)` | ≥ 0 | How much of that deviation lives on domain *d*, after correcting for the fact that long-context probes accumulate larger raw `‖δ‖` even when the underlying per-token behavior is stable. |
| **Specialization z-score** `z_{v,d}` | ~[−2.5, +2.5] | Relative to this variant's *own* row mean, which domain is its signature? `z ≥ +1` = this variant is notably more active on this domain than its average across domains. Recovers "what was this variant trained for." |
| **Cosine similarity** `cos(δ_u, δ_v)` | [−1, +1] | Do variants *u* and *v* push base in the same probe-by-probe direction? `+1` = perfect agreement, `0` = independent, `−1` = opposed. |
| **Selective cosine / Pearson r** | [−1, +1] | Same, after subtracting each variant's mean δ. Strips out any uniform "variant is X nats harder on every probe" offset and keeps only probe-specific agreement. If raw cosine is high but selective is low, variants agreed because of a shared offset, not because they favor the same probes. |

**Why per-task normalization matters.** In a heterogeneous probe mix (short MCQ + long extractive QA), raw `‖δ‖²` is dominated by the longest probes: in our 4-variant Llama-2 experiment, longbench contributed **88–99%** of each variant's raw `‖δ‖²`. Per-task normalization makes magnitudes comparable across domains so that specialization signatures become visible.

**Why specialization z-score matters.** Per-domain magnitudes already remove length bias, but variants still differ in *overall* activity level. A globally-active variant ranks highest on every domain; to see "which domain is this variant's peak," subtract each variant's own row mean. That's the z-score. Direct absolute comparison answers *"in domain d, who's most active?"*; z-score answers *"for variant v, which domain is its signature?"* — different questions, both tables produced.

**Why the two cosines.** Raw cosine tells you whether variants agree on probe-level direction at all. Selective cosine separates "they have the same offset" from "they prefer the same probes." If `yarn` and `long` both have raw cosine 0.95 with `code`, but `yarn-code` selective is 0.94 and `long-code` is 0.85, then `yarn` and `code` share probe-specific preferences while `long-code` agreement was more offset-driven.

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

**lm-eval-harness tasks** (`[lm-eval]` extra) load directly into ProbeSets:

```python
from lmdiff.probes.adapters import from_lm_eval
probes = from_lm_eval("hellaswag", limit=100, seed=42)  # or arc_challenge, gsm8k, ...
```

## Example: Llama-2-7B family comparison

One base model, seven variants, 90 completion-style probes across math/knowledge/code:

| Variant | Modification | BD | KL | ΔEntropy | Reading |
|---|---|---|---|---|---|
| 7B + temp=1.5 | Decoding only | 0.59 | 0.00 | +0.00 | Behavior shifts (BD>0) but weights and confidence unchanged — sampling-only effect. |
| CodeLlama-7B | Domain fine-tune | 0.79 | — | — | Different vocab; KL/Entropy undefined (BD uses BPB normalization). |
| Llama-2-13B | Scale up | 0.85 | 0.17 | −0.06 | Weights differ but confidence nearly unchanged — scaling is mostly quiet. |
| YaRN-128k | RoPE scaling | 0.99 | 0.35 | +0.05 | Behavior shifts noticeably, confidence unchanged — extends context range without adding uncertainty. |
| Llama-2-7B-32K | Continued pretrain | 1.07 | 0.71 | **+0.41** | Higher uncertainty across the board — pretraining substantially loosened the distribution. |
| 7B + system prompt | Prefix context | 1.09 | 1.62 | −0.11 | Largest KL of the set. A single prompt reshapes next-token distributions more than 13B scaling does. |
| Llama-2-7B-chat | RLHF | 1.15 | 1.14 | **−0.41** | Most confident (lowest entropy) and most behaviorally distant — RLHF sharpens the distribution. |

BD = Behavioral Distance (nats). KL = symmetric TokenKL. ΔEntropy = entropy(variant) − entropy(base). CodeLlama has a different vocabulary; KL/Entropy require matching tokenizers.

**What this table surfaces that accuracy benchmarks don't:**

- A system prompt moves behavior more (KL 1.62) than adding 6B parameters does (KL 0.17).
- Temperature 1.5 has `KL = 0` — it changes what gets sampled, not what the model believes.
- YaRN and 32K both extend context, but differently: YaRN shifts without adding uncertainty (ΔEntropy ≈ 0), while 32K's continued pretraining loosens the distribution (ΔEntropy = +0.41).
- RLHF is the only modification here with **negative** ΔEntropy — it makes the model more certain, not less.

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
