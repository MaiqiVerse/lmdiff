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

## Example: what lmdiff finds

Llama-2-7B vs YaRN-Llama-2-7b-128k on short prompts:

- **BD = 1.03 nats** — significant distributional shift even on prompts well within the original 4k context.
- **TokenEntropy delta ≈ 0** — the distributions shifted *direction*, not spread; YaRN didn't make the model more or less uncertain on average.
- **TokenKL = 0.35** — single-step distributions are similar, but multi-step generation diverges much further.
- **Stopping behavior changed** — YaRN learned to emit EOS after short answers; base Llama-2 keeps generating. Invisible to perplexity benchmarks; obvious in BD on generated continuations.

The point: same parameter count, similar single-step KL, but the generation behavior is meaningfully different — and the *kind* of difference is what lmdiff surfaces.

## What gets measured

Three output-level metrics:

- **BehavioralDistance** — symmetric, self-entropy-baseline-subtracted cross-entropy distance. BPB-normalized when tokenizers differ.
- **TokenEntropy** — mean per-token next-token entropy delta, A vs B.
- **TokenKL** — symmetric KL divergence over full vocab.

**CapabilityRadar** adds per-domain accuracy + BD breakdown across math/knowledge/code (or any multi-domain probe set).

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

Phase 1 shipped — published to PyPI as `lmdiff-kit` v0.1.0. Working: BehavioralDistance, TokenEntropy, TokenKL, CapabilityRadar, CLI, JSON reports, Python API.

Phase 2 in progress: **Change Geometry** — treating behavioral changes as vectors with direction/magnitude/cosine similarity across multiple variants.

Not yet: representation/trajectory/causal metrics, HTML/LaTeX reports, viz. See `CLAUDE.md` for the full roadmap.

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
