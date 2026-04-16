# ModelDiff

Compare language model **configurations** — not just weights, but weights + context + decoding + adapter + agent — via behavioral distance and multi-level diagnostics.

## Status

Phase 1: output-level metrics only. Works: BehavioralDistance, TokenEntropy, TokenKL, Python API. Not yet: probes loader, tasks, representation/trajectory/causal metrics, HTML/LaTeX reports, CLI. See `CLAUDE.md` for the full roadmap.

## Install

```bash
mamba create -n modeldiff python=3.12 && mamba activate modeldiff
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e .
```

`cu130` is for RTX 5090 / Blackwell. Pick the CUDA version that matches your GPU.

## 30-second demo

```python
from modeldiff import Config, ModelDiff
from modeldiff.report.terminal import print_report

probes = ["The capital of France is", "2 + 2 =", "Once upon a time"]
md = ModelDiff(
    Config(model="gpt2"),
    Config(model="distilgpt2"),
    probes,
)
report = md.run(level="output", max_new_tokens=16)
print_report(report)
```

Output: a rich table with BD, token_entropy, token_kl values and a per-probe breakdown of behavioral distance.

## What gets measured

Three output-level metrics today:

- **BehavioralDistance** — symmetric, self-entropy-baseline-subtracted cross-entropy distance. BPB-normalized when tokenizers differ. See `CLAUDE.md` for the formula.
- **TokenEntropy** — mean per-token next-token entropy delta, A vs B.
- **TokenKL** — symmetric KL divergence over full vocab. Requires matching tokenizers.

All return `MetricResult` with per-probe breakdowns in `.details`.

## Configuration abstraction

A `Config` is more than a model name:

```python
Config(
    model="gpt2",
    system_prompt="You are concise.",
    context=[{"role": "user", "content": "..."}],
    decode={"strategy": "sample", "temperature": 0.7},
    adapter=None,  # Phase 2
    ttt=None,      # Phase 2
    agent=None,    # Phase 2
    name="gpt2-concise",
)
```

Same weights + different context/decoding = different config = measurable behavioral difference. That is the point.

## Development

```bash
pytest                  # fast tests (gpt2 only)
pytest -m slow -o "addopts="  # includes llama2-7b; needs HF auth + ~14 GB VRAM
```

Architecture rules, implementation order, and coding conventions live in `CLAUDE.md` — read that before contributing.

## License

TBD
