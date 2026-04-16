# CLAUDE.md

## Development Environment

- Python: use mamba to manage environments: `mamba create -n modeldiff python=3.12 && mamba activate modeldiff`
- GPU: RTX 5090 (Blackwell, sm_120)
- PyTorch: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`

## What is this project

ModelDiff: a Python framework for comparing language model **configurations** (not just models). A "config" = model weights + context/ICL + decoding strategy + adapter + agent scaffold. The framework measures behavioral differences using conditional KL-based distance and multi-level internal metrics.

**Core novelty is NOT the distance metric** (that's well-established: Takase et al. 2026, Amini et al. 2025). The novelty is:
1. Change Geometry — treating behavioral changes as vectors with direction/magnitude/cosine similarity
2. Configuration abstraction — unifying weight, context, decoding, TTT, agent modifications
3. Engineering — a usable multi-level diagnostic toolkit

## Architecture rules (NEVER violate)

- **`engine.py` is the ONLY module that imports `transformers` or touches models.** All metrics, tasks, reports, viz receive engine outputs, never model objects.
- **Metrics are zero-coupled.** No metric imports another metric. Each takes engine outputs → returns `MetricResult`.
- **Tasks are zero-coupled with metrics.** Tasks use engines directly for generate+evaluate. Metrics measure distributional stats. They are independent systems.
- **`viz` and `html` are optional deps.** Core functionality (metrics, BD, CLI JSON output) must work with only `torch`, `transformers`, `numpy`, `typer`, `rich`, `pyyaml`.
- **ProbeSet is immutable after loading.** Never mutate probes in place.
- **All metric results use `MetricResult` dataclass.** Report layer doesn't know metric internals.

## Key interfaces (keep stable)

```python
# config.py
@dataclass
class Config:
    model: str | Any
    context: list[dict] | None = None
    system_prompt: str | None = None
    decode: dict = field(default_factory=lambda: {"strategy": "greedy"})
    adapter: str | None = None
    ttt: dict | None = None
    agent: Any | None = None
    name: str | None = None

# engine.py — the ONLY layer that touches models
class InferenceEngine:
    def generate(self, prompts: list[str], n_samples: int = 1) -> GenerationResult
    def score(self, prompts: list[str], continuations: list[str]) -> ForwardResult
    def forward_with_hidden(self, prompts: list[str], layers: list[int] | None) -> HiddenStatesResult
    def get_logits(self, prompts: list[str], topk: int = 256) -> ForwardResult

# metrics/base.py
@dataclass
class MetricResult:
    name: str
    level: MetricLevel  # OUTPUT | CALIBRATION | REPRESENTATION | TRAJECTORY | CAUSAL
    value: float | dict | np.ndarray
    details: dict | None = None
    metadata: dict | None = None

class BaseMetric(ABC):
    name: str
    level: MetricLevel
    def compute(self, engine_a, engine_b, probes, **kwargs) -> MetricResult
    @classmethod
    def is_applicable(cls, config_a, config_b) -> bool
    @classmethod
    def requirements(cls) -> dict  # {"logits": True, "hidden_states": False, ...}

# diff.py
class ModelDiff:
    def __init__(self, config_a: Config, config_b: Config, prompts, n_samples=5)
    def run(self, level="output", metrics=None) -> DiffReport
    def run_task(self, task: Task) -> TaskResult
    def run_tasks(self, tasks: list[Task]) -> FullReport

# geometry.py
class ChangeGeometry:
    def __init__(self, base: Config, variants: dict[str, Config], prompts, n_samples=5)
    def analyze(self) -> GeoResult
```

## Behavioral Distance formula

```
CE(B, A) = −(1/N) Σ log P_B(y_A_t | x, y_A_<t)     # B scores A's output, per-token normalized
CE(A, A) = −(1/N) Σ log P_A(y_A_t | x, y_A_<t)     # A's self-entropy baseline

BD(A, B) = ½[CE(A,B) − CE(B,B)] + ½[CE(B,A) − CE(A,A)]   # symmetric
Asymmetry = [CE(B,A) − CE(A,A)] − [CE(A,B) − CE(B,B)]     # positive = B narrowed
```

When tokenizers differ: normalize by UTF-8 byte count (BPB) instead of token count. Auto-detect via `config.shares_tokenizer_with(other)`.

## Change vector definition

```
δ_V[i] = CE(base_scores_variant_output[i]) − CE(variant_self_score[i])
```
One scalar per prompt → high-dim vector. Then: `cos(δ_B, δ_C)` = direction similarity, `‖δ‖` = magnitude, PCA for visualization.

## Directory layout

```
modeldiff/
├── __init__.py          # public API: Config, ModelDiff, ChangeGeometry
├── config.py
├── engine.py            # ONLY file that imports transformers
├── tokenizer_utils.py
├── diff.py
├── geometry.py
├── metrics/
│   ├── base.py
│   ├── output/          # behavioral_distance, token_entropy, token_kl, confidence_diff, perplexity_shift, semantic_entropy
│   ├── calibration/     # ece_shift, confidence_correctness
│   ├── representation/  # cosine_similarity, norm_diff, effective_rank, dead_neuron_rate, cka, intrinsic_dim, attention_drift
│   ├── trajectory/      # logit_lens, tuned_lens, concept_probing
│   └── causal/          # activation_patching, model_stitching, steering_vector
├── probes/
│   ├── loader.py        # ProbeSet class
│   └── adapters.py      # from_hf_dataset(), from_lm_eval()
├── tasks/
│   ├── base.py          # Task, BaseEvaluator
│   ├── evaluators.py    # ExactMatch, ContainsAnswer, MultipleChoice, FormatChecker, RefusalDetector, FlexibleMatch, LLMJudge
│   ├── capability_radar.py
│   ├── knowledge_drift.py
│   ├── safety_regression.py, hallucination_probe.py, consistency_check.py, style_drift.py, instruction_following.py, crosslingual.py
│   ├── yaml_parser.py
│   └── data/            # built-in probe JSON files per task
├── report/              # terminal.py, json_report.py, html_report.py, latex_report.py
├── viz/                 # domain_bar, direction_heatmap, pca_scatter, layer_curve, token_divergence, radar
└── cli.py               # typer app: compare, geometry, run-task, list-metrics
```

## Implementation order

Build and test in this exact sequence. Each step must pass tests before moving on.

1. `config.py` + `tests/test_config.py`
2. `engine.py` (use gpt2 for all tests) + `tests/test_engine.py`
3. `tokenizer_utils.py` + tests
4. `metrics/base.py`
5. `metrics/output/token_entropy.py` + `token_kl.py` + tests (use mock logits)
6. `metrics/output/behavioral_distance.py` + tests (use gpt2, tiny prompts)
7. `probes/loader.py` + `probes/v01.json` (30 probes minimum) + tests
8. `tasks/base.py` + `tasks/evaluators.py` (ExactMatch, ContainsAnswer, MultipleChoice) + tests
9. `tasks/capability_radar.py` (3 dimensions: math/code/knowledge, 30 probes each) + tests
10. `diff.py` (wire engine + metrics + probes) + integration test
11. `report/terminal.py` + `report/json_report.py`
12. `cli.py` (compare + run-task commands) + end-to-end test

## Coding conventions

- Python 3.10+. Use `X | Y` union syntax, not `Union[X, Y]`.
- Type hints on all public functions.
- Docstrings: one-line summary + Args/Returns for public methods. No docstrings on obvious internal helpers.
- Tests use `pytest`. Fixtures in `conftest.py`: `tiny_model` (gpt2), `mock_logits`, `sample_probes`.
- No wildcard imports. Explicit `from modeldiff.config import Config`.
- f-strings, not `.format()`.
- `rich` for terminal colors, never raw ANSI codes.
- When in doubt, keep it simple. No metaprogramming, no decorators-on-decorators.

## Common mistakes to avoid

- Do NOT import `transformers` outside `engine.py`.
- Do NOT make metrics depend on each other.
- Do NOT use `WidthType.PERCENTAGE` (wrong context — this is a Python package, not docx).
- Do NOT hardcode model names. Always parameterize. Tests use `gpt2`.
- Do NOT compute BD without self-entropy baseline subtraction.
- Do NOT forget BPB normalization when tokenizers differ.
- Do NOT make ProbeSet mutable.
- Do NOT put matplotlib in core dependencies. It's in `[viz]` extra.
