import pytest

from modeldiff.config import Config
from modeldiff.engine import InferenceEngine

TEST_MODELS = [
    pytest.param("gpt2", id="gpt2"),
    pytest.param(
        "meta-llama/Llama-2-7b-hf",
        id="llama2-7b",
        marks=pytest.mark.slow,
    ),
]

_engine_cache: dict[str, InferenceEngine] = {}


def _get_engine(model_name: str) -> InferenceEngine:
    if model_name not in _engine_cache:
        config = Config(model=model_name)
        _engine_cache[model_name] = InferenceEngine(config)
    return _engine_cache[model_name]


@pytest.fixture(params=TEST_MODELS, scope="session")
def engine(request):
    """Parameterized engine fixture — runs each test against every model."""
    return _get_engine(request.param)


@pytest.fixture(scope="session")
def tiny_model():
    return _get_engine("gpt2")


@pytest.fixture(scope="session")
def llama_engine():
    return _get_engine("meta-llama/Llama-2-7b-hf")


@pytest.fixture(scope="session")
def distil_engine():
    return _get_engine("distilgpt2")


@pytest.fixture
def gpt2_config():
    return Config(model="gpt2")


@pytest.fixture
def gpt2_config_with_context():
    return Config(
        model="gpt2",
        context=[{"role": "user", "content": "Hello"}],
        system_prompt="You are helpful.",
        name="gpt2-ctx",
    )


@pytest.fixture
def sample_probes():
    return [
        "The capital of France is",
        "2 + 2 =",
        "def fibonacci(n):",
    ]
