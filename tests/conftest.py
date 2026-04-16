import pytest

from modeldiff.config import Config
from modeldiff.engine import InferenceEngine

TEST_MODEL = "gpt2"


@pytest.fixture
def gpt2_config():
    return Config(model=TEST_MODEL)


@pytest.fixture
def gpt2_config_with_context():
    return Config(
        model=TEST_MODEL,
        context=[{"role": "user", "content": "Hello"}],
        system_prompt="You are helpful.",
        name="gpt2-ctx",
    )


@pytest.fixture(scope="session")
def tiny_model():
    """Session-scoped engine to avoid reloading gpt2 per test."""
    config = Config(model=TEST_MODEL)
    return InferenceEngine(config)


@pytest.fixture
def sample_probes():
    return [
        "The capital of France is",
        "2 + 2 =",
        "def fibonacci(n):",
    ]
