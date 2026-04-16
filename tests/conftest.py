import pytest

from modeldiff.config import Config


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
