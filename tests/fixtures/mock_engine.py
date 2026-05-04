"""MockEngine — fake :class:`~lmdiff._engine.Engine` for unit tests.

Conforms to the Engine Protocol via duck typing. Outputs are
deterministic given ``(prompt, continuation, seed)``. Use in unit tests
to validate metric implementations and rendering without GPU / torch /
model downloads.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Optional

import numpy as np

from lmdiff._config import Config
from lmdiff._engine import (
    AttentionWeightsResult,
    GenerateResult,
    HiddenStatesResult,
    RESERVED_CAPABILITIES,
    ScoreResult,
)


class MockEngine:
    """Mock :class:`~lmdiff._engine.Engine` for unit tests.

    Returns deterministic synthetic outputs. Conforms to the Engine
    Protocol so any metric or pipeline that accepts an ``Engine`` can be
    exercised against it.

    Outputs are deterministic given ``(prompt, continuation, seed)``.

    Parameters
    ----------
    config : Config, optional
        Backing config. Defaults to ``Config(model="mock_model")``.
    capabilities : frozenset[str], optional
        Override capability set for testing capability-mismatch behavior.
    seed : int
        Base seed for deterministic outputs.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        capabilities: Optional[frozenset[str]] = None,
        seed: int = 42,
        n_layers: int = 12,
        hidden_dim: int = 768,
    ) -> None:
        self._config = config or Config(model="mock_model")

        if capabilities is None:
            self._capabilities = frozenset({
                "score", "generate", "hidden_states", "attention_weights",
                "logprobs_full", "batch",
            })
        else:
            unknown = set(capabilities) - RESERVED_CAPABILITIES
            if unknown:
                raise ValueError(f"Unknown capability names: {unknown}")
            self._capabilities = frozenset(capabilities)

        self._seed = seed
        self._n_layers = n_layers
        self._hidden_dim = hidden_dim
        self._tokenizer_id = self._mock_tokenizer_id()
        self._closed = False

    @property
    def name(self) -> str:
        return self._config.display_name

    @property
    def tokenizer_id(self) -> str:
        return self._tokenizer_id

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def capabilities(self) -> frozenset[str]:
        return self._capabilities

    def with_config(self, config: Config) -> "MockEngine":
        """Return a copy of this MockEngine bound to a different ``Config``."""
        return MockEngine(
            config=config,
            capabilities=self._capabilities,
            seed=self._seed,
            n_layers=self._n_layers,
            hidden_dim=self._hidden_dim,
        )

    def _mock_tokenizer_id(self) -> str:
        """Stable id keyed off ``config.model`` so two MockEngines for the
        same model agree."""
        return hashlib.sha256(self._config.model.encode("utf-8")).hexdigest()[:16]

    # ── v0.4.0 commit 4.0 — Engine Protocol additions ─────────────────

    def token_count(self, text: str) -> int:
        """Mock: word-split tokenization (matches the synthetic logprobs
        in ``score`` / ``generate``)."""
        return max(len(text.split()), 1)

    def tokenizers_equivalent_to(self, other: Any) -> bool:
        """Default Protocol behaviour — compare ``tokenizer_id``. Two
        MockEngines built from the same ``Config.model`` agree."""
        return self.tokenizer_id == other.tokenizer_id

    # ── Required methods ──────────────────────────────────────────────

    def score(self, prompt: str, continuation: str) -> ScoreResult:
        if "score" not in self._capabilities:
            raise NotImplementedError("score capability not in mock capabilities")
        rng = random.Random(self._seed + hash((prompt, continuation)))
        n_tokens = max(len(continuation.split()), 1)
        logprobs = np.asarray(
            [rng.gauss(-5.0, 2.0) for _ in range(n_tokens)],
            dtype=np.float32,
        )
        tokens = list(range(n_tokens))
        avg = float(logprobs.mean()) if len(logprobs) > 0 else 0.0
        return ScoreResult(logprobs=logprobs, tokens=tokens, avg_logprob=avg)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> GenerateResult:
        if "generate" not in self._capabilities:
            raise NotImplementedError("generate capability not in mock capabilities")
        effective_seed = (seed if seed is not None else self._seed) + hash(prompt)
        rng = random.Random(effective_seed)
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        n = min(max_new_tokens, 8)
        text = " ".join(rng.choice(words) for _ in range(n))
        tokens = list(range(n))
        logprobs = np.asarray(
            [rng.gauss(-3.0, 1.0) for _ in range(n)],
            dtype=np.float32,
        )
        return GenerateResult(text=text, tokens=tokens, logprobs=logprobs)

    def hidden_states(
        self,
        prompt: str,
        *,
        layers: Optional[list[int]] = None,
        position: str = "last",
    ) -> HiddenStatesResult:
        if "hidden_states" not in self._capabilities:
            raise NotImplementedError("hidden_states not in mock capabilities")
        rng = np.random.RandomState((self._seed + hash(prompt)) % (2**31 - 1))
        layers_to_return = layers if layers is not None else list(range(self._n_layers))
        h = rng.randn(len(layers_to_return), self._hidden_dim).astype(np.float32)
        return HiddenStatesResult(hidden_states=h, position=position)

    def attention_weights(
        self,
        prompt: str,
        *,
        layers: Optional[list[int]] = None,
        heads: Optional[list[int]] = None,
    ) -> AttentionWeightsResult:
        if "attention_weights" not in self._capabilities:
            raise NotImplementedError("attention_weights not in mock capabilities")
        rng = np.random.RandomState((self._seed + hash(prompt)) % (2**31 - 1))
        n_tokens = max(len(prompt.split()), 1)
        n_heads = 12
        layers_to_return = layers if layers is not None else list(range(self._n_layers))
        aw = rng.rand(
            len(layers_to_return), n_heads, n_tokens, n_tokens,
        ).astype(np.float32)
        # Softmax-normalize each row in the last dim so it looks like attention.
        aw = aw / aw.sum(axis=-1, keepdims=True)
        if heads is not None:
            aw = aw[:, heads, :, :]
        return AttentionWeightsResult(attention_weights=aw)

    def apply_steering(
        self,
        prompt: str,
        steering_spec: Any,
        *,
        max_new_tokens: int = 16,
    ) -> GenerateResult:
        raise NotImplementedError("MockEngine does not support steering")

    def extract_steering_vector(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
        *,
        layer: int,
    ) -> Any:
        raise NotImplementedError("MockEngine does not support steering")

    def close(self) -> None:
        self._closed = True
