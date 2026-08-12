"""MinimalEngine — copy-paste template for users integrating custom backends.

This file is intentionally simple. To use:

  1. Read this file.
  2. Subclass :class:`MinimalEngine`.
  3. Override ``_score_impl()`` and ``_generate_impl()`` with calls to your backend.
  4. Set ``capabilities`` to reflect what your backend supports (must be a
     subset of :data:`lmdiff._engine.RESERVED_CAPABILITIES`).

Example
-------
::

    class MyCustomEngine(MinimalEngine):
        def __init__(self, config):
            super().__init__(
                config,
                capabilities=frozenset({"score", "generate"}),
                n_layers=24,
                hidden_dim=2048,
            )
            self._client = MyAPIClient(config.model)

        def _score_impl(self, prompt, continuation):
            result = self._client.score(prompt, continuation)
            return result.logprobs, result.tokens

        def _generate_impl(self, prompt, max_new_tokens, temperature, top_p, seed):
            result = self._client.generate(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            return result.text, result.tokens, result.logprobs
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional

from lmdiff._config import Config
from lmdiff._engine import (
    AttentionWeightsResult,
    GenerateResult,
    HiddenStatesResult,
    RESERVED_CAPABILITIES,
    ScoreResult,
)


class MinimalEngine:
    """Template :class:`~lmdiff._engine.Engine` — copy and customize.

    Implements the bare minimum: ``score`` and ``generate``. Does NOT
    support ``hidden_states``, ``attention_weights``, or ``steering`` by
    default. Subclasses override ``_score_impl`` and ``_generate_impl``.

    Parameters
    ----------
    config : Config
        The configuration this engine wraps.
    capabilities : frozenset[str], optional
        Capabilities your backend supports. Default: ``{"score", "generate"}``.
        Must be a subset of :data:`RESERVED_CAPABILITIES`.
    n_layers : int, optional
        Number of transformer layers (if exposed by your backend; else ``0``).
    hidden_dim : int, optional
        Hidden state dimension (if exposed; else ``0``).
    tokenizer_id : str, optional
        Stable hash for your tokenizer. If not provided, computed from the
        model name so two ``MinimalEngine`` instances with the same model
        name share a ``tokenizer_id``.
    """

    def __init__(
        self,
        config: Config,
        *,
        capabilities: frozenset[str] = frozenset({"score", "generate"}),
        n_layers: int = 0,
        hidden_dim: int = 0,
        tokenizer_id: Optional[str] = None,
    ) -> None:
        unknown = set(capabilities) - RESERVED_CAPABILITIES
        if unknown:
            raise ValueError(
                f"Unknown capability names: {unknown}. "
                f"See lmdiff._engine.RESERVED_CAPABILITIES for the list. "
                f"Naming registry is intentionally narrow — propose new names "
                f"as an issue first."
            )

        self._config = config
        self._capabilities = frozenset(capabilities)
        self._n_layers = n_layers
        self._hidden_dim = hidden_dim
        self._tokenizer_id = tokenizer_id or self._compute_default_tokenizer_id()

    def _compute_default_tokenizer_id(self) -> str:
        """Stable fallback id derived from the model name.

        Two ``MinimalEngine`` instances with the same ``config.model``
        share a ``tokenizer_id``.
        """
        return hashlib.sha256(self._config.model.encode("utf-8")).hexdigest()[:16]

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

    # ── v0.4.0 commit 4.0 — Engine Protocol additions ─────────────────

    def token_count(self, text: str) -> int:
        """Tokens this engine's tokenizer would assign to ``text``.

        Custom backends override ``_token_count_impl`` to expose their
        tokenizer's count behaviour (no special tokens). Default falls
        back to byte length, which is suitable for engines that don't
        actually have a tokenizer (mock-only / hosted-API backends);
        family-pipeline numbers using such an engine will be off, so
        prefer overriding when a real tokenizer is available.
        """
        return self._token_count_impl(text)

    def tokenizers_equivalent_to(self, other: "Engine") -> bool:
        """Default Protocol behaviour — compare ``tokenizer_id`` only.

        Custom backends owning a tokenizer object should override to
        also do a canary-string equivalence check (see L-011). The
        Protocol-level tokenizer_id comparison is the safe default
        when the engine has no tokenizer object to inspect.
        """
        return self.tokenizer_id == other.tokenizer_id

    def _token_count_impl(self, text: str) -> int:
        """Default: UTF-8 byte length. Override when your backend has
        a real tokenizer; the family pipeline uses this for per-probe
        normalization."""
        return len(text.encode("utf-8"))

    def max_context_length(self) -> Optional[int]:
        """Largest sequence length this engine can score without truncation.

        Default returns ``None`` (unknown / unlimited). Custom backends
        with a known limit override ``_max_context_impl`` to expose it
        — e.g. an API engine that knows its provider's hard cap, or a
        local model wrapper that reads a config field. v0.4.1
        measurement validity framework consumes this to flag
        out-of-context probes.
        """
        return self._max_context_impl()

    def _max_context_impl(self) -> Optional[int]:
        """Default: ``None`` (unknown / unlimited). Override to expose
        the engine's max scoreable sequence length. Returning an int
        enables validity filtering in the family pipeline; ``None``
        keeps every probe valid for this engine.
        v0.4.1.
        """
        return None

    # ── Required methods ──────────────────────────────────────────────

    def score(self, prompt: str, continuation: str) -> ScoreResult:
        if "score" not in self._capabilities:
            raise NotImplementedError(
                f"Engine {self.name} does not support `score` capability."
            )
        logprobs, tokens = self._score_impl(prompt, continuation)
        n = len(logprobs)
        avg = float(sum(logprobs) / n) if n > 0 else 0.0
        return ScoreResult(logprobs=logprobs, tokens=list(tokens), avg_logprob=avg)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: Optional[int] = None,
    ) -> GenerateResult:
        if "generate" not in self._capabilities:
            raise NotImplementedError(
                f"Engine {self.name} does not support `generate` capability."
            )
        # ``top_k`` accepted for Engine-Protocol parity (HFEngine + the
        # v0.4.0 pipeline pass it for sample-decode variants). Subclasses
        # that override ``_generate_impl`` may consume it; the default
        # implementation does not.
        del top_k
        text, tokens, logprobs = self._generate_impl(
            prompt, max_new_tokens, temperature, top_p, seed,
        )
        return GenerateResult(text=text, tokens=list(tokens), logprobs=logprobs)

    # ── Optional methods (raise unless overridden) ─────────────────────

    def hidden_states(
        self,
        prompt: str,
        *,
        layers: Optional[list[int]] = None,
        position: str = "last",
    ) -> HiddenStatesResult:
        raise NotImplementedError(
            f"Engine {self.name} does not support hidden_states. "
            f"Override this method to enable."
        )

    def attention_weights(
        self,
        prompt: str,
        *,
        layers: Optional[list[int]] = None,
        heads: Optional[list[int]] = None,
    ) -> AttentionWeightsResult:
        raise NotImplementedError(
            f"Engine {self.name} does not support attention_weights. "
            f"Override this method to enable."
        )

    def apply_steering(
        self,
        prompt: str,
        steering_spec: Any,
        *,
        max_new_tokens: int = 16,
    ) -> GenerateResult:
        raise NotImplementedError(
            f"Engine {self.name} does not support steering."
        )

    def extract_steering_vector(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
        *,
        layer: int,
    ) -> Any:
        raise NotImplementedError(
            f"Engine {self.name} does not support steering vector extraction."
        )

    def close(self) -> None:
        """Override if your backend needs explicit cleanup. No-op by default."""
        pass

    # ── Override these in your subclass ────────────────────────────────

    def _score_impl(self, prompt: str, continuation: str) -> tuple[Any, Any]:
        """Override: return ``(per_token_logprobs, token_ids)``.

        Returns
        -------
        logprobs : array-like of float
            Per-token log-probabilities of the continuation.
        tokens : array-like of int
            Token IDs corresponding to ``logprobs``.
        """
        raise NotImplementedError(
            "Override _score_impl in your subclass. "
            "Return (logprobs, token_ids) for the continuation."
        )

    def _generate_impl(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: Optional[int],
    ) -> tuple[str, Any, Any]:
        """Override: return ``(text, token_ids, logprobs_or_None)``.

        Returns
        -------
        text : str
            Generated continuation text (no prompt).
        token_ids : array-like of int
            Token IDs for the generated text.
        logprobs : array-like of float | None
            Per-token logprobs of the generated tokens, or ``None`` if
            your backend doesn't expose them.
        """
        raise NotImplementedError(
            "Override _generate_impl in your subclass. "
            "Return (text, token_ids, logprobs_or_None)."
        )
