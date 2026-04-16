from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=False)
class Config:
    """A model configuration: weights + context + decoding + adapter + agent scaffold."""

    model: str | Any
    context: list[dict] | None = None
    system_prompt: str | None = None
    decode: dict = field(default_factory=lambda: {"strategy": "greedy"})
    adapter: str | None = None
    ttt: dict | None = None
    agent: Any | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.model is None:
            raise ValueError("model must not be None")
        if self.decode is not None and not isinstance(self.decode, dict):
            raise TypeError("decode must be a dict")

    @property
    def display_name(self) -> str:
        if self.name:
            return self.name
        if isinstance(self.model, str):
            return self.model
        return type(self.model).__name__

    def shares_tokenizer_with(self, other: Config) -> bool | None:
        """Check if two configs use the same tokenizer.

        Returns True if same model string (fast path), None if tokenizer
        loading is needed to determine equivalence. The engine layer should
        use tokenizer_utils.tokenizers_equivalent() for the full check.
        """
        if isinstance(self.model, str) and isinstance(other.model, str):
            if self.model == other.model:
                return True
        return None

    def with_override(self, **kwargs: Any) -> Config:
        """Return a new Config with specified fields overridden."""
        current = {
            "model": self.model,
            "context": self.context,
            "system_prompt": self.system_prompt,
            "decode": self.decode,
            "adapter": self.adapter,
            "ttt": self.ttt,
            "agent": self.agent,
            "name": self.name,
        }
        current.update(kwargs)
        return Config(**current)
