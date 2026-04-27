from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=False)
class Config:
    """A model configuration: weights + context + decoding + adapter + agent scaffold.

    .. deprecated:: 0.3.0
       Use :class:`lmdiff.Config` (re-exported from ``lmdiff._config``) and
       its sub-spec dataclasses (``DecodeSpec``, ``AdapterSpec``, etc.).
       This class is the v0.2.x configuration object; it continues to work
       for backward compatibility but emits a ``DeprecationWarning``.
       Will be removed in v0.4.0.
    """

    model: str | Any
    context: list[dict] | None = None
    system_prompt: str | None = None
    decode: dict = field(default_factory=lambda: {"strategy": "greedy"})
    adapter: str | None = None
    ttt: dict | None = None
    agent: Any | None = None
    name: str | None = None
    use_chat_template: bool = False
    """When True, engine uses tokenizer.apply_chat_template to build prompts (Phase 2)."""
    dtype: str | None = None
    """Model precision: 'bfloat16', 'float16', 'float32', or None (auto: bfloat16 on CUDA, float32 on CPU)."""

    def __post_init__(self) -> None:
        warnings.warn(
            "lmdiff.config.Config (v0.2.x) is deprecated since v0.3.0; "
            "use `from lmdiff import Config` (the new v0.3.0 dataclass in "
            "lmdiff._config) and its DecodeSpec / AdapterSpec sub-specs instead. "
            "Will be removed in v0.4.0. See docs/migration/v02-to-v03.md.",
            DeprecationWarning,
            stacklevel=3,
        )
        if self.model is None:
            raise ValueError("model must not be None")
        if self.decode is not None and not isinstance(self.decode, dict):
            raise TypeError("decode must be a dict")
        if self.dtype is not None and self.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(
                f"dtype must be 'bfloat16', 'float16', 'float32', or None; got '{self.dtype}'"
            )

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
            "use_chat_template": self.use_chat_template,
            "dtype": self.dtype,
        }
        current.update(kwargs)
        return Config(**current)
