"""Configuration class — the unit of comparison in lmdiff (v0.3.0).

A ``Config`` packages all the things that can vary across an LLM inference
setup: weights, adapter, quantization, prompt, decoding, steering, etc.
Two ``Config`` s that differ in any of these slots can be compared via
``lmdiff.compare()`` or ``lmdiff.family()`` (introduced in commit 1.1).

Designed to be:
  - frozen (immutable, hashable for plain-data fields)
  - serializable to JSON via :meth:`Config.to_dict` / :meth:`Config.from_dict`
  - serializable to pickle (for multiprocessing in Phase 5)
  - lazy: does NOT trigger model loading on construction; only an Engine
    fed a ``Config`` will load. ``import lmdiff._config`` does not pull
    torch or transformers.

This module is the v0.3.0 successor to ``lmdiff.config``. The pre-v0.3.0
``Config`` (in ``lmdiff/config.py``) remains importable for backward
compatibility until v0.4.0; the deprecation shim is added in commit 1.1.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Literal, Optional

# numpy is NOT imported at module level. soft_prompts and SteeringSpec.vectors
# are typed as ``Any`` and validated lazily (at Engine load time, Phase 5).

__all__ = [
    "Config",
    "AdapterSpec",
    "QuantSpec",
    "PruneSpec",
    "ICLExample",
    "Message",
    "KVCacheSpec",
    "DecodeSpec",
    "SteeringSpec",
    "RUNTIME_ONLY_FIELDS",
    "MODEL_SPECIFIC_COMPARATORS",
]


# ── Engine-reuse classification (v0.3.2 — see CHANGELOG) ───────────────
#
# A field is in ``RUNTIME_ONLY_FIELDS`` iff changing its value can be
# applied at score()/generate() call time without re-loading weights.
# Two Configs that share ``model`` and differ only in fields from this
# set can share a single loaded engine — see
# ``Config.is_runtime_only_modification_of``.
#
# Classification (audit performed once; revisit when adding new fields):
#
#   RUNTIME-ONLY (reuse engine OK):
#     name                       — pure label, never read at inference
#     system_prompt              — concatenated into the prompt
#     icl_examples               — concatenated into the prompt
#     context                    — concatenated into the prompt
#     decode                     — passed as ``model.generate`` kwargs
#     tokenizer_id_override      — metadata id only; tokenizer object
#                                  is determined by ``model``
#     capabilities_required      — caller-side contract assertion;
#                                  doesn't change how the engine loads
#     training_recipe_summary    — pure documentation
#
#   WEIGHT-AFFECTING (force separate engine):
#     model                      — the model identity itself
#     adapter                    — load-time weight transform (LoRA etc.)
#     quantization               — load-time weight transform
#     pruning                    — load-time weight transform
#     soft_prompts               — needs an embedding tensor bound to
#                                  the model; not safe to swap per-call
#     kv_cache_compression       — installs hooks on attention layers
#     steering                   — installs hooks on forward path
#
# Default-to-strict policy: if you're unsure about a future field,
# leave it OUT of this set. Being too strict only costs reload time;
# being too lax causes silent correctness bugs (an engine running
# under the wrong weight transform).
RUNTIME_ONLY_FIELDS: frozenset[str] = frozenset({
    "name",
    "system_prompt",
    "icl_examples",
    "context",
    "decode",
    "tokenizer_id_override",
    "capabilities_required",
    "training_recipe_summary",
})


# Model-specific override hook. Empty by default. Register a comparator
# under a model id when a downstream user has added a Config field that
# needs custom reuse semantics, without touching the base classification.
#
# Example::
#
#     def my_model_compatible(self_cfg, other_cfg):
#         # both must use the same custom scaffold
#         return self_cfg.training_recipe_summary == other_cfg.training_recipe_summary
#
#     MODEL_SPECIFIC_COMPARATORS["my-org/my-model"] = my_model_compatible
MODEL_SPECIFIC_COMPARATORS: dict[
    str, Callable[["Config", "Config"], bool]
] = {}


# ── Helpers ────────────────────────────────────────────────────────────


def _coerce_metadata(value: Any) -> Optional[tuple[tuple[str, Any], ...]]:
    """Coerce a dict / list-of-pairs into a sorted tuple-of-pairs (hashable).

    ``None`` passes through. The sort key is the pair's first element
    (assumed string) so that equal-content metadata is hash-stable
    regardless of insertion order.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return tuple(sorted(value.items(), key=lambda kv: kv[0]))
    if isinstance(value, list):
        # Already a sequence of pairs — coerce each pair to a tuple.
        coerced = tuple(tuple(p) for p in value)  # type: ignore[arg-type]
        return tuple(sorted(coerced, key=lambda kv: kv[0]))
    if isinstance(value, tuple):
        return tuple(sorted(value, key=lambda kv: kv[0]))
    raise TypeError(
        f"metadata must be dict, list, tuple, or None; got {type(value).__name__}"
    )


def _is_numpy_array(value: Any) -> bool:
    """True if ``value`` quacks like a numpy array (without importing numpy)."""
    return (
        type(value).__module__.startswith("numpy")
        and type(value).__name__ == "ndarray"
    )


def _np_to_dict(arr: Any) -> dict[str, Any]:
    """Serialize numpy array to a JSON-safe dict. Numpy is imported only here."""
    return {
        "__numpy__": True,
        "data": arr.tolist(),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def _dict_to_np(d: dict[str, Any]) -> Any:
    """Reconstruct a numpy array from a ``__numpy__`` dict."""
    import numpy as np
    arr = np.asarray(d["data"], dtype=d["dtype"])
    return arr.reshape(tuple(d["shape"]))


def _looks_like_numpy_dict(value: Any) -> bool:
    return isinstance(value, dict) and value.get("__numpy__") is True


def _values_equal(a: Any, b: Any) -> bool:
    """Equality check that handles numpy arrays and dict-of-arrays.

    Plain ``a == b`` raises ``ValueError`` on numpy arrays with more
    than one element ("The truth value of an array with more than one
    element is ambiguous"). The reuse predicate must compare every
    Config field including ones that may hold numpy arrays
    (``soft_prompts``, ``SteeringSpec.vectors``), so this helper
    encodes the safe comparison rules.
    """
    if a is b:
        return True
    if a is None or b is None:
        return a is b
    a_np = _is_numpy_array(a)
    b_np = _is_numpy_array(b)
    if a_np or b_np:
        if not (a_np and b_np):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        import numpy as np
        return bool(np.array_equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_values_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b) or type(a) is not type(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b))
    try:
        return a == b
    except (ValueError, TypeError):
        return False


def _serialize_value(value: Any) -> Any:
    """Recursive JSON-safe serializer for Config field values."""
    if value is None:
        return None
    if _is_numpy_array(value):
        return _np_to_dict(value)
    if isinstance(value, dict):
        # Generic dict (e.g. SteeringSpec.vectors). Recurse on values; keys
        # are JSON-safe primitives (we don't enforce, but document expectation).
        if any(_is_numpy_array(v) for v in value.values()):
            return {
                "__numpy_dict__": True,
                "items": [
                    [k, _serialize_value(v)] for k, v in value.items()
                ],
            }
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, frozenset):
        return sorted(value)
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if is_dataclass(value):
        # Sub-spec instance — recurse field-by-field so we get our own
        # numpy / frozenset handling, not vanilla asdict().
        return {f.name: _serialize_value(getattr(value, f.name)) for f in fields(value)}
    return value


def _deserialize_subspec(cls: type, value: dict[str, Any]) -> Any:
    """Reconstruct a sub-spec dataclass from its serialized dict."""
    kwargs: dict[str, Any] = {}
    spec_fields = {f.name: f for f in fields(cls)}
    for key, raw in value.items():
        if key not in spec_fields:
            raise ValueError(f"Unknown field {key!r} for {cls.__name__}")
        if _looks_like_numpy_dict(raw):
            kwargs[key] = _dict_to_np(raw)
        elif isinstance(raw, dict) and raw.get("__numpy_dict__") is True:
            kwargs[key] = {
                k: (_dict_to_np(v) if _looks_like_numpy_dict(v) else v)
                for k, v in raw["items"]
            }
        elif isinstance(raw, list) and key == "target_modules":
            kwargs[key] = tuple(raw)
        else:
            kwargs[key] = raw
    return cls(**kwargs)


# ── Sub-specs ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AdapterSpec:
    """Adapter (LoRA / QLoRA / IA³ / prefix) loaded on top of a base model."""

    type: Literal["lora", "qlora", "ia3", "prefix"] = "lora"
    path: Optional[str] = None
    rank: Optional[int] = None
    target_modules: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.path is None:
            raise ValueError("AdapterSpec requires `path`")
        if self.type in ("lora", "qlora") and self.rank is None:
            raise ValueError(f"{self.type} adapter requires `rank` to be set")
        if isinstance(self.target_modules, list):
            object.__setattr__(self, "target_modules", tuple(self.target_modules))


@dataclass(frozen=True)
class QuantSpec:
    """Quantization spec (INT8 / INT4 / GPTQ / AWQ / FP8)."""

    method: Literal["int8", "int4", "gptq", "awq", "fp8"] = "int4"
    bits: Optional[int] = None
    compute_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    config_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.method in ("gptq", "awq") and self.config_path is None:
            raise ValueError(f"{self.method} requires `config_path`")


@dataclass(frozen=True)
class PruneSpec:
    """Pruning spec or load-already-pruned reference."""

    type: Literal["unstructured", "structured", "preloaded"] = "preloaded"
    sparsity: Optional[float] = None
    pattern: Optional[str] = None
    config_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.type == "unstructured":
            if self.sparsity is None:
                raise ValueError("unstructured pruning requires `sparsity`")
            if not 0.0 <= self.sparsity <= 1.0:
                raise ValueError(f"sparsity must be in [0, 1], got {self.sparsity}")
        if self.type == "structured" and self.pattern is None:
            raise ValueError("structured pruning requires `pattern`")
        if self.type == "preloaded" and self.config_path is None:
            raise ValueError("preloaded pruning requires `config_path`")


@dataclass(frozen=True)
class ICLExample:
    """Single in-context-learning example. Used in tuples for few-shot."""

    user: str
    assistant: str
    metadata: Optional[tuple[tuple[str, Any], ...]] = None

    def __post_init__(self) -> None:
        coerced = _coerce_metadata(self.metadata)
        if coerced is not self.metadata:
            object.__setattr__(self, "metadata", coerced)


@dataclass(frozen=True)
class Message:
    """Single message in a multi-turn context."""

    role: Literal["system", "user", "assistant"]
    content: str
    metadata: Optional[tuple[tuple[str, Any], ...]] = None

    def __post_init__(self) -> None:
        coerced = _coerce_metadata(self.metadata)
        if coerced is not self.metadata:
            object.__setattr__(self, "metadata", coerced)


@dataclass(frozen=True)
class KVCacheSpec:
    """KV-cache compression / management spec."""

    method: Literal["h2o", "kv_quant", "snapkv", "streamingllm", "none"] = "none"
    keep_ratio: Optional[float] = None
    compute_dtype: Literal["bf16", "fp16", "int8"] = "bf16"

    def __post_init__(self) -> None:
        if self.method == "none":
            return
        if self.method == "h2o" and self.keep_ratio is None:
            raise ValueError("h2o KV cache requires `keep_ratio`")
        if self.keep_ratio is not None and not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")


@dataclass(frozen=True)
class DecodeSpec:
    """Decoding strategy + parameters."""

    strategy: Literal[
        "greedy", "sample", "beam", "best_of_n", "self_consistency"
    ] = "greedy"
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    num_samples: int = 1
    max_new_tokens: int = 16
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.strategy in ("best_of_n", "self_consistency") and self.num_samples < 2:
            raise ValueError(
                f"{self.strategy} requires `num_samples` >= 2, "
                f"got {self.num_samples}"
            )
        if self.strategy == "greedy" and self.temperature != 1.0:
            warnings.warn(
                f"greedy decoding ignores temperature={self.temperature}",
                stacklevel=2,
            )
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")


@dataclass(frozen=True)
class SteeringSpec:
    """Steering tensor framework. v0.3.0 has minimal shape; Phase 5 adds
    the concrete ``Engine.apply_steering()`` integration.

    Each spec applies one or more vectors (shape: ``hidden_dim``) at the
    specified layers, optionally scaled, and applied via ``add`` or
    ``replace`` at the chosen positions.

    Note
    ----
    ``vectors`` is typed ``Any`` because the numpy import is deferred. In
    practice it's a ``dict[int, np.ndarray]`` mapping layer index to vector.
    Per-vector shape validation happens at Engine load time, not here.
    """

    vectors: Optional[Any] = None
    scale: float = 1.0
    application: Literal["add", "replace"] = "add"
    positions: Literal["all", "last", "first"] = "all"

    def __post_init__(self) -> None:
        if self.vectors is None or len(self.vectors) == 0:
            raise ValueError("SteeringSpec requires at least one vector")


# ── Main Config class ─────────────────────────────────────────────────


_SUBSPEC_TYPES: dict[str, type] = {
    "adapter": AdapterSpec,
    "quantization": QuantSpec,
    "pruning": PruneSpec,
    "kv_cache_compression": KVCacheSpec,
    "decode": DecodeSpec,
    "steering": SteeringSpec,
}


@dataclass(frozen=True)
class Config:
    """A complete LLM inference configuration.

    A ``Config`` packages all the things that can vary: weights, adapter,
    quantization, prompt, decoding, steering. Two ``Config`` s that differ
    in any field can be compared via ``lmdiff.compare()`` or
    ``lmdiff.family()``.

    Examples
    --------
    Compare two models:

    >>> a = Config(model="gpt2")
    >>> b = Config(model="distilgpt2")

    Compare with vs. without a system prompt:

    >>> base = Config(model="meta-llama/Llama-2-7b-hf")
    >>> sysprompt = Config(
    ...     model="meta-llama/Llama-2-7b-hf",
    ...     system_prompt="You are a helpful assistant.",
    ... )

    Compare different decoding strategies:

    >>> greedy = Config(model="gpt2", decode=DecodeSpec(strategy="greedy"))
    >>> sampled = Config(
    ...     model="gpt2",
    ...     decode=DecodeSpec(strategy="sample", temperature=0.7),
    ... )

    Compare with vs. without a LoRA adapter:

    >>> base = Config(model="meta-llama/Llama-2-7b-hf")
    >>> lora = Config(
    ...     model="meta-llama/Llama-2-7b-hf",
    ...     adapter=AdapterSpec(type="lora", path="path/to/lora", rank=16),
    ... )
    """

    # Identity
    model: str
    name: Optional[str] = None

    # Weights modifications
    adapter: Optional[AdapterSpec] = None
    quantization: Optional[QuantSpec] = None
    pruning: Optional[PruneSpec] = None

    # Context
    system_prompt: Optional[str] = None
    icl_examples: Optional[tuple[ICLExample, ...]] = None
    context: Optional[tuple[Message, ...]] = None
    soft_prompts: Optional[Any] = None  # numpy array; lazy validation
    kv_cache_compression: Optional[KVCacheSpec] = None

    # Decoding
    decode: DecodeSpec = field(default_factory=DecodeSpec)

    # Steering
    steering: Optional[SteeringSpec] = None

    # Metadata
    tokenizer_id_override: Optional[str] = None
    capabilities_required: frozenset[str] = field(default_factory=frozenset)
    training_recipe_summary: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("Config.model is required and must be non-empty")

        if isinstance(self.icl_examples, list):
            object.__setattr__(self, "icl_examples", tuple(self.icl_examples))
        if isinstance(self.context, list):
            object.__setattr__(self, "context", tuple(self.context))
        if isinstance(self.capabilities_required, (set, list)):
            object.__setattr__(
                self, "capabilities_required", frozenset(self.capabilities_required),
            )

    @property
    def display_name(self) -> str:
        """Human-readable name; defaults to ``model`` if ``name`` is not set."""
        return self.name if self.name else self.model

    def differs_in(self, other: "Config") -> tuple[str, ...]:
        """Return tuple of field names where this Config differs from ``other``.

        Useful for reports: "These two configs differ only in
        ``system_prompt`` and ``decode``."
        """
        diffs: list[str] = []
        for f in fields(self):
            if getattr(self, f.name) != getattr(other, f.name):
                diffs.append(f.name)
        return tuple(diffs)

    def is_runtime_only_modification_of(self, other: "Config") -> bool:
        """True iff one loaded engine can serve both ``self`` and ``other``.

        Two Configs are runtime-compatible — i.e., share-an-engine safe —
        when (a) they reference the same ``model``, and (b) every
        weight-affecting field has the same value in both. The
        weight-affecting set is the complement of
        :data:`RUNTIME_ONLY_FIELDS`. See the audit at the top of
        ``lmdiff/_config.py`` for the per-field rationale.

        For downstream models with custom Config fields requiring
        non-default reuse semantics, register a comparator in
        :data:`MODEL_SPECIFIC_COMPARATORS` keyed by model id; it bypasses
        the default check.

        Reflexive (a Config is always runtime-compatible with itself)
        and symmetric. Not transitive in the presence of model-specific
        comparators that don't satisfy transitivity.
        """
        if self.model != other.model:
            return False
        comparator = MODEL_SPECIFIC_COMPARATORS.get(self.model)
        if comparator is not None:
            return bool(comparator(self, other))
        # Check every weight-affecting field individually. We can't use
        # ``differs_in`` because some fields (``soft_prompts``,
        # ``SteeringSpec.vectors``) hold numpy arrays whose ``!=`` is
        # ambiguous — that's exactly the reason we keep them out of
        # RUNTIME_ONLY_FIELDS, and we need to *check* them here.
        for f in fields(self):
            if f.name in RUNTIME_ONLY_FIELDS or f.name == "model":
                continue
            if not _values_equal(getattr(self, f.name), getattr(other, f.name)):
                return False
        return True

    # ── Serialization ──────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Numpy arrays in ``soft_prompts`` and ``SteeringSpec.vectors`` are
        wrapped in ``{"__numpy__": True, "data": ..., "dtype": ..., "shape": ...}``.
        ``frozenset`` fields serialize to sorted lists for deterministic output.
        Reverse via :meth:`Config.from_dict`.
        """
        return {f.name: _serialize_value(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        """Reconstruct from a dict produced by :meth:`Config.to_dict`.

        Raises ``ValueError`` on unknown keys.
        """
        own_fields = {f.name: f for f in fields(cls)}
        unknown = [k for k in d if k not in own_fields]
        if unknown:
            raise ValueError(
                f"Unknown field(s) for Config: {unknown}. "
                f"Valid fields: {sorted(own_fields)}"
            )

        kwargs: dict[str, Any] = {}
        for key, raw in d.items():
            if raw is None:
                kwargs[key] = None
                continue
            # Sub-spec dict → reconstruct
            if key in _SUBSPEC_TYPES and isinstance(raw, dict):
                kwargs[key] = _deserialize_subspec(_SUBSPEC_TYPES[key], raw)
                continue
            # icl_examples / context — list of dicts → tuple of dataclasses
            if key == "icl_examples" and isinstance(raw, list):
                kwargs[key] = tuple(_deserialize_subspec(ICLExample, item) for item in raw)
                continue
            if key == "context" and isinstance(raw, list):
                kwargs[key] = tuple(_deserialize_subspec(Message, item) for item in raw)
                continue
            # Numpy
            if _looks_like_numpy_dict(raw):
                kwargs[key] = _dict_to_np(raw)
                continue
            # capabilities_required — list (from JSON) → frozenset
            if key == "capabilities_required" and isinstance(raw, list):
                kwargs[key] = frozenset(raw)
                continue
            kwargs[key] = raw
        return cls(**kwargs)

    def __repr__(self) -> str:
        """Concise repr showing only non-default fields."""
        parts = [f"model={self.model!r}"]
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        if self.adapter is not None:
            parts.append(f"adapter={self.adapter!r}")
        if self.quantization is not None:
            parts.append(f"quantization={self.quantization!r}")
        if self.pruning is not None:
            parts.append(f"pruning={self.pruning!r}")
        if self.system_prompt is not None:
            sp = self.system_prompt
            if len(sp) > 30:
                sp = sp[:27] + "..."
            parts.append(f"system_prompt={sp!r}")
        if self.icl_examples:
            parts.append(f"icl_examples=<{len(self.icl_examples)} examples>")
        if self.context:
            parts.append(f"context=<{len(self.context)} messages>")
        if self.soft_prompts is not None:
            parts.append("soft_prompts=<array>")
        if self.kv_cache_compression is not None:
            parts.append(f"kv_cache_compression={self.kv_cache_compression!r}")
        if self.decode != DecodeSpec():
            parts.append(f"decode={self.decode!r}")
        if self.steering is not None:
            parts.append("steering=<spec>")
        if self.tokenizer_id_override is not None:
            parts.append(f"tokenizer_id_override={self.tokenizer_id_override!r}")
        if self.capabilities_required:
            parts.append(f"capabilities_required={set(self.capabilities_required)!r}")
        if self.training_recipe_summary is not None:
            trs = self.training_recipe_summary
            if len(trs) > 30:
                trs = trs[:27] + "..."
            parts.append(f"training_recipe_summary={trs!r}")
        return f"Config({', '.join(parts)})"
