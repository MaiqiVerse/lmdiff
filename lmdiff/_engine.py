"""Engine protocol — the abstraction between lmdiff metrics and backends.

``Engine`` is a :pep:`544` Protocol: any class with the right methods conforms
via duck typing — no subclassing required. This lets users plug in custom
backends (vLLM, TGI, hosted APIs, TransformerLens) without inheriting.

Capability negotiation:
  - Each Engine declares which capabilities it supports via ``capabilities``.
  - Each Metric (Phase 4+) declares which capabilities it requires.
  - Mismatch raises :class:`CapabilityError` with an actionable suggestion
    (not silent garbage).

Reserved capability names live in ``RESERVED_CAPABILITIES`` to prevent
naming drift across backends.

torch / transformers are imported lazily inside ``HFEngine.__init__`` and
methods. ``from lmdiff import Engine, HFEngine, MinimalEngine`` does NOT
trigger a torch import — only ``HFEngine(...)`` instantiation does.
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional, Protocol, runtime_checkable

# torch / transformers are imported lazily inside HFEngine. Module-level
# imports MUST stay torch-free so `from lmdiff import Engine` is cheap.

from ._config import Config

__all__ = [
    "Engine",
    "HFEngine",
    "ScoreResult",
    "GenerateResult",
    "HiddenStatesResult",
    "AttentionWeightsResult",
    "CapabilityError",
    "CrossTokenizerError",
    "RESERVED_CAPABILITIES",
]


# ── Reserved capability registry ──────────────────────────────────────


RESERVED_CAPABILITIES: frozenset[str] = frozenset({
    # Implemented v0.3-v0.7
    "score", "generate", "hidden_states", "attention_weights",
    "logprobs_full", "batch", "steering",
    # Reserved for v0.8+
    "sampling_cloud",
    # Reserved for v2.0+
    "model_weights", "patch_activations", "capture_activations", "agentic",
})


# ── Errors ────────────────────────────────────────────────────────────


class CapabilityError(Exception):
    """Raised when a metric requires a capability the Engine doesn't support.

    Carries the missing capability set + engine + metric names so callers
    can format their own messages. The default message lists the missing
    capabilities and suggests two recovery paths (different engine,
    different metric set).
    """

    def __init__(self, missing: set[str], engine_name: str, metric_name: str):
        self.missing = set(missing)
        self.engine_name = engine_name
        self.metric_name = metric_name
        super().__init__(
            f"Metric `{metric_name}` requires capabilities {self.missing} "
            f"that engine `{engine_name}` does not support. "
            f"To fix: either use a different engine, or skip this metric "
            f"by passing metrics=['default'] excluding `{metric_name}`."
        )


class CrossTokenizerError(Exception):
    """Raised when comparing configs with different tokenizers in a context
    that requires the same tokenizer (e.g. CKA, token-level cosine).

    Behavioral metrics (BD, KL with BPB normalization) handle the
    cross-tokenizer case gracefully and do not raise this; only metrics
    that operate at the token-id level do.
    """


# ── Result types ──────────────────────────────────────────────────────


class ScoreResult:
    """Result of a :meth:`Engine.score` call.

    Attributes
    ----------
    logprobs : numpy.ndarray
        Shape ``(n_tokens,)``. Per-token log P(token | prefix) for the
        continuation tokens only (not the prompt).
    tokens : list[int]
        Token IDs for the continuation, aligned with ``logprobs``.
    avg_logprob : float
        Mean of ``logprobs``. ``0.0`` for an empty continuation.
    """

    __slots__ = ("logprobs", "tokens", "avg_logprob")

    def __init__(self, logprobs: Any, tokens: list[int], avg_logprob: float) -> None:
        self.logprobs = logprobs
        self.tokens = tokens
        self.avg_logprob = avg_logprob

    def __repr__(self) -> str:
        return (
            f"ScoreResult(n={len(self.tokens)}, "
            f"avg_logprob={self.avg_logprob:.4f})"
        )


class GenerateResult:
    """Result of a :meth:`Engine.generate` call.

    Attributes
    ----------
    text : str
        Decoded continuation (does not include the prompt).
    tokens : list[int]
        Token IDs for the generated continuation.
    logprobs : numpy.ndarray | None
        Per-token log-probability of each chosen token. May be ``None``
        if the backend doesn't expose them.
    full_logprobs : numpy.ndarray | None
        Full vocabulary log-distribution at each generated position.
        Only populated when the engine has ``logprobs_full`` capability
        AND the caller requested them. May be ``None`` for memory reasons.
    """

    __slots__ = ("text", "tokens", "logprobs", "full_logprobs")

    def __init__(
        self,
        text: str,
        tokens: list[int],
        logprobs: Any = None,
        full_logprobs: Any = None,
    ) -> None:
        self.text = text
        self.tokens = tokens
        self.logprobs = logprobs
        self.full_logprobs = full_logprobs

    def __repr__(self) -> str:
        snippet = self.text[:30].replace("\n", " ")
        return f"GenerateResult(text={snippet!r}..., n_tokens={len(self.tokens)})"


class HiddenStatesResult:
    """Result of a :meth:`Engine.hidden_states` call.

    Attributes
    ----------
    hidden_states : numpy.ndarray
        Shape ``(n_layers, hidden_dim)`` for a single position, or
        ``(n_layers, seq_len, hidden_dim)`` if a future caller requests
        all positions.
    position : str
        Which token position was extracted (``"last"``, ``"first"``, or
        a stringified integer index).
    """

    __slots__ = ("hidden_states", "position")

    def __init__(self, hidden_states: Any, position: str) -> None:
        self.hidden_states = hidden_states
        self.position = position


class AttentionWeightsResult:
    """Result of an :meth:`Engine.attention_weights` call.

    Attributes
    ----------
    attention_weights : numpy.ndarray
        Shape ``(n_layers, n_heads, n_tokens, n_tokens)``. Each row over
        the last dimension sums to 1 (softmax-normalized attention).
    """

    __slots__ = ("attention_weights",)

    def __init__(self, attention_weights: Any) -> None:
        self.attention_weights = attention_weights


# ── Engine Protocol ───────────────────────────────────────────────────


@runtime_checkable
class Engine(Protocol):
    """Engine protocol — backend-agnostic interface to model inference.

    To implement a custom Engine, conform to this protocol:

      1. Provide ``name``, ``tokenizer_id``, ``n_layers``, ``hidden_dim``,
         ``capabilities`` properties.
      2. Implement ``score()``, ``generate()``, ``close()``.
      3. Optionally implement ``hidden_states()``, ``attention_weights()``,
         ``apply_steering()``, ``extract_steering_vector()``.

    See :class:`MinimalEngine` (in ``lmdiff.engines.minimal``) for a
    copy-paste template, and :class:`HFEngine` for the canonical Hugging
    Face Transformers implementation.
    """

    @property
    def name(self) -> str:
        """Display name of the engine + model (e.g. ``'gpt2'``)."""
        ...

    @property
    def tokenizer_id(self) -> str:
        """Stable hash of vocab + special tokens.

        Two engines with the same ``tokenizer_id`` can be compared at the
        token level (CKA, token-level cosine). Different ids → cross-
        tokenizer comparison; falls back to BPB normalization.
        """
        ...

    @property
    def n_layers(self) -> int:
        """Number of transformer layers."""
        ...

    @property
    def hidden_dim(self) -> int:
        """Hidden state dimension."""
        ...

    @property
    def capabilities(self) -> frozenset[str]:
        """Set of capability names this engine supports.

        See ``RESERVED_CAPABILITIES`` for the registered list.
        """
        ...

    # Required methods

    def score(self, prompt: str, continuation: str) -> ScoreResult:
        """Compute log-probability of ``continuation`` given ``prompt``."""
        ...

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> GenerateResult:
        """Generate text continuing from ``prompt``."""
        ...

    def close(self) -> None:
        """Release model resources. Idempotent."""
        ...

    # Optional methods (gated by `capabilities`)

    def hidden_states(
        self,
        prompt: str,
        *,
        layers: Optional[list[int]] = None,
        position: str = "last",
    ) -> HiddenStatesResult:
        """Extract hidden states. Requires ``hidden_states`` capability."""
        raise NotImplementedError(
            "This engine does not support hidden_states extraction. "
            "Add 'hidden_states' to capabilities to enable."
        )

    def attention_weights(
        self,
        prompt: str,
        *,
        layers: Optional[list[int]] = None,
        heads: Optional[list[int]] = None,
    ) -> AttentionWeightsResult:
        """Extract attention weights. Requires ``attention_weights`` capability."""
        raise NotImplementedError(
            "This engine does not support attention_weights extraction."
        )

    def apply_steering(
        self,
        prompt: str,
        steering_spec: Any,
        *,
        max_new_tokens: int = 16,
    ) -> GenerateResult:
        """Generate with a steering vector applied. Requires ``steering`` capability.

        v0.3.0: not implemented (raises ``NotImplementedError``).
        v0.7.0 (Phase 5 commit 5.6): full HFEngine implementation.
        """
        raise NotImplementedError(
            "This engine does not support steering. Available in v0.7.0 (HFEngine)."
        )

    def extract_steering_vector(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
        *,
        layer: int,
    ) -> Any:
        """Extract a contrastive steering vector. Requires ``steering`` capability.

        v0.3.0: not implemented. v0.7.0 (Phase 5 commit 5.6): full implementation.
        """
        raise NotImplementedError(
            "This engine does not support steering vector extraction. "
            "Available in v0.7.0 (HFEngine)."
        )


# ── HFEngine — canonical Hugging Face implementation ──────────────────


class HFEngine:
    """Engine backed by Hugging Face Transformers.

    Loads model + tokenizer at ``__init__`` time. Applies any
    ``Config``-specified weight modifications (adapter, quantization,
    pruning) at load time. Inference-time modifications (KV-cache
    compression, soft prompts, steering) are applied on the relevant
    method calls.

    .. warning::
       HFEngine is **NOT thread-safe**. A single instance must be used by
       a single thread. For parallelism, use ``accelerate`` (Phase 5) or
       run multiple HFEngine instances in separate processes.

    Note
    ----
    ``torch`` and ``transformers`` are imported lazily — only when this
    class is instantiated. Importing ``HFEngine`` from ``lmdiff`` does
    not load torch.

    Examples
    --------
    >>> from lmdiff import Config, HFEngine  # doctest: +SKIP
    >>> engine = HFEngine(Config(model="gpt2"))  # doctest: +SKIP
    >>> result = engine.score("The capital of France is", " Paris")  # doctest: +SKIP
    >>> result.avg_logprob  # doctest: +SKIP
    -2.3
    """

    def __init__(
        self,
        config: Config,
        *,
        device: str = "auto",
        dtype: Optional[str] = None,
        capabilities: Optional[frozenset[str]] = None,
    ) -> None:
        # Lazy imports — torch/transformers stay out of module load
        import torch  # noqa: F401  (sanity: triggers the dependency presence check)
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

        self._config = config
        self._device = device
        self._dtype = dtype or "bf16"

        self._tokenizer = AutoTokenizer.from_pretrained(config.model)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._tokenizer_id = (
            config.tokenizer_id_override
            or self._compute_tokenizer_id(self._tokenizer)
        )

        self._model = self._load_model(config)
        self._model.eval()

        self._n_layers = self._model.config.num_hidden_layers
        self._hidden_dim = self._model.config.hidden_size

        if capabilities is not None:
            unknown = set(capabilities) - RESERVED_CAPABILITIES
            if unknown:
                raise ValueError(
                    f"Unknown capability names: {unknown}. "
                    f"See lmdiff._engine.RESERVED_CAPABILITIES for the list."
                )
            self._capabilities = frozenset(capabilities)
        else:
            self._capabilities = self._infer_capabilities()

    # ── Properties ─────────────────────────────────────────────────────

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

    # ── Loading helpers ────────────────────────────────────────────────

    @staticmethod
    def _compute_tokenizer_id(tokenizer: Any) -> str:
        """Compute a stable 16-char hex id from vocab + special tokens.

        Two ``gpt2`` tokenizers loaded in different processes produce the
        same id; ``gpt2`` and ``qwen`` produce different ids.
        """
        vocab = tokenizer.get_vocab()
        special = tokenizer.all_special_tokens or []
        vocab_str = "|".join(f"{tok}:{idx}" for tok, idx in sorted(vocab.items()))
        special_str = "|".join(sorted(special))
        h = hashlib.sha256(f"{vocab_str}__{special_str}".encode("utf-8")).hexdigest()
        return h[:16]

    def _load_model(self, config: Config) -> Any:
        """Load model applying adapter / quantization specified in Config.

        Order of operations:

          1. Base model load (with quantization kwargs if requested).
          2. PEFT adapter on top, if ``config.adapter`` is set.
          3. Pruning ``type="preloaded"`` means the model path is already
             pruned; nothing extra to do.
          4. KV-cache compression and soft prompts apply at inference
             time, not load time.
        """
        import torch
        from transformers import AutoModelForCausalLM

        load_kwargs: dict[str, Any] = {}
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }

        if config.quantization is not None:
            qs = config.quantization
            if qs.method == "int8":
                load_kwargs["load_in_8bit"] = True
            elif qs.method == "int4":
                load_kwargs["load_in_4bit"] = True
            elif qs.method in ("gptq", "awq"):
                raise NotImplementedError(
                    f"{qs.method} loading not yet implemented in v0.3.0. "
                    f"Tracked for Phase 5."
                )
            elif qs.method == "fp8":
                raise NotImplementedError(
                    "fp8 loading not yet implemented in v0.3.0."
                )
            load_kwargs["torch_dtype"] = dtype_map[qs.compute_dtype]
        else:
            load_kwargs["torch_dtype"] = dtype_map.get(self._dtype, torch.bfloat16)

        load_kwargs["device_map"] = (
            "auto" if self._device == "auto" else self._device
        )

        model = AutoModelForCausalLM.from_pretrained(config.model, **load_kwargs)

        if config.adapter is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "Adapter loading requires `peft`. Install with `pip install peft`."
                ) from exc
            model = PeftModel.from_pretrained(model, config.adapter.path)

        return model

    def _infer_capabilities(self) -> frozenset[str]:
        """Default capability set for HF causal LMs.

        ``steering`` is reserved but not yet implemented in v0.3.0, so it's
        excluded. ``sampling_cloud`` is handled at the metric layer via
        ``generate()``; the engine doesn't need a separate capability for it
        in v0.3.0.
        """
        return frozenset({
            "score", "generate", "hidden_states", "attention_weights",
            "logprobs_full", "batch",
        })

    # ── Required methods ──────────────────────────────────────────────

    def score(self, prompt: str, continuation: str) -> ScoreResult:
        """Compute log-probability of ``continuation`` given ``prompt``.

        Returns per-token logprobs at each position of the continuation.
        """
        import numpy as np
        import torch

        prompt_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        full_ids = self._tokenizer(prompt + continuation, return_tensors="pt").input_ids

        cont_start = prompt_ids.shape[1]
        cont_token_ids = full_ids[0, cont_start:].tolist()

        if len(cont_token_ids) == 0:
            return ScoreResult(
                logprobs=np.array([], dtype=np.float32),
                tokens=[],
                avg_logprob=0.0,
            )

        device = next(self._model.parameters()).device
        full_ids_dev = full_ids.to(device)

        with torch.no_grad():
            outputs = self._model(full_ids_dev)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        logprobs_all = torch.log_softmax(logits, dim=-1)

        per_token_logprobs: list[float] = []
        for i, tok_id in enumerate(cont_token_ids):
            position = cont_start + i - 1  # logits at t predict token t+1
            per_token_logprobs.append(
                float(logprobs_all[0, position, tok_id].item())
            )

        arr = np.asarray(per_token_logprobs, dtype=np.float32)
        return ScoreResult(
            logprobs=arr,
            tokens=cont_token_ids,
            avg_logprob=float(arr.mean()),
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> GenerateResult:
        """Generate text continuing from ``prompt``."""
        import numpy as np
        import torch

        if seed is not None:
            torch.manual_seed(seed)

        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)
        prompt_len = input_ids.shape[1]

        # do_sample iff caller asked for non-greedy behavior.
        do_sample = (temperature != 1.0) or (top_p < 1.0)

        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        new_token_ids = output.sequences[0, prompt_len:].tolist()
        text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)

        logprobs: Any = None
        if output.scores is not None:
            logprobs_list: list[float] = []
            for step, score_logits in enumerate(output.scores):
                if step >= len(new_token_ids):
                    break
                lp = torch.log_softmax(score_logits[0], dim=-1)
                logprobs_list.append(float(lp[new_token_ids[step]].item()))
            logprobs = np.asarray(logprobs_list, dtype=np.float32)

        return GenerateResult(text=text, tokens=new_token_ids, logprobs=logprobs)

    def hidden_states(
        self,
        prompt: str,
        *,
        layers: Optional[list[int]] = None,
        position: str = "last",
    ) -> HiddenStatesResult:
        """Extract per-layer hidden states at ``position``."""
        if "hidden_states" not in self._capabilities:
            raise NotImplementedError("hidden_states capability not enabled.")

        import numpy as np
        import torch

        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = self._model(
                input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        # outputs.hidden_states is a tuple of n_layers + 1 tensors
        # (the first entry is the embedding output).
        all_hidden = outputs.hidden_states

        layers_to_use = (
            list(range(1, len(all_hidden))) if layers is None else list(layers)
        )

        if position == "last":
            pos_idx = -1
        elif position == "first":
            pos_idx = 0
        else:
            try:
                pos_idx = int(position)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Unknown position {position!r}; expected 'last', 'first', "
                    f"or an int index."
                ) from exc

        result = np.stack(
            [
                all_hidden[layer][0, pos_idx].float().cpu().numpy()
                for layer in layers_to_use
            ],
            axis=0,
        )
        return HiddenStatesResult(hidden_states=result, position=position)

    def attention_weights(
        self,
        prompt: str,
        *,
        layers: Optional[list[int]] = None,
        heads: Optional[list[int]] = None,
    ) -> AttentionWeightsResult:
        """Extract attention weights for the requested layers / heads."""
        if "attention_weights" not in self._capabilities:
            raise NotImplementedError("attention_weights capability not enabled.")

        import numpy as np
        import torch

        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = self._model(
                input_ids,
                output_attentions=True,
                return_dict=True,
            )

        all_attn = outputs.attentions  # tuple of (n_layers,), each (1, heads, seq, seq)

        layers_to_use = (
            list(range(len(all_attn))) if layers is None else list(layers)
        )
        layer_tensors = [all_attn[i][0].float().cpu().numpy() for i in layers_to_use]
        if heads is not None:
            layer_tensors = [t[heads, :, :] for t in layer_tensors]

        result = np.stack(layer_tensors, axis=0)
        return AttentionWeightsResult(attention_weights=result)

    def close(self) -> None:
        """Release model + tokenizer resources. Idempotent."""
        if getattr(self, "_model", None) is not None:
            del self._model
            self._model = None
        if getattr(self, "_tokenizer", None) is not None:
            del self._tokenizer
            self._tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __del__(self) -> None:  # pragma: no cover - finalizer best-effort
        try:
            self.close()
        except Exception:
            pass

    def with_config(self, config: Config) -> "HFEngine":
        """Return a fresh ``HFEngine`` with a different ``Config``.

        v0.3.0: reloads the model from scratch; no weight sharing. Phase 5
        may add weight sharing when only adapter / decode changes.
        """
        return HFEngine(
            config,
            device=self._device,
            dtype=self._dtype,
            capabilities=self._capabilities,
        )
