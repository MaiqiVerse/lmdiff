from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import transformers

from lmdiff.config import Config


@dataclass
class GenerationResult:
    """Output of InferenceEngine.generate()."""
    prompts: list[str]
    completions: list[list[str]]
    token_ids: list[list[list[int]]] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ForwardResult:
    """Output of InferenceEngine.score() and get_logits().

    probe_slices (get_logits only) locate the probe tokens within input_ids:
    input_ids[probe_slice] == probe token ids. To get the logits that predict
    probe tokens, index logits with slice(probe_slice.start - 1, probe_slice.stop - 1).
    When probe_slice.start == 0 (no prefix, no BOS), the first probe token has
    no preceding position, so only len(probe) - 1 predictions are available.
    """
    prompts: list[str]
    log_probs: list[np.ndarray] | None = None
    logits: list[torch.Tensor] | None = None
    token_ids: list[list[int]] | list[list[list[int]]] | None = None
    cross_entropies: list[float] | None = None
    probe_slices: list[slice] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class HiddenStatesResult:
    """Output of InferenceEngine.forward_with_hidden()."""
    prompts: list[str]
    hidden_states: dict[int, list[torch.Tensor]]
    metadata: dict = field(default_factory=dict)


class InferenceEngine:
    """The ONLY layer that loads models and runs inference."""

    def __init__(self, config: Config, device: str | None = None) -> None:
        from lmdiff._progress import lifecycle_log

        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        lifecycle_log(
            "InferenceEngine.init",
            model=config.model if isinstance(config.model, str) else type(config.model).__name__,
            name=config.display_name,
        )
        self._model, self._tokenizer = self._load(config)
        # device_map="auto" can shard layers across cuda + cpu when GPU
        # memory is tight (notably after several sequential 7B loads).
        # Anchor self.device to wherever the input-embedding layer actually
        # ended up — that's where we must place input_ids — so .generate()
        # / .score() don't hit cross-device index_select errors.
        try:
            embed = self._model.get_input_embeddings()
            self.device = str(embed.weight.device)
        except (AttributeError, StopIteration):
            pass

    def _load(
        self, config: Config
    ) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase]:
        if isinstance(config.model, str):
            # Visibility: the transformers ``Loading checkpoint shards``
            # progress bar shows percent but not which model. In a
            # multi-variant family run that means a flat sequence of
            # identical-looking bars — impossible to tell which variant
            # is currently loading. Print the model id up front. Always
            # on (cheap, single line) — not gated behind tqdm/progress.
            print(f"[lmdiff] loading weights: {config.model}", flush=True)
            tokenizer = transformers.AutoTokenizer.from_pretrained(config.model)
            load_kwargs: dict[str, Any] = {}
            if config.dtype:
                load_kwargs["torch_dtype"] = getattr(torch, config.dtype)
            elif self.device == "cuda":
                load_kwargs["torch_dtype"] = torch.bfloat16
            else:
                load_kwargs["torch_dtype"] = torch.float32
            if self.device == "cuda":
                load_kwargs["device_map"] = "auto"
            model = transformers.AutoModelForCausalLM.from_pretrained(
                config.model, **load_kwargs,
            )
            if self.device != "cuda":
                model = model.to(self.device)
        else:
            model = config.model
            tokenizer = transformers.AutoTokenizer.from_pretrained(model.config._name_or_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        return model, tokenizer

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def model_name(self) -> str:
        return self.config.display_name

    # ── Runtime-param resolution helpers ────────────────────────────────
    #
    # These resolve runtime-only Config fields (system_prompt, context,
    # decode) from either an explicit per-call override or self.config.
    # Centralised so the engine can be SHARED across Configs that differ
    # only in these fields — a single loaded engine serves both, and
    # callers pass the per-config runtime params at score()/generate()
    # time. See ``_config.RUNTIME_ONLY_FIELDS`` and the v0.3.2
    # engine-reuse work in ``ChangeGeometry.analyze``.

    def _resolve_system_prompt(self, override: str | None) -> str | None:
        return override if override is not None else self.config.system_prompt

    def _resolve_context(
        self, override: list[dict] | None
    ) -> list[dict] | None:
        return override if override is not None else self.config.context

    def _resolve_decode(self, override: dict | None) -> dict:
        return override if override is not None else self.config.decode

    def _build_prompt(
        self,
        text: str,
        *,
        system_prompt: str | None = None,
        context: list[dict] | None = None,
    ) -> str:
        # ``None`` falls back to self.config so legacy callers
        # (e.g. forward_with_hidden) keep working unchanged.
        sp = self._resolve_system_prompt(system_prompt)
        ctx = self._resolve_context(context)
        parts: list[str] = []
        if sp:
            parts.append(sp)
        if ctx:
            for msg in ctx:
                parts.append(msg.get("content", ""))
        parts.append(text)
        return "\n".join(parts)

    def _prefix_text(
        self,
        *,
        system_prompt: str | None = None,
        context: list[dict] | None = None,
    ) -> str:
        """Text before the probe, or '' if no system_prompt/context."""
        sp = self._resolve_system_prompt(system_prompt)
        ctx = self._resolve_context(context)
        parts: list[str] = []
        if sp:
            parts.append(sp)
        if ctx:
            for msg in ctx:
                parts.append(msg.get("content", ""))
        if not parts:
            return ""
        return "\n".join(parts) + "\n"

    def _encode_for_model(
        self,
        probe_text: str,
        *,
        system_prompt: str | None = None,
        context: list[dict] | None = None,
    ) -> tuple[list[int], slice]:
        """Segment-tokenize prefix and probe separately, concat ids.

        Ensures the probe occupies exactly len(probe_ids) positions regardless
        of prefix content, so position-aligned metrics (TokenKL, TokenEntropy)
        can compare the probe span across configs with different prefixes
        without BPE-boundary drift.

        ``system_prompt`` / ``context`` default to this engine's stored
        config; pass explicit values to override per call (engine reuse).

        Returns (full_ids, probe_slice) where full_ids = prefix_ids + probe_ids
        and input_ids[probe_slice] == probe_ids.
        """
        prefix_text = self._prefix_text(
            system_prompt=system_prompt, context=context,
        )
        prefix_ids = self._tokenizer(prefix_text, add_special_tokens=True)["input_ids"]
        probe_ids = self._tokenizer(probe_text, add_special_tokens=False)["input_ids"]
        full_ids = list(prefix_ids) + list(probe_ids)
        probe_slice = slice(len(prefix_ids), len(full_ids))
        return full_ids, probe_slice

    def _decode_params(self, decode: dict | None = None) -> dict[str, Any]:
        d = decode if decode is not None else self.config.decode
        strategy = d.get("strategy", "greedy")
        params: dict[str, Any] = {}
        if strategy == "greedy":
            params["do_sample"] = False
        elif strategy == "sample":
            params["do_sample"] = True
            params["temperature"] = d.get("temperature", 1.0)
            params["top_p"] = d.get("top_p", 1.0)
            params["top_k"] = d.get("top_k", 0)
        return params

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        n_samples: int = 1,
        max_new_tokens: int = 64,
        *,
        system_prompt: str | None = None,
        context: list[dict] | None = None,
        decode: dict | None = None,
        progress: bool | None = None,
        progress_desc: str = "generate",
    ) -> GenerationResult:
        """Generate completions for a list of prompts.

        Runtime-only Config fields can be passed as kwargs to override
        the engine's stored config for this call — required when the
        same engine is shared across configs that differ only in
        runtime params (the v0.3.2 engine-reuse pathway in
        ``ChangeGeometry.analyze``). When ``None``, the engine's stored
        ``self.config`` value is used (the v0.3.0 / v0.3.1 behaviour).

        Set ``progress=True`` to render a per-probe progress bar with
        elapsed time and ETA — useful for long ``family()`` runs where
        a single ``.generate(500_prompts)`` call can take 30+ minutes
        with no output. Default behaviour is to auto-enable on a tty
        and stay silent in pipelines / when stdout is redirected.
        See :func:`lmdiff._progress.iterate` for env-var override.
        """
        from lmdiff._progress import iterate as _progress_iter

        decode_params = self._decode_params(decode)

        if n_samples > 1 and not decode_params.get("do_sample", False):
            raise ValueError(
                "n_samples > 1 requires a sampling decode strategy, "
                "but current strategy is greedy"
            )

        all_completions: list[list[str]] = []
        all_token_ids: list[list[list[int]]] = []

        for probe in _progress_iter(
            prompts, desc=progress_desc, total=len(prompts), enable=progress,
        ):
            full_ids, _ = self._encode_for_model(
                probe, system_prompt=system_prompt, context=context,
            )
            input_ids = torch.tensor([full_ids], device=self.device)
            attention_mask = torch.ones_like(input_ids)
            prompt_len = input_ids.shape[1]

            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n_samples,
                **decode_params,
            )

            samples: list[str] = []
            sample_ids: list[list[int]] = []
            for seq in outputs:
                new_tokens = seq[prompt_len:]
                samples.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True))
                sample_ids.append(new_tokens.tolist())

            all_completions.append(samples)
            all_token_ids.append(sample_ids)

        return GenerationResult(
            prompts=prompts,
            completions=all_completions,
            token_ids=all_token_ids,
        )

    @torch.no_grad()
    def score(
        self,
        prompts: list[str],
        continuations: list[str] | None = None,
        continuation_ids: list[list[int]] | None = None,
        *,
        system_prompt: str | None = None,
        context: list[dict] | None = None,
        progress: bool | None = None,
        progress_desc: str = "score",
    ) -> ForwardResult:
        """Score continuations given prompts. Returns per-token log-probs and cross-entropy.

        Pass exactly one of continuations (strings) or continuation_ids (token id lists).
        For self-scoring (same engine that generated), prefer continuation_ids to avoid
        decode→retokenize round-trip errors.

        ``system_prompt`` / ``context`` override this engine's stored
        config for the current call — required for engine reuse where
        one engine is shared across configs that differ only in runtime
        params. When ``None``, ``self.config`` values are used.

        Pass ``progress=True`` to render a per-probe progress bar; see
        :meth:`InferenceEngine.generate` for the auto-detect rules.
        """
        from lmdiff._progress import iterate as _progress_iter

        if (continuations is None) == (continuation_ids is None):
            raise ValueError("pass exactly one of continuations or continuation_ids")
        n = len(continuations or continuation_ids)
        if len(prompts) != n:
            raise ValueError("prompts and continuations must have same length")

        all_log_probs: list[np.ndarray] = []
        all_cross_entropies: list[float] = []
        all_token_ids: list[list[int]] = []

        for idx in _progress_iter(
            range(n), desc=progress_desc, total=n, enable=progress,
        ):
            prompt_ids, _ = self._encode_for_model(
                prompts[idx], system_prompt=system_prompt, context=context,
            )

            if continuation_ids is not None:
                cont_ids = continuation_ids[idx]
            else:
                cont_ids = self._tokenizer(continuations[idx], add_special_tokens=False)["input_ids"]

            if len(cont_ids) == 0:
                all_log_probs.append(np.array([], dtype=np.float32))
                all_token_ids.append([])
                all_cross_entropies.append(float("nan"))
                continue

            full_ids = torch.tensor([prompt_ids + cont_ids], device=self.device)
            prompt_len = len(prompt_ids)

            logits = self._model(input_ids=full_ids).logits[0]
            shift_logits = logits[prompt_len - 1 : -1]
            target_ids = full_ids[0, prompt_len:]

            log_probs_all = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs_all.gather(1, target_ids.unsqueeze(1)).squeeze(1)

            lp_np = token_log_probs.cpu().float().numpy()
            all_log_probs.append(lp_np)
            all_token_ids.append(target_ids.cpu().tolist())

            n_tokens = len(target_ids)
            ce = -lp_np.sum() / n_tokens
            all_cross_entropies.append(float(ce))

        return ForwardResult(
            prompts=prompts,
            log_probs=all_log_probs,
            token_ids=all_token_ids,
            cross_entropies=all_cross_entropies,
        )

    @torch.no_grad()
    def forward_with_hidden(self, prompts: list[str], layers: list[int] | None = None) -> HiddenStatesResult:
        """Run forward pass and return hidden states at specified layers."""
        built = [self._build_prompt(p) for p in prompts]
        hidden_map: dict[int, list[torch.Tensor]] = {}

        for text in built:
            inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
            out = self._model(**inputs, output_hidden_states=True)

            all_hidden = out.hidden_states
            target_layers = layers if layers is not None else list(range(len(all_hidden)))
            for li in target_layers:
                if li not in hidden_map:
                    hidden_map[li] = []
                hidden_map[li].append(all_hidden[li][0].cpu())

        return HiddenStatesResult(prompts=prompts, hidden_states=hidden_map)

    @torch.no_grad()
    def get_logits(self, prompts: list[str], topk: int = 256) -> ForwardResult:
        """Get top-k logits for each prompt, plus the probe_slice locating probe tokens in input_ids."""
        all_logits: list[torch.Tensor] = []
        all_token_ids: list[list[int]] = []
        all_probe_slices: list[slice] = []

        for probe in prompts:
            full_ids, probe_slice = self._encode_for_model(probe)
            input_ids = torch.tensor([full_ids], device=self.device)
            logits = self._model(input_ids=input_ids).logits[0]

            if topk > 0 and topk < logits.shape[-1]:
                top_vals, top_ids = logits.topk(topk, dim=-1)
                all_logits.append(top_vals.cpu())
                all_token_ids.append(top_ids.cpu().tolist())
            else:
                # topk=0 → full vocab. Caller owns the memory budget.
                # KL-based metrics require this; ranking-only metrics should
                # use topk > 0 to avoid seq_len × vocab × 4B allocations.
                all_logits.append(logits.cpu())
                all_token_ids.append(logits.argmax(dim=-1).cpu().tolist())

            all_probe_slices.append(probe_slice)

        return ForwardResult(
            prompts=prompts,
            logits=all_logits,
            token_ids=all_token_ids,
            probe_slices=all_probe_slices,
        )


def release_cuda_cache() -> None:
    """Free cached CUDA memory. Safe no-op on CPU-only systems.

    Exposed so non-engine modules (e.g. geometry) can trigger CUDA
    memory release without importing torch directly, preserving the
    "engine.py is the only module that imports torch/transformers"
    invariant.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
