"""Config → prompt assembly and decode kwargs. One definition.

These three functions translate a v0.3 ``Config`` into the two things
every engine call needs: the text that precedes a probe, and the decode
parameters. They lived in ``_pipeline`` as module-private helpers until
v0.4.5, when the task layer's engine port needed them too.

**Why they had to move rather than be copied.** The legacy
``InferenceEngine`` resolves ``system_prompt`` / ``context`` / ``decode``
from its *stored config* whenever the caller passes nothing. ``HFEngine``
is stateless and takes them explicitly. So a port that does not pass them
silently drops the configuration: a ``system_prompt`` variant loses its
scaffold, and a ``temperature=1.5`` variant becomes **greedy**, because
``HFEngine.generate`` computes ``do_sample`` from ``temperature != 1.0``
and the default is ``1.0``. Neither raises. Both produce plausible
numbers for a configuration that was never measured.

Two callers needing the same translation is two implementations that
agree today and can diverge tomorrow (L-035) — and the detail most
likely to be lost is written down two lines below this: the trailing
newline in ``prefix_text`` is load-bearing for byte-equivalence with the
v0.2.x calibration baseline.

Engine-free and torch-free by construction: pure functions of a Config.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lmdiff._config import Config

__all__ = ["prefix_text", "assemble_prompt", "generate_kwargs"]


def prefix_text(config: "Config") -> str:
    """Build the prefix that precedes a probe, from a v0.3 ``Config``.

    Mirrors v0.2.x ``InferenceEngine._prefix_text`` byte-for-byte:
    ``"\\n".join([system_prompt, *context_contents]) + "\\n"`` when
    any prefix material is present; ``""`` otherwise.

    Note: keeping the trailing ``"\\n"`` matters for byte-equivalence
    with the v0.2.x calibration baseline. Don't strip.
    """
    parts: list[str] = []
    if config.system_prompt:
        parts.append(config.system_prompt)
    if config.context:
        for msg in config.context:
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if content:
                parts.append(content)
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def assemble_prompt(config: "Config", probe_text: str) -> str:
    """Concatenate prefix + probe. The Engine sees a single string."""
    return prefix_text(config) + probe_text


def generate_kwargs(config: "Config", max_new_tokens: int) -> dict[str, Any]:
    """Translate the v0.3 ``DecodeSpec`` into Engine.generate kwargs.

    The Engine Protocol's ``generate`` signature (v0.4.0) is::

        generate(prompt, *, max_new_tokens, temperature, top_p,
                 top_k, seed, prefix_text)

    Mirrors v0.2.x ``InferenceEngine._decode_params`` — temperature,
    top_p, top_k all flow through. ``top_k`` defaults to 0 (no
    filtering); HF's ``model.generate`` defaults to top_k=50 when the
    kwarg is omitted, which silently truncates sample-decode
    distributions. Passing top_k=0 explicitly is what makes ``temp_1.5``
    variants byte-equivalent to v0.3.2.

    NB: ``seed`` is **not** included here. Seed is applied once per
    variant at probe 0 by ``_delta_for_variant`` (see ``_resolve_seed``);
    repeating it on every probe call would reset RNG between probes
    and force every probe in a sampling variant to see the same RNG
    state, which is the wrong granularity (lab convention is "pin
    once per experiment, let RNG advance naturally").
    """
    decode = config.decode
    out: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if decode.strategy == "greedy":
        # Defaults are temperature=1.0, top_p=1.0 → HFEngine sets do_sample=False.
        return out
    if decode.strategy == "sample":
        out["temperature"] = decode.temperature
        out["top_p"] = decode.top_p
        out["top_k"] = decode.top_k
        return out
    # beam / best_of_n / self_consistency aren't yet wired through
    # HFEngine.generate; the v0.2.x path didn't support them either.
    # Fall through to greedy defaults for byte-equivalence with v0.3.2.
    return out
