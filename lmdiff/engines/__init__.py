"""Engine implementations beyond the canonical :class:`HFEngine`.

Currently:

  - :class:`MinimalEngine` — copy-paste template for custom backends

Future (Phase 6):

  - ``APIEngine`` — hosted-API backends (OpenAI, Anthropic)
  - ``vLLMEngine`` — high-throughput batched inference
"""
from lmdiff.engines.minimal import MinimalEngine

__all__ = ["MinimalEngine"]
