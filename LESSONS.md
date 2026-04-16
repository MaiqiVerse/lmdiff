# LESSONS.md

Append-only incident log. Each entry records a bug, artifact, or design decision that took non-trivial time to work out. Future-you (and future-Claude) should grep this before debugging anything that looks familiar.

Format: L-NNN (zero-padded, sequential), never renumber, never delete entries (strike through with ~~text~~ if superseded).

## Index

- L-001: v01 probe style artifact on base models
- L-002: TokenKL requires full vocab (not top-k)
- L-003: BD without self-entropy baseline is meaningless

---

## L-001: v01 probe style artifact on base models

**Date:** 2026-04-16  
**Phase:** 1  
**Severity:** Would have corrupted results.

**Symptom:**
- BD(gpt2, distilgpt2) on math = 2.42 nats (overall 1.37)
- CE(gpt2, distilgpt2_output) > 4.6 nats on 10/10 math probes
- Two probes with different text (17+25 vs 7*8) produced CE values identical to 4+ decimal places — effective n=1 not n=10
- 10/30 distilgpt2 outputs were pure `\n`

**Root cause:** v01 math probes were instruction-style ("What is 17 + 25? Answer with just the number."). gpt2 and distilgpt2 are base models with no instruction tuning. gpt2 echoed the instruction back; distilgpt2 emitted `\n` spam. BD was measuring distance between two degenerate-output modes, not capability difference.

**Fix:** Rewrote v01 as completion-style ("17 + 25 = "). Added "Probe design principles" section to CLAUDE.md requiring completion style for base-model probe sets.

**Diagnostic signature (match any two):**
- Two semantically different probes → identical CE to 4+ decimals
- ce_ba or ce_ab > 4 nats on closely-related models
- Output is pure whitespace/newlines or echoes the prompt
- Per-domain BD > 1 nat between distillation siblings

**Diagnostic tool:** `scripts/inspect_v01_bd.py` — dumps per-probe outputs + four CE values. Run on any new probe set before trusting aggregate BD.

**Prevented going forward:** CLAUDE.md "Probe design principles".

---

## L-002: TokenKL requires full vocab

**Date:** 2026-04-16  
**Phase:** 1  
**Severity:** Silent correctness bug.

**Symptom:** TokenKL with topk > 0 would run without error but produce nonsense — `engine.get_logits(topk=K)` returns each engine's own top-K values + indices; the top-K index sets for A and B are different, so position-wise KL over those tensors is aligning disjoint vocabulary subsets.

**Fix:** `TokenKL.compute` forces `topk=0`. Added vocab-size warning for >100k vocab (memory cost scales linearly; full-vocab KL is ~20 MB per position on gemma-sized models).

**Signature:** If you see nan/inf or bizarrely large KL values and topk was passed, this is it.

**Future:** Sparse KL over top-K union is a real optimization (Amini et al. 2025). Not implemented in Phase 1.

---

## L-003: BD without self-entropy baseline is meaningless

**Date:** 2026-04-16 (inferred from CLAUDE.md formula)  
**Phase:** 1  
**Severity:** Conceptual — affects all BD interpretations.

**Why:** Raw CE(A,B) conflates (a) how different A and B are with (b) how predictable A's output is in general. Degenerate outputs (e.g. `\n` spam) have low self-CE (~0.5) but high cross-CE (>4) — this looks like "A and B are very different" when really "A and B produce different flavors of garbage".

Subtracting self-entropy CE(X,X) from cross-entropy CE(Y,X) gives "excess surprise B has about A's output beyond A's own uncertainty" — which is what we actually want to measure.

This was validated by L-001: without baseline subtraction, the v01 math artifact would have looked like genuine 4+ nat divergence.

**Rule:** CLAUDE.md "Common mistakes to avoid" bullet: "Do NOT compute BD without self-entropy baseline subtraction."
