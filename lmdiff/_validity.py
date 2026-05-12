"""Per-probe measurement validity tracking — v0.4.1.

A probe is "valid" for an engine when the engine can score it without
exceeding its trained context window. Validity is per (engine, probe):
the same 9000-token long-context probe is valid for Yarn-128K
(``max_context_length() == 131072``) and out-of-range for Llama-2-7B
(``max_context_length() == 4096``).

The family pipeline builds these records before its three per-probe
sub-loops (``generate``, ``score base|v``, ``score v|v``) and skips
sub-loop work for probes flagged invalid for the relevant engine.
Resulting δ values for invalid probes are NaN; the existing global
NaN-filter (``_universally_valid_indices``) drops them from
``change_vectors``.

Aggregation in ``geometry.py`` then uses ``compute_domain_status`` to
classify each (variant, domain) pair as ``full`` / ``partial`` /
``variant_only`` / ``out_of_range``, and ``_compute_per_domain_normalized``
runs the corrected pdn formula (``sqrt(mean(δ²))``, Q9.10 Formula A)
over the *valid* probe subset.

Design rationale: see ``docs/internal/v041_validity_design.md`` §1–§2,
PHASE_PLAN_v6.md Update 5 Y.4 components 1–3, and L-033.

This module is Protocol-clean — no torch / transformers imports, no
heavy deps. Importable on a CPU-only box for serialization /
deserialization round-trips.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class EngineValidity:
    """Validity record for one (engine, probe) pair.

    Attributes
    ----------
    engine_name : str
        The engine's ``.name`` property (display name).
    max_context : int | None
        Engine's max scoreable sequence length (from
        ``Engine.max_context_length()``). ``None`` means the limit is
        unknown — every probe is treated as valid for this engine.
    T_i : int
        The probe's tokenized length **including the worst-case
        continuation budget** for this engine. Specifically:
        ``T_i = T_prefix + T_prompt + max_new_tokens`` (Q9.6 — the
        conservative pre-generation bound).
    is_valid : bool
        ``True`` iff ``max_context is None`` or ``T_i <= max_context``.
    reason : str
        One of:
        - ``"valid"`` — within context, scoreable.
        - ``"exceeds_context"`` — ``T_i > max_context``.
        - ``"unknown_limit"`` — ``max_context is None``; treated as valid
          but flagged so downstream consumers know there was no real
          check.
        - ``"tokenizer_error"`` — reserved; not currently raised.
    """

    engine_name: str
    max_context: Optional[int]
    T_i: int
    is_valid: bool
    reason: str


@dataclass(frozen=True)
class ProbeValidity:
    """Per-probe validity across all engines in a family() call.

    Attributes
    ----------
    probe_id : str
        The probe's ``.id`` from the ProbeSet.
    domain : str | None
        Denormalized from ``probe.domain`` for fast group-by-domain
        lookups in ``compute_domain_status``. ``None`` carries through
        for probes that have no assigned domain.
    per_engine : dict[str, EngineValidity]
        Keyed by engine name. Always contains the base engine's record
        plus one record for each variant engine that scored against
        this probe.
    """

    probe_id: str
    domain: Optional[str]
    per_engine: dict[str, EngineValidity] = field(default_factory=dict)

    @property
    def valid_for_all(self) -> bool:
        """True iff this probe is valid for every engine in
        ``per_engine``. Equivalent to "no engine had to skip this probe."
        """
        return all(ev.is_valid for ev in self.per_engine.values())

    def valid_for(self, engine_name: str) -> bool:
        """True iff the probe is valid for the named engine.

        Returns ``False`` when ``engine_name`` is not in ``per_engine``
        — caller asked about an engine that didn't participate in the
        family run, which is a bug we'd rather surface than silently
        treat as "valid by default."
        """
        ev = self.per_engine.get(engine_name)
        return ev is not None and ev.is_valid


# ── Domain status ────────────────────────────────────────────────────


def compute_domain_status(
    probes_in_domain: list[ProbeValidity],
    base_name: str,
    variant_name: str,
) -> str:
    """Classify a (variant, domain) pair as one of four states.

    Status definitions
    ------------------
    ``full``
        Every probe in the domain is valid for both base and variant.
        Domain participates fully in ``share_per_domain`` and in
        per-domain pdn / magnitudes.

    ``partial``
        Domain has a mix — some probes valid for both, some invalid for
        one or the other. Domain participates in ``share_per_domain``
        using only the valid-for-both subset.

    ``variant_only``
        Every probe is invalid for base, but at least one is valid for
        the variant. Base couldn't measure the comparison; the variant
        side has signal that v0.5.0+ ``variant_only_metrics`` will
        surface. v0.4.1 excludes the domain from ``share_per_domain``
        and assigns ``share[v][d] = None``.

    ``out_of_range``
        Every probe is invalid for every engine. Domain entirely
        excluded.

    Tie-breaking for hybrid cases (e.g. 80 valid-for-both + 20
    valid-for-variant-only): ``partial`` wins. Rationale per audit
    §2.1: the 80 valid-for-both probes still produce signal worth
    aggregating; the 20 variant-only probes feed the (v0.5.0+)
    variant_only sub-table without affecting the v0.4.1 share.

    Edge: empty ``probes_in_domain`` → ``out_of_range`` (defensive;
    ``share`` for an empty domain is meaningless).
    """
    n = len(probes_in_domain)
    if n == 0:
        return "out_of_range"

    base_valid = [p.valid_for(base_name) for p in probes_in_domain]
    var_valid = [p.valid_for(variant_name) for p in probes_in_domain]
    n_both = sum(1 for b, v in zip(base_valid, var_valid) if b and v)
    n_neither = sum(1 for b, v in zip(base_valid, var_valid) if not b and not v)
    n_var_only = sum(1 for b, v in zip(base_valid, var_valid) if v and not b)

    if n_both == n:
        return "full"
    if n_neither == n:
        return "out_of_range"
    if n_both == 0 and n_var_only > 0:
        return "variant_only"
    return "partial"


__all__ = [
    "EngineValidity",
    "ProbeValidity",
    "compute_domain_status",
]
