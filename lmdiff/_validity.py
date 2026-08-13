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

Classification is gated by :data:`DEFAULT_MIN_VALID_FRACTION`: a domain
whose measurable subset falls below the floor reports no share at all
rather than a share resting on a handful of probes. See that constant's
docstring for the rationale.

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

#: Minimum fraction of a domain's probes that must be measurable before
#: the domain is allowed to contribute a ``share_per_domain`` value.
#:
#: A domain whose valid-for-both subset falls below this floor is
#: reclassified away from ``partial`` — to ``variant_only`` when the
#: variant alone can still cover the floor, otherwise ``out_of_range``.
#: Either way the domain's ``share`` and ``pdn`` become ``None``.
#:
#: Why a floor exists at all: without one the effective threshold is
#: ``1/n`` — a single surviving probe out of a hundred would still
#: produce a share plotted next to fully-measured domains, with nothing
#: but a hatch pattern to signal that it rests on one measurement. The
#: v0.4.1 Llama-2 calibration hit exactly this: 9 of 100 long-context
#: probes fit inside the 4096-token base window, and those 9 are the
#: short left tail of the length distribution — the probes that test
#: long-context capability least. They yielded long-context shares of
#: 27.6 % (``temp_1.5``) and 18.4 % (``chat``).
#:
#: Why 0.5 specifically: it is a round majority-rule choice, not a
#: derived constant. Its defensibility is relative — any fixed floor is
#: arbitrary, but ``1/n`` is both worse and invisible. See
#: ``docs/methodology/normalization.md`` §"Minimum valid fraction".
DEFAULT_MIN_VALID_FRACTION = 0.5


def compute_domain_status(
    probes_in_domain: list[ProbeValidity],
    base_name: str,
    variant_name: str,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
) -> str:
    """Classify a (variant, domain) pair as one of four states.

    Parameters
    ----------
    probes_in_domain : list[ProbeValidity]
        Every probe assigned to this domain, pre-NaN-filter.
    base_name, variant_name : str
        Engine display names, as keyed in ``ProbeValidity.per_engine``.
    min_valid_fraction : float
        Floor in ``[0.0, 1.0]`` on the fraction of the domain's probes
        that must be measurable for the domain to report a share.
        Defaults to :data:`DEFAULT_MIN_VALID_FRACTION`. Passing ``0.0``
        disables the floor and restores the pre-v0.4.1 behaviour in
        which any single valid probe was enough to sustain ``partial``.

    Status definitions
    ------------------
    ``full``
        Every probe in the domain is valid for both base and variant.
        Domain participates fully in ``share_per_domain`` and in
        per-domain pdn / magnitudes. Reported regardless of the floor —
        100 % valid always clears it.

    ``partial``
        Domain has a mix, and the valid-for-both subset meets
        ``min_valid_fraction``. Domain participates in
        ``share_per_domain`` using only that subset.

    ``variant_only``
        The valid-for-both subset is below the floor, but the variant
        alone can measure at least ``min_valid_fraction`` of the domain.
        Base couldn't sustain the comparison; the variant side has
        signal that v0.5.0+ ``variant_only_metrics`` will surface.
        v0.4.1 excludes the domain from ``share_per_domain`` and assigns
        ``share[v][d] = None``.

    ``out_of_range``
        Every probe is invalid for every engine, or the valid-for-both
        subset is below the floor with no variant-side coverage to
        redeem it. Domain entirely excluded; ``share[v][d] = None``.

    Tie-breaking
    ------------
    Above the floor, ``partial`` wins hybrid cases (e.g. 80
    valid-for-both + 20 valid-for-variant-only): the 80 valid-for-both
    probes produce signal worth aggregating, and the 20 variant-only
    probes feed the (v0.5.0+) variant_only sub-table without affecting
    the v0.4.1 share.

    Below the floor that reasoning inverts — too thin a base to
    aggregate — so the domain drops out of the share entirely. Note this
    makes ``variant_only`` and ``out_of_range`` mean "not enough
    measurable probes to trust" rather than the literal "no measurable
    probes"; a domain with a handful of valid-for-both probes can land
    in either. Thresholding on *sufficiency* rather than *existence* is
    the point of the floor.

    Boundaries are inclusive on both sides: a fraction exactly equal to
    ``min_valid_fraction`` clears it.

    Edge: empty ``probes_in_domain`` → ``out_of_range`` (defensive;
    ``share`` for an empty domain is meaningless).

    Raises
    ------
    ValueError
        If ``min_valid_fraction`` is outside ``[0.0, 1.0]``.
    """
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError(
            f"min_valid_fraction must be in [0.0, 1.0], got {min_valid_fraction}",
        )

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

    frac_both = n_both / n
    frac_var_only = n_var_only / n

    if frac_both < min_valid_fraction:
        # Too few valid-for-both probes to aggregate. The variant side
        # may still carry enough coverage to be worth flagging for the
        # v0.5.0+ variant_only sub-table.
        if frac_var_only >= min_valid_fraction:
            return "variant_only"
        return "out_of_range"
    # Reachable only when the floor is disabled (min_valid_fraction ==
    # 0.0); with any positive floor, n_both == 0 implies frac_both == 0
    # and the branch above has already returned. Retained so that
    # min_valid_fraction=0.0 exactly reproduces pre-v0.4.1 semantics.
    if n_both == 0 and n_var_only > 0:
        return "variant_only"
    return "partial"


# ── Consumer-side filtering ──────────────────────────────────────────

#: Statuses whose (variant, domain) cells carry a real measurement and
#: may be displayed, aggregated, or ranked.
MEASURED_STATUSES: frozenset[str] = frozenset({"full", "partial"})


def is_measured(
    domain_status: Optional[dict],
    variant: str,
    domain: str,
) -> bool:
    """True iff this (variant, domain) cell carries a real measurement.

    ``domain_status`` may be ``None`` or empty — results predating the
    v0.4.1 validity framework (v0.2.x ``ChangeGeometry``, v1–v5 loads)
    have no status at all, and every cell in them is measured. An
    unknown (variant, domain) key likewise defaults to measured, so a
    consumer working from a domain list that outruns the status map
    degrades to pre-v0.4.1 behaviour rather than silently blanking.
    """
    if not domain_status:
        return True
    return domain_status.get(variant, {}).get(domain, "full") in MEASURED_STATUSES


def filter_measured_cells(
    domain_status: Optional[dict],
    per_domain: dict,
    variant: Optional[str] = None,
) -> dict:
    """Drop cells the validity framework excluded from a per-domain map.

    Accepts either shape:

    - ``{domain: value}`` — pass ``variant`` to say whose row it is.
    - ``{variant: {domain: value}}`` — pass ``variant=None`` and every
      row is filtered against its own status.

    Why this exists as one function. ``domain_heatmap()`` and
    ``magnitudes_per_task_normalized()`` are deliberately
    validity-unaware: they are raw accessors and other callers want them
    that way. Every *consumer* that turns them into a user-facing claim
    has to apply the same predicate, and by v0.4.2 that was six copies
    across ``_findings.py``, three report renderers, and three viz
    modules. Six copies of a predicate is six chances for one to drift —
    which is exactly how ``magnitudes_per_task_normalized`` kept
    computing the superseded formula after the field was corrected
    (L-035). One predicate, one edit point.

    Cells are *removed* from the returned dict rather than set to
    ``None``. Consumers that aggregate (mean, std, max, set membership)
    then exclude them by construction instead of needing their own
    NaN-skip — the distinction that made the specialization z-score
    wrong rather than merely mis-displayed.
    """
    if variant is not None:
        return {
            d: v for d, v in per_domain.items()
            if is_measured(domain_status, variant, d)
        }
    return {
        v: {d: val for d, val in row.items()
            if is_measured(domain_status, v, d)}
        for v, row in per_domain.items()
    }


__all__ = [
    "DEFAULT_MIN_VALID_FRACTION",
    "MEASURED_STATUSES",
    "EngineValidity",
    "ProbeValidity",
    "compute_domain_status",
    "filter_measured_cells",
    "is_measured",
]
