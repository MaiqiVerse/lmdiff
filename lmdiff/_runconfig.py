"""Run-configuration serialization — v0.4.3, Phase 2 commit 4.2.

Emits the YAML artifact that accompanies every report: the executable
configuration for a run, plus a read-only ``provenance`` block that a
loader drops. Design and rationale in
``docs/internal/v043_runconfig_design.md``; the settled decisions it
builds on are PHASE_PLAN Update 7 AA.1–AA.8.

Three properties this module exists to hold:

**The emitted YAML is runnable** (AA.2). One schema, and the artifact is
a valid input — someone reading a report copies it, edits two lines and
runs it. Runtime facts live in ``provenance`` and are ignored on load.

**Defaults are expanded** (AA.3). Every parameter affecting a computed
value is written explicitly, defaulted or not. Pinning by omission fails
when a default changes, and this project has changed a numeric default's
effective meaning twice in two minor versions.

**Serialization is one-directional** (AA.4). This is a serialization of
``Config``, not a second configuration API. ``Config.to_dict`` /
``from_dict`` do the work; this module decides what wraps them.

Execution — a loader that runs one of these files — is deliberately
later (AA.6). What ships here is emission, so the schema can be
calibrated against real reports before anything depends on parsing it.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover
    from lmdiff._config import Config

#: Run-config grammar version. Independent of ``geo_schema`` (which
#: versions an *output*) and of the package version (which moves every
#: release). See the design audit §10.5 for why all three exist: they
#: version an input, an output and a semantics, and change for unrelated
#: reasons at unrelated rates.
RUNCONFIG_SCHEMA_VERSION = 1

#: Reserved for out-of-line array payloads. The mechanism — writing
#: ``soft_prompts`` and steering vectors to sibling ``.npy`` files rather
#: than inlining megabytes of float literals — lands later (§1.2). The
#: **key is reserved now** because adding it afterwards would be a
#: grammar change and a schema bump, where reserving it costs a line.
#:
#: A config carrying a ``__ref__`` is a *bundle*: the YAML plus a sibling
#: directory. See §1.4 for what that costs and why a missing target is
#: not the same thing as ``reproducible: false``.
REF_KEY = "__ref__"

#: Keys the loader must ignore rather than treat as configuration.
PROVENANCE_KEY = "provenance"


def _variant_block(cfg: "Config") -> Any:
    """A variant as either a bare model-id or an expanded mapping.

    The string form is sugar for "this model, all defaults" and is used
    whenever the Config carries nothing but ``model``. It is not
    cosmetic: §3.2 shows the seven-variant document is unreadable
    without it — five near-identical seven-line ``decode`` blocks bury
    the two variants that actually differ, and readability is this
    artifact's purpose.

    The loader expands a string to ``Config(model=s)``, so there is one
    code path rather than two representations.
    """
    from lmdiff._config import Config as _C

    d = cfg.to_dict()
    trimmed = {k: v for k, v in d.items() if v is not None}
    trimmed.pop("name", None)          # the mapping key already carries it
    if not trimmed.get("capabilities_required"):
        trimmed.pop("capabilities_required", None)   # empty frozenset

    default_decode = _C(model="_").to_dict()["decode"]
    is_plain = set(trimmed) <= {"model", "decode"} and (
        trimmed.get("decode", default_decode) == default_decode
    )
    return cfg.model if is_plain else trimmed


def _provenance(probes: Any) -> dict[str, Any]:
    """Six fields, plus ``lm_eval`` when the probe spec needs it.

    The criterion (§4) is: record what cannot be recovered from the
    GeoResult, and nothing else. ``probes_excluded`` and
    ``domain_status_summary`` are deliberately absent — the GeoResult
    holds both exactly, and duplication is the failure this project has
    hit three times (L-035).

    ``lm_eval`` is conditional because an ``lm_eval:`` identifier
    resolves against an *optional* dependency whose version determines
    probe text, splits and ordering. Without it the artifact would pin
    the metric set, the seed and every decode parameter while leaving
    the probes themselves unpinned (§3.3).
    """
    import lmdiff
    from lmdiff.report.json_report import SCHEMA_VERSION

    out: dict[str, Any] = {
        "lmdiff": lmdiff.__version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}."
                  f"{sys.version_info.micro}",
        "geo_schema": SCHEMA_VERSION,
    }
    # str() every version, because they are not all `str`. torch reports
    # a `torch.torch_version.TorchVersion`, a str *subclass*, and
    # `yaml.safe_dump` represents exact built-in types only — it raises
    # RepresenterError on subclasses rather than falling back. Found by
    # emitting the real 7-variant calibration rather than a synthetic
    # one; a hand-written example would not have carried a real torch.
    for name, mod in (("torch", "torch"), ("transformers", "transformers")):
        try:
            out[name] = str(__import__(mod).__version__)
        except Exception:  # pragma: no cover - dependency absent
            out[name] = None

    if isinstance(probes, str) and probes.startswith("lm_eval:"):
        try:
            import lm_eval  # noqa: F401
            out["lm_eval"] = str(getattr(lm_eval, "__version__", "unknown"))
        except Exception:  # pragma: no cover
            out["lm_eval"] = None
    return out


def build_run_config(
    *,
    base: "Config",
    variants: dict[str, "Config"],
    probes: Any,
    n_probes: Optional[int],
    metrics: list[str],
    max_new_tokens: int,
    task_overrides: Optional[dict],
    seed: Optional[int],
    min_valid_fraction: float,
    run_id: str,
) -> dict[str, Any]:
    """Assemble the run-config mapping, defaults expanded.

    ``metrics`` must already be the **resolved** list, not the raw
    argument: ``"default"`` names something that moves, and AA.3's test
    is whether a reader can determine from the artifact what was used
    (§3.3). Callers pass the output of ``_resolve_metrics``.
    """
    doc: dict[str, Any] = {
        "lmdiff_schema": RUNCONFIG_SCHEMA_VERSION,
        "base": _variant_block(base),
        "variants": {name: _variant_block(c) for name, c in variants.items()},
        "probes": probes,
        "n_probes": n_probes,
        "metrics": list(metrics),
        "max_new_tokens": max_new_tokens,
        "task_overrides": task_overrides or {},
        "seed": seed,
        "min_valid_fraction": min_valid_fraction,
        # AA.5 — ships now with no current occupant. §1.1 confirmed
        # nothing on Config is unserializable today, which makes these
        # exactly the fields an implementer drops as "not needed yet".
        # The criterion for setting reproducible: false is that NO
        # POSSIBLE FILE could express the value (§5) — not that a
        # payload is awkward, and not that a __ref__ target is missing.
        "reproducible": True,
        PROVENANCE_KEY: _provenance(probes),
    }
    return doc


# ── Emission ─────────────────────────────────────────────────────────
#
# `yaml.safe_dump` cannot emit comments, and §3.2 identified two places
# the document is ambiguous without them: `decode.seed: null` sitting
# beside a top-level `seed: 42` reads as a contradiction to anyone
# copying the file, and `task_overrides` duplicates `max_new_tokens` at
# two levels without showing which wins.
#
# So emission is templated — dump, then annotate — and that carries a
# constraint worth stating plainly: **the annotated output must still be
# valid YAML that parses back to the same mapping.** A comment-bearing
# artifact that no longer round-trips would break AA.2's "the emitted
# YAML is runnable" decision, which is the whole premise of one schema
# rather than two.
#
# Hence step 4. It is an assertion in the emitter rather than only a
# test, because the failure mode is silent: an artifact that looks right
# and does not load. A test runs in CI; the emitter runs every time.

#: Comment injected beside `decode.seed: null` inside a variant block.
_SEED_INHERIT_NOTE = "inherits the family seed"

#: Comment injected on the `task_overrides` key.
_TASK_OVERRIDE_NOTE = (
    "per-task values below override the top-level max_new_tokens"
)


class RunConfigEmitError(RuntimeError):
    """The annotated YAML no longer parses back to what was dumped.

    Raised by the emitter's own round-trip check. Means the annotation
    step corrupted the document — never a caller error.
    """


def _annotate(text: str, doc: dict) -> str:
    """Inject the two clarifying comments §3.2 calls for.

    Only ever appends ``  # ...`` to a line, which YAML ignores, so the
    parsed mapping is unchanged by construction. Step 4 verifies that
    rather than trusting it.
    """
    family_seed = doc.get("seed")
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "seed: null" and line.startswith(" " * 6):
            # Nested under a variant's `decode:` — the family seed is
            # what actually applies (L-031 precedence).
            note = (
                f"{_SEED_INHERIT_NOTE}: {family_seed}"
                if family_seed is not None
                else f"{_SEED_INHERIT_NOTE} (unpinned)"
            )
            line = f"{line}  # {note}"
        elif stripped == "task_overrides:" and not line.startswith(" "):
            line = f"{line}  # {_TASK_OVERRIDE_NOTE}"
        out.append(line)
    # splitlines() drops the trailing newline safe_dump emits; restore it
    # so the artifact ends the way a YAML file should.
    return "\n".join(out) + "\n"


def emit_run_config_yaml(doc: dict) -> str:
    """Render a run-config mapping to annotated YAML.

    Four steps (§3.4): build, dump, annotate, and re-parse to confirm the
    annotation changed nothing but comments.

    Raises
    ------
    RunConfigEmitError
        If the annotated text does not parse back to ``doc``.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyYAML is required to emit a run configuration. "
            "Install with: pip install pyyaml"
        ) from exc

    dumped = yaml.safe_dump(
        doc, sort_keys=False, allow_unicode=True, width=88, default_flow_style=False,
    )
    annotated = _annotate(dumped, doc)

    # Step 4 — the self-check. Comments are safe by construction, but
    # "by construction" is exactly the kind of claim that stops being
    # true after someone edits _annotate.
    reparsed = yaml.safe_load(annotated)
    if reparsed != doc:
        raise RunConfigEmitError(
            "run-config annotation changed the parsed document; the "
            "emitted YAML would not load back to the configuration it "
            "was built from. This is an lmdiff bug — please report it."
        )
    return annotated


__all__ = [
    "PROVENANCE_KEY",
    "REF_KEY",
    "RUNCONFIG_SCHEMA_VERSION",
    "RunConfigEmitError",
    "build_run_config",
    "emit_run_config_yaml",
]
