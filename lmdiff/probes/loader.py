from __future__ import annotations

import json
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KNOWN_OUTPUT_TYPES: frozenset[str] = frozenset({
    "multiple_choice",
    "generate_until",
    "loglikelihood",
    "loglikelihood_rolling",
})
"""How a probe is queried. lm-eval's vocabulary, verbatim.

Copied rather than translated on purpose: these values come out of
lm-eval task configs, and giving them lmdiff-specific synonyms is how
one quantity ends up with two names (L-035). ``from_lm_eval`` validates
against this same set — it is the only definition.

Closed *with a warning*, not with an exception. An unrecognised value in
a probe file is worth telling the author about at load; it is not worth
refusing to run over, because nothing downstream branches on membership.
"""


@dataclass(frozen=True)
class Probe:
    id: str
    text: str
    domain: str | None = None
    output_type: str | None = None
    """How the model is queried for this probe (v0.4.4).

    One of ``KNOWN_OUTPUT_TYPES``, or ``None`` for unlabelled. ``None``
    is not a synonym for any value — it means nobody stated one, which
    is different from stating ``generate_until``."""
    scoring: str | None = None
    """Which evaluator judges this probe's output (v0.4.4).

    A key of ``lmdiff.tasks.registry.EVALUATOR_REGISTRY``, or ``None``
    to leave the choice to whoever runs the task. Open-ended: the
    registry is the vocabulary, so a name this version does not ship
    warns at load and falls back at run time rather than failing."""
    expected: str | None = None
    metadata: dict = field(default_factory=dict)


def _warn_unknown(values: dict[str, list[str]]) -> None:
    """One warning per unrecognised value, naming the probes."""
    for field_name, entries in values.items():
        for value, ids in sorted(entries.items()):
            shown = ", ".join(ids[:3]) + (f" (+{len(ids) - 3} more)" if len(ids) > 3 else "")
            warnings.warn(
                f"probe {field_name}={value!r} is not recognised "
                f"[{shown}]. Probes keep the value; anything dispatching "
                f"on it will fall back to its default.",
                UserWarning,
                stacklevel=3,
            )


class ProbeSet:
    """Immutable collection of Probe objects."""

    __slots__ = ("_probes", "_name", "_version")

    def __init__(
        self,
        probes: Iterable[Probe],
        name: str | None = None,
        version: str | None = None,
    ) -> None:
        object.__setattr__(self, "_probes", tuple(probes))
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_version", version)

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("ProbeSet is immutable")

    def __delattr__(self, key: str) -> None:
        raise AttributeError("ProbeSet is immutable")

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def version(self) -> str | None:
        return self._version

    def __len__(self) -> int:
        return len(self._probes)

    def __iter__(self):
        return iter(self._probes)

    def __getitem__(self, idx: int | slice) -> Probe | ProbeSet:
        if isinstance(idx, slice):
            return ProbeSet(self._probes[idx], name=self._name, version=self._version)
        return self._probes[idx]

    @property
    def texts(self) -> list[str]:
        return [p.text for p in self._probes]

    @property
    def ids(self) -> list[str]:
        return [p.id for p in self._probes]

    @property
    def domains(self) -> list[str]:
        return sorted({p.domain for p in self._probes if p.domain is not None})

    @property
    def output_types(self) -> list[str]:
        """Distinct ``output_type`` values present. ``None`` dropped,
        exactly as ``domains`` drops unassigned domains."""
        return sorted({
            p.output_type for p in self._probes if p.output_type is not None
        })

    @property
    def scorings(self) -> list[str]:
        """Distinct ``scoring`` values present. ``None`` dropped."""
        return sorted({p.scoring for p in self._probes if p.scoring is not None})

    def filter(
        self,
        domain: str | None = None,
        ids: Iterable[str] | None = None,
        output_type: str | None = None,
        scoring: str | None = None,
    ) -> ProbeSet:
        result = list(self._probes)
        if domain is not None:
            result = [p for p in result if p.domain == domain]
        if ids is not None:
            id_set = set(ids)
            result = [p for p in result if p.id in id_set]
        if output_type is not None:
            result = [p for p in result if p.output_type == output_type]
        if scoring is not None:
            result = [p for p in result if p.scoring == scoring]
        return ProbeSet(result, name=self._name, version=self._version)

    def by_domain(self) -> dict[str, ProbeSet]:
        return self._group_by(lambda p: p.domain)

    def by_output_type(self) -> dict[str, ProbeSet]:
        """Group by ``output_type``; unlabelled probes land in
        ``"unknown"``, matching ``by_domain``'s convention."""
        return self._group_by(lambda p: p.output_type)

    def _group_by(self, key) -> dict[str, ProbeSet]:
        groups: dict[str, list[Probe]] = {}
        for p in self._probes:
            groups.setdefault(key(p) or "unknown", []).append(p)
        return {
            k: ProbeSet(probes, name=self._name, version=self._version)
            for k, probes in groups.items()
        }

    @classmethod
    def from_json(cls, path: str | Path) -> ProbeSet:
        from lmdiff.tasks.registry import KNOWN_SCORINGS

        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        probes = [
            Probe(
                id=p["id"],
                text=p["text"],
                domain=p.get("domain"),
                output_type=p.get("output_type"),
                scoring=p.get("scoring"),
                expected=p.get("expected"),
                metadata=p.get("metadata", {}),
            )
            for p in data["probes"]
        ]

        unknown: dict[str, dict[str, list[str]]] = {"output_type": {}, "scoring": {}}
        for p in probes:
            if p.output_type is not None and p.output_type not in KNOWN_OUTPUT_TYPES:
                unknown["output_type"].setdefault(p.output_type, []).append(p.id)
            if p.scoring is not None and p.scoring not in KNOWN_SCORINGS:
                unknown["scoring"].setdefault(p.scoring, []).append(p.id)
        _warn_unknown(unknown)

        return cls(probes, name=data.get("name"), version=data.get("version"))

    @classmethod
    def from_list(cls, texts: list[str], domain: str = "default") -> ProbeSet:
        probes = [
            Probe(id=f"{domain}_{i:03d}", text=t, domain=domain)
            for i, t in enumerate(texts)
        ]
        return cls(probes)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "name": self._name,
            "version": self._version,
            "probes": [
                {
                    "id": p.id,
                    "text": p.text,
                    **({"domain": p.domain} if p.domain else {}),
                    **({"output_type": p.output_type} if p.output_type else {}),
                    **({"scoring": p.scoring} if p.scoring else {}),
                    **({"expected": p.expected} if p.expected else {}),
                    **({"metadata": p.metadata} if p.metadata else {}),
                }
                for p in self._probes
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        return (
            f"ProbeSet(name={self._name!r}, n={len(self._probes)}, "
            f"domains={self.domains})"
        )
