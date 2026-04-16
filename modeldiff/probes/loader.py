from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Probe:
    id: str
    text: str
    domain: str | None = None
    expected: str | None = None
    metadata: dict = field(default_factory=dict)


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

    def filter(
        self,
        domain: str | None = None,
        ids: Iterable[str] | None = None,
    ) -> ProbeSet:
        result = list(self._probes)
        if domain is not None:
            result = [p for p in result if p.domain == domain]
        if ids is not None:
            id_set = set(ids)
            result = [p for p in result if p.id in id_set]
        return ProbeSet(result, name=self._name, version=self._version)

    def by_domain(self) -> dict[str, ProbeSet]:
        groups: dict[str, list[Probe]] = {}
        for p in self._probes:
            d = p.domain or "unknown"
            groups.setdefault(d, []).append(p)
        return {
            d: ProbeSet(probes, name=self._name, version=self._version)
            for d, probes in groups.items()
        }

    @classmethod
    def from_json(cls, path: str | Path) -> ProbeSet:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        probes = [
            Probe(
                id=p["id"],
                text=p["text"],
                domain=p.get("domain"),
                expected=p.get("expected"),
                metadata=p.get("metadata", {}),
            )
            for p in data["probes"]
        ]
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
