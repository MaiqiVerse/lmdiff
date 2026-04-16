import json
from pathlib import Path

import pytest

from lmdiff.probes.loader import Probe, ProbeSet

V01_PATH = Path(__file__).parent.parent / "lmdiff" / "probes" / "v01.json"


class TestProbeImmutable:
    def test_frozen(self):
        p = Probe(id="a", text="hello")
        with pytest.raises(AttributeError):
            p.text = "x"


class TestProbeSetBasics:
    def test_len(self):
        ps = ProbeSet([Probe(id="1", text="a"), Probe(id="2", text="b")])
        assert len(ps) == 2

    def test_iter(self):
        probes = [Probe(id="1", text="a"), Probe(id="2", text="b")]
        ps = ProbeSet(probes)
        assert list(ps) == probes

    def test_index_returns_probe(self):
        p = Probe(id="1", text="a")
        ps = ProbeSet([p])
        assert ps[0] is p

    def test_slice_returns_probeset(self):
        probes = [Probe(id=str(i), text=str(i)) for i in range(5)]
        ps = ProbeSet(probes, name="test")
        sliced = ps[1:3]
        assert isinstance(sliced, ProbeSet)
        assert len(sliced) == 2
        assert sliced.name == "test"


class TestProbeSetImmutability:
    def test_setattr_raises(self):
        ps = ProbeSet([Probe(id="1", text="a")])
        with pytest.raises(AttributeError, match="immutable"):
            ps.probes = []

    def test_internal_is_tuple(self):
        ps = ProbeSet([Probe(id="1", text="a")])
        assert isinstance(ps._probes, tuple)


class TestFromList:
    def test_auto_ids(self):
        ps = ProbeSet.from_list(["hello", "world"])
        assert ps.ids == ["default_000", "default_001"]
        assert ps.texts == ["hello", "world"]

    def test_custom_domain(self):
        ps = ProbeSet.from_list(["q"], domain="math")
        assert ps[0].domain == "math"
        assert ps[0].id == "math_000"


class TestFromJson:
    def test_load_v01(self):
        ps = ProbeSet.from_json(V01_PATH)
        assert len(ps) == 90
        assert ps.name == "v01"
        assert ps.version == "0.2.1"
        assert set(ps.domains) == {"math", "knowledge", "code"}

    def test_v01_domain_counts(self):
        ps = ProbeSet.from_json(V01_PATH)
        by_d = ps.by_domain()
        assert len(by_d["math"]) == 30
        assert len(by_d["knowledge"]) == 30
        assert len(by_d["code"]) == 30


class TestToJsonRoundTrip:
    def test_round_trip(self, tmp_path):
        original = ProbeSet.from_json(V01_PATH)
        out = tmp_path / "out.json"
        original.to_json(out)
        reloaded = ProbeSet.from_json(out)

        assert len(reloaded) == len(original)
        assert reloaded.name == original.name
        assert reloaded.version == original.version
        for a, b in zip(original, reloaded):
            assert a.id == b.id
            assert a.text == b.text
            assert a.domain == b.domain
            assert a.expected == b.expected


class TestFilter:
    def test_filter_domain(self):
        ps = ProbeSet.from_json(V01_PATH)
        math = ps.filter(domain="math")
        assert len(math) == 30
        assert all(p.domain == "math" for p in math)
        assert len(ps) == 90

    def test_filter_ids(self):
        ps = ProbeSet.from_json(V01_PATH)
        subset = ps.filter(ids=["math_001", "code_001"])
        assert len(subset) == 2
        assert {p.id for p in subset} == {"math_001", "code_001"}

    def test_filter_combined(self):
        ps = ProbeSet.from_json(V01_PATH)
        result = ps.filter(domain="math", ids=["math_001", "math_002", "code_001"])
        assert len(result) == 2
        assert all(p.domain == "math" for p in result)


class TestByDomain:
    def test_returns_dict_of_probesets(self):
        ps = ProbeSet.from_json(V01_PATH)
        by_d = ps.by_domain()
        assert isinstance(by_d, dict)
        for v in by_d.values():
            assert isinstance(v, ProbeSet)


class TestProperties:
    def test_texts(self):
        ps = ProbeSet([Probe(id="1", text="hello"), Probe(id="2", text="world")])
        assert ps.texts == ["hello", "world"]

    def test_ids(self):
        ps = ProbeSet([Probe(id="a", text="x"), Probe(id="b", text="y")])
        assert ps.ids == ["a", "b"]

    def test_domains_sorted_unique(self):
        ps = ProbeSet([
            Probe(id="1", text="x", domain="b"),
            Probe(id="2", text="y", domain="a"),
            Probe(id="3", text="z", domain="b"),
        ])
        assert ps.domains == ["a", "b"]


class TestRepr:
    def test_repr_contents(self):
        ps = ProbeSet.from_json(V01_PATH)
        r = repr(ps)
        assert "v01" in r
        assert "n=90" in r
        assert "code" in r
        assert "math" in r
        assert "knowledge" in r
