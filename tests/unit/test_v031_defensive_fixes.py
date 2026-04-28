"""Defensive fixes shipped in v0.3.1 (commit 1).

Two unrelated bugs surfaced after the v0.3.0 demo run:

1. ``result.save("nonexistent_dir/r.json")`` raised ``FileNotFoundError``
   because ``json_report.write_json`` and ``json_report.render`` did not
   ``mkdir(parents=True)`` before writing. Other channels (HTML / markdown /
   figures) already auto-created parents — only JSON was inconsistent.

2. ``InferenceEngine.device`` was hardcoded to ``"cuda"`` after load, even
   when ``device_map="auto"`` had sharded the embedding layer to CPU under
   memory pressure. Subsequent ``.score()`` placed input tensors on cuda
   while embedding weights lived on cpu → ``RuntimeError: Expected all
   tensors to be on the same device``. The bug only triggered after several
   sequential 7B loads in a single process, so unit tests verify only the
   anchor invariant (the original RuntimeError requires accelerate sharding
   that is hard to simulate in tests).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lmdiff.geometry import GeoResult, _compute_share_per_domain
from lmdiff.report.json_report import render as json_render, write_json


# ── Fixture helpers ──────────────────────────────────────────────────


def _tiny_geo() -> GeoResult:
    cv = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
    geo = GeoResult(
        base_name="b",
        variant_names=list(cv),
        n_probes=2,
        magnitudes={v: float(np.linalg.norm(cv[v])) for v in cv},
        cosine_matrix={v: {w: 1.0 if v == w else 0.5 for w in cv} for v in cv},
        change_vectors=cv,
        per_probe={v: {f"p{i}": cv[v][i] for i in range(2)} for v in cv},
        probe_domains=("d", "d"),
        avg_tokens_per_probe=(8.0, 8.0),
    )
    geo.share_per_domain = _compute_share_per_domain(geo)
    return geo


# ── 1a. Save autocreate ──────────────────────────────────────────────


class TestSaveAutocreatesParentDir:
    def test_write_json_creates_missing_parent(self, tmp_path):
        geo = _tiny_geo()
        deep = tmp_path / "level1" / "level2" / "level3" / "out.json"
        assert not deep.parent.exists()
        write_json(geo, deep)
        assert deep.exists()
        # Round-trip parses cleanly.
        json.loads(deep.read_text(encoding="utf-8"))

    def test_render_writes_to_missing_parent(self, tmp_path):
        geo = _tiny_geo()
        deep = tmp_path / "missing_dir" / "r.json"
        assert not deep.parent.exists()
        json_render(geo, path=str(deep))
        assert deep.exists()

    def test_result_save_creates_parent(self, tmp_path):
        geo = _tiny_geo()
        target = tmp_path / "runs" / "v030-demo" / "family_geometry.json"
        assert not target.parent.exists()
        # The end-to-end convenience path the user actually hit.
        geo.save(str(target))
        assert target.exists()
        d = json.loads(target.read_text(encoding="utf-8"))
        assert d["schema_version"] == "5"

    def test_existing_parent_dir_no_error(self, tmp_path):
        # exist_ok=True semantics — re-running into the same dir is fine.
        geo = _tiny_geo()
        target = tmp_path / "x.json"
        geo.save(str(target))
        geo.save(str(target))  # second time, parent exists — should not raise
        assert target.exists()


# ── 1b. InferenceEngine device anchor ────────────────────────────────


_TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


@pytest.mark.slow
class TestInferenceEngineDeviceAnchor:
    """``slow`` because constructing InferenceEngine downloads a tiny HF
    model and pulls torch+transformers. Not GPU-gated; runs on CPU."""

    def test_device_anchored_to_actual_embedding_location(self):
        from lmdiff.engine import InferenceEngine
        from lmdiff.config import Config as V02Config
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            cfg = V02Config(model=_TINY_MODEL)
        engine = InferenceEngine(cfg)
        embedding_device = engine._model.get_input_embeddings().weight.device
        assert engine.device == str(embedding_device), (
            f"engine.device={engine.device!r} but embedding actually on "
            f"{embedding_device!r}. Subsequent score() / generate() would "
            f"hit 'Expected all tensors to be on the same device'."
        )

    def test_score_runs_without_device_mismatch(self):
        from lmdiff.engine import InferenceEngine
        from lmdiff.config import Config as V02Config
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            cfg = V02Config(model=_TINY_MODEL)
        engine = InferenceEngine(cfg)
        # Smoke: invoke score() and confirm no RuntimeError. Score values
        # themselves are tested elsewhere; here we just need the call to
        # complete on the anchored device.
        result = engine.score(["hello"], continuations=[" world"])
        assert result.cross_entropies is not None
        assert len(result.cross_entropies) == 1
