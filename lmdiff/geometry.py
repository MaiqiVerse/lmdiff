"""Change Geometry: treat behavioral shifts as vectors.

ChangeGeometry compares one base config against N variant configs on a
shared probe set. For each variant V and probe i it computes a scalar:

    δ_V[i] = CE(base scores V's output)[i] − CE(V scores V's own output)[i]

Stacking the scalars over probes gives a change vector δ_V ∈ R^N_probes.
Then:

- ‖δ_V‖₂  = how far V's behavior moved from base (magnitude).
- cos(δ_V, δ_W) = whether two variants moved in the same direction.

Zero-coupled with `metrics/*`: we compute CE directly via `engine.score`
instead of reusing `BehavioralDistance.compute`. Zero-coupled with torch:
CUDA memory is released via `engine.release_cuda_cache()` so this file
doesn't import torch.
"""
from __future__ import annotations

import gc
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lmdiff.config import Config
from lmdiff.engine import InferenceEngine, release_cuda_cache
from lmdiff.probes.loader import ProbeSet
from lmdiff.tokenizer_utils import bpb_from_ce, tokenizers_equivalent


@dataclass
class GeoResult:
    """Result of ChangeGeometry.analyze().

    - variant_names preserves the insertion order of the variants dict.
    - change_vectors[v] has the same length for every variant (equal to
      n_probes, which is the post-NaN-filter count).
    - cosine_matrix[v][v] == 1.0 when ‖δ_v‖ > 0, else NaN.
    - per_probe[v] is keyed by probe text, useful for point lookup. After a
      JSON round-trip, its keys come back alphabetically sorted because
      json_report.to_json emits with sort_keys=True — do NOT use
      list(per_probe[v].keys()) as a proxy for probe order; use
      change_vectors[v] (list, order preserved) or the original ProbeSet
      instead. See LESSONS L-018.

    Decomposition fields (δ = c·𝟙 + ε):
    - delta_means[v] = c (= mean of δ), same unit as magnitudes[v].
    - selective_magnitudes[v] = ‖δ − c·𝟙‖.
    - selective_cosine_matrix[v][w] = cos of centered δ vectors
      (= Pearson correlation). Self-entries 1.0, NaN when a variant
      has zero selective magnitude.

    All three decomposition fields default to empty dicts. A GeoResult
    read back from a v1 JSON will have them empty; fresh ones from
    analyze() or v2 JSON will be populated. See LESSONS L-017.
    """
    base_name: str
    variant_names: list[str]
    n_probes: int
    magnitudes: dict[str, float]
    cosine_matrix: dict[str, dict[str, float]]
    change_vectors: dict[str, list[float]]
    per_probe: dict[str, dict[str, float]]
    metadata: dict = field(default_factory=dict)
    delta_means: dict[str, float] = field(default_factory=dict)
    selective_magnitudes: dict[str, float] = field(default_factory=dict)
    selective_cosine_matrix: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary_table(self) -> list[dict]:
        """One row per variant: {variant, magnitude, cosines}."""
        rows: list[dict] = []
        for v in self.variant_names:
            rows.append({
                "variant": v,
                "magnitude": self.magnitudes[v],
                "cosines": dict(self.cosine_matrix[v]),
            })
        return rows

    @property
    def constant_fractions(self) -> dict[str, float]:
        """Per-variant ‖c·𝟙‖² / ‖δ‖², i.e. energy fraction of the uniform offset.

        Computed on demand from magnitudes and selective_magnitudes via
        Pythagoras: ‖δ‖² = ‖c·𝟙‖² + ‖ε‖², so ‖c·𝟙‖² = ‖δ‖² − ‖ε‖². Returns
        NaN when ‖δ‖ == 0 (can't divide). Returns empty dict when the
        decomposition fields are unpopulated (legacy v1 GeoResult).
        """
        if not self.delta_means or not self.selective_magnitudes:
            return {}
        result: dict[str, float] = {}
        for name in self.variant_names:
            mag = self.magnitudes.get(name, 0.0)
            if mag <= 0:
                result[name] = float("nan")
                continue
            sel_mag = self.selective_magnitudes.get(name, 0.0)
            const_energy = mag ** 2 - sel_mag ** 2
            result[name] = float(const_energy / (mag ** 2))
        return result


class ChangeGeometry:
    """One base config vs N named variants → change-vector geometry.

    The base engine is loaded lazily and kept alive for the duration of
    analyze(); variant engines are created inside the loop and released
    at the end of each iteration so peak VRAM is (1× base + 1× current
    variant) rather than (1× base + N× variants).
    """

    def __init__(
        self,
        base: Config,
        variants: dict[str, Config],
        prompts: list[str] | ProbeSet,
        n_samples: int = 1,
    ) -> None:
        if not variants:
            raise ValueError("variants must contain at least one entry")

        self.base_config = base
        self.variants = dict(variants)

        if isinstance(prompts, ProbeSet):
            self.probe_set: ProbeSet | None = prompts
            self.prompts: list[str] = list(prompts.texts)
        else:
            self.probe_set = None
            self.prompts = list(prompts)

        # Reserved for a future multi-sample averaging path; analyze()
        # currently always calls generate(n_samples=1) to stay aligned
        # with BehavioralDistance, which hard-codes n_samples=1.
        self.n_samples = n_samples

        self._base_engine: InferenceEngine | None = None

    @property
    def base_engine(self) -> InferenceEngine:
        if self._base_engine is None:
            self._base_engine = InferenceEngine(self.base_config)
        return self._base_engine

    def analyze(self, max_new_tokens: int = 16) -> GeoResult:
        n_total = len(self.prompts)
        if n_total == 0:
            raise ValueError("cannot analyze on an empty probe set")

        variant_names = list(self.variants.keys())
        base_engine = self.base_engine

        raw_deltas: dict[str, list[float]] = {}
        bpb_flags: dict[str, bool] = {}

        for name in variant_names:
            v_config = self.variants[name]
            v_engine = InferenceEngine(v_config)
            try:
                raw_deltas[name], bpb_flags[name] = self._delta_for_variant(
                    base_engine=base_engine,
                    v_engine=v_engine,
                    v_config=v_config,
                    max_new_tokens=max_new_tokens,
                )
            finally:
                # Order matters: drop Python refs first so empty_cache has
                # something to actually reclaim. Doing empty_cache before
                # gc.collect() leaves CUDA blocks still owned by live
                # tensors and effectively does nothing.
                del v_engine
                gc.collect()
                release_cuda_cache()

        valid_indices = _universally_valid_indices(raw_deltas, n_total)
        n_valid = len(valid_indices)

        change_vectors: dict[str, list[float]] = {
            name: [raw_deltas[name][i] for i in valid_indices]
            for name in variant_names
        }
        per_probe: dict[str, dict[str, float]] = {}
        for name in variant_names:
            per_probe[name] = {
                self.prompts[i]: raw_deltas[name][i] for i in valid_indices
            }

        magnitudes: dict[str, float] = {
            name: float(np.linalg.norm(change_vectors[name])) if n_valid > 0 else 0.0
            for name in variant_names
        }
        cosine_matrix = _cosine_matrix(variant_names, change_vectors, magnitudes)

        # δ = c·𝟙 + ε decomposition (L-017). Operates on the already-filtered
        # change_vectors so everything stays on the same probe basis.
        delta_means, selective_magnitudes, selective_cosine_matrix = (
            _selective_decomposition(variant_names, change_vectors)
        )

        metadata = {
            "n_total_probes": n_total,
            "n_skipped": n_total - n_valid,
            "bpb_normalized": bpb_flags,
            "max_new_tokens": max_new_tokens,
        }
        if self.probe_set is not None:
            if self.probe_set.name:
                metadata["probe_set_name"] = self.probe_set.name
            if self.probe_set.version:
                metadata["probe_set_version"] = self.probe_set.version

        return GeoResult(
            base_name=self.base_config.display_name,
            variant_names=variant_names,
            n_probes=n_valid,
            magnitudes=magnitudes,
            cosine_matrix=cosine_matrix,
            change_vectors=change_vectors,
            per_probe=per_probe,
            metadata=metadata,
            delta_means=delta_means,
            selective_magnitudes=selective_magnitudes,
            selective_cosine_matrix=selective_cosine_matrix,
        )

    def _delta_for_variant(
        self,
        base_engine: InferenceEngine,
        v_engine: InferenceEngine,
        v_config: Config,
        max_new_tokens: int,
    ) -> tuple[list[float], bool]:
        """Compute the raw δ vector (possibly containing NaN) for one variant."""
        gen_v = v_engine.generate(
            self.prompts, n_samples=1, max_new_tokens=max_new_tokens,
        )
        v_outputs = [comps[0] for comps in gen_v.completions]
        v_ids = [tids[0] for tids in gen_v.token_ids]

        score_b_of_v = base_engine.score(self.prompts, continuations=v_outputs)
        score_v_self = v_engine.score(self.prompts, continuation_ids=v_ids)

        same_tok = v_config.shares_tokenizer_with(self.base_config)
        if same_tok is None:
            same_tok = tokenizers_equivalent(
                base_engine.tokenizer, v_engine.tokenizer,
            )
        use_bpb = not same_tok

        deltas: list[float] = []
        for i in range(len(self.prompts)):
            ce_bv = score_b_of_v.cross_entropies[i]
            ce_vv = score_v_self.cross_entropies[i]

            if math.isnan(ce_bv) or math.isnan(ce_vv):
                deltas.append(float("nan"))
                continue

            if use_bpb:
                ce_bv = bpb_from_ce(
                    ce_bv,
                    n_tokens=len(score_b_of_v.token_ids[i]),
                    text=v_outputs[i],
                )
                ce_vv = bpb_from_ce(
                    ce_vv,
                    n_tokens=len(score_v_self.token_ids[i]),
                    text=v_outputs[i],
                )

            deltas.append(float(ce_bv - ce_vv))

        return deltas, use_bpb


def _universally_valid_indices(
    raw_deltas: dict[str, list[float]], n_total: int,
) -> list[int]:
    """Indices where every variant produced a finite δ.

    Global filter (not per-variant) so all variants end up with vectors
    of equal length and aligned basis — a prerequisite for computing
    cos(δ_A, δ_B).
    """
    valid: list[int] = []
    for i in range(n_total):
        if all(not math.isnan(raw_deltas[name][i]) for name in raw_deltas):
            valid.append(i)
    return valid


def _cosine_matrix(
    variant_names: list[str],
    change_vectors: dict[str, list[float]],
    magnitudes: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Build the full pairwise cosine matrix, handling zero-vector edges."""
    vectors = {name: np.asarray(change_vectors[name], dtype=float) for name in variant_names}
    matrix: dict[str, dict[str, float]] = {a: {} for a in variant_names}

    # Compute each unordered pair once, write both cells, so the result
    # is byte-exactly symmetric (not just symmetric up to float drift).
    for i, a in enumerate(variant_names):
        matrix[a][a] = 1.0 if magnitudes[a] > 0 else float("nan")
        for b in variant_names[i + 1:]:
            if magnitudes[a] == 0 or magnitudes[b] == 0:
                cos = float("nan")
            else:
                dot = float(np.dot(vectors[a], vectors[b]))
                cos = dot / (magnitudes[a] * magnitudes[b])
                cos = max(-1.0, min(1.0, cos))
            matrix[a][b] = cos
            matrix[b][a] = cos
    return matrix


def _selective_decomposition(
    variant_names: list[str],
    change_vectors: dict[str, list[float]],
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
    """Compute δ = c·𝟙 + ε per variant and the centered-cosine (Pearson) matrix.

    Returns (delta_means, selective_magnitudes, selective_cosine_matrix).
    Handles the n_valid == 0 case by returning 0.0 for means / magnitudes and
    NaN for every cell of the cosine matrix.
    """
    delta_means: dict[str, float] = {}
    selective_magnitudes: dict[str, float] = {}
    selective_vecs: dict[str, np.ndarray] = {}

    for name in variant_names:
        vec = np.asarray(change_vectors[name], dtype=float)
        if vec.size == 0:
            delta_means[name] = 0.0
            selective_magnitudes[name] = 0.0
            selective_vecs[name] = vec
            continue
        c = float(vec.mean())
        sel = vec - c
        delta_means[name] = c
        selective_magnitudes[name] = float(np.linalg.norm(sel))
        selective_vecs[name] = sel

    matrix: dict[str, dict[str, float]] = {a: {} for a in variant_names}
    for i, a in enumerate(variant_names):
        sa = selective_magnitudes[a]
        matrix[a][a] = 1.0 if sa > 0 else float("nan")
        for b in variant_names[i + 1:]:
            sb = selective_magnitudes[b]
            if sa == 0 or sb == 0:
                cos = float("nan")
            else:
                dot = float(np.dot(selective_vecs[a], selective_vecs[b]))
                cos = dot / (sa * sb)
                cos = max(-1.0, min(1.0, cos))
            matrix[a][b] = cos
            matrix[b][a] = cos
    return delta_means, selective_magnitudes, matrix
