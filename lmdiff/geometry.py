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


@dataclass(frozen=True)
class PCAResult:
    """PCA projection of change vectors onto principal axes.

    coords[variant] is a tuple of length n_components giving the variant's
    coordinates in PC-space. Base is implicitly at the origin (it is the
    zero vector in δ-space by construction).
    """
    coords: dict[str, tuple[float, ...]]
    explained_variance_ratio: tuple[float, ...]
    n_components: int
    n_variants: int


@dataclass(frozen=True)
class ComplementarityResult:
    """Per-variant-pair decomposition of where two modifications affect
    the same vs different domains.

    A domain counts as "affected" by a variant if that variant's magnitude
    on the domain exceeds `threshold * overall_magnitude`. Default 0.3.
    """
    v1: str
    v2: str
    cosine: float
    selective_cosine: float
    magnitude_ratio: float
    overlap_domains: tuple[str, ...]
    unique_v1_domains: tuple[str, ...]
    unique_v2_domains: tuple[str, ...]
    threshold: float


@dataclass(frozen=True)
class ClusterResult:
    """Hierarchical clustering of variants by change-vector similarity.

    Distance metric: 1 - cosine(δ_i, δ_j) (using GeoResult.cosine_matrix)
    or euclidean on raw change_vectors. Linkage methods: single, complete,
    average (default), ward. `linkage_matrix` is scipy's (n-1, 4) format
    as a nested list.
    """
    labels: tuple[str, ...]
    linkage_matrix: list[list[float]]
    method: str
    distance_metric: str
    n_variants: int


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

    probe_domains (schema v3) is a tuple aligned with change_vectors[v]
    (length n_probes), one entry per probe. Enables domain_heatmap(),
    complementarity(), and per-domain analysis. () when the caller passed
    a bare list[str] to ChangeGeometry or when the JSON is v1/v2.

    avg_tokens_per_probe + magnitudes_normalized (schema v4) provide a
    per-token-normalized view of magnitude that's comparable across probe
    sets with very different prompt lengths (e.g. ~30-token MCQ vs
    ~9000-token long-context). See LESSONS L-022.
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
    probe_domains: tuple[str | None, ...] = ()
    avg_tokens_per_probe: tuple[float, ...] = ()
    """Per-probe token count (using base_engine.tokenizer at analyze time).
    Aligned with change_vectors after the NaN filter; len == n_probes when
    populated. Empty tuple for legacy v1/v2/v3 GeoResult or when reconstructed
    from a JSON without the field. Schema v4 (L-022)."""

    magnitudes_normalized: dict[str, float] = field(default_factory=dict)
    """Per-token bulk-normalized ‖δ‖: raw / sqrt(n_probes × mean_tokens).

    Interprets the L2 norm as RMS per-token CE difference, comparable across
    probe sets with very different prompt lengths. Empty when
    avg_tokens_per_probe is empty (no token data available). Schema v4
    (L-022)."""

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

    # ── Phase 2 Commit A: analysis helpers ──────────────────────────

    def pca_map(self, n_components: int = 2) -> PCAResult:
        """Project variants into PC space via SVD on stacked change vectors.

        Does not center across variants — magnitude information in δ is
        meaningful (see L-017). Uses numpy SVD; sklearn is not a dependency.

        Raises:
            ValueError: when n_components exceeds n_variants or n_probes,
                        or when n_variants < 2 (PCA undefined).
        """
        n_variants = len(self.variant_names)
        n_probes = self.n_probes
        if n_variants < 2:
            raise ValueError(
                f"pca_map requires n_variants >= 2; got {n_variants}"
            )
        if n_components > n_variants:
            raise ValueError(
                f"n_components={n_components} > n_variants={n_variants}"
            )
        if n_components > n_probes:
            raise ValueError(
                f"n_components={n_components} > n_probes={n_probes}"
            )
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1; got {n_components}")

        # Stack vectors as rows: X shape (n_variants, n_probes)
        X = np.asarray(
            [self.change_vectors[name] for name in self.variant_names],
            dtype=float,
        )
        # SVD: X = U diag(S) Vt. PC coords are U * S (same as X @ V).
        U, S, _Vt = np.linalg.svd(X, full_matrices=False)
        coords_full = U * S  # shape (n_variants, min(n_variants, n_probes))
        coords_trunc = coords_full[:, :n_components]

        total_var = float((S ** 2).sum())
        if total_var <= 0:
            ratios = tuple(0.0 for _ in range(n_components))
        else:
            ratios = tuple(
                float((S[i] ** 2) / total_var) if i < len(S) else 0.0
                for i in range(n_components)
            )

        coords: dict[str, tuple[float, ...]] = {}
        for i, name in enumerate(self.variant_names):
            coords[name] = tuple(float(x) for x in coords_trunc[i])

        return PCAResult(
            coords=coords,
            explained_variance_ratio=ratios,
            n_components=n_components,
            n_variants=n_variants,
        )

    def domain_heatmap(self) -> dict[str, dict[str, float]]:
        """Per-variant per-domain magnitude of the change vector.

        Returns {variant: {domain: ‖δ_variant restricted to domain‖₂}}.
        Raises ValueError when probe_domains is empty. None domains are
        coalesced under the key "unknown".
        """
        if not self.probe_domains:
            raise ValueError(
                "domain_heatmap requires probe_domains; was this GeoResult "
                "built from a bare list[str] prompt list or a v1/v2 JSON? "
                "Use a ProbeSet with per-probe domain labels."
            )
        if len(self.probe_domains) != self.n_probes:
            raise ValueError(
                f"probe_domains length {len(self.probe_domains)} != n_probes {self.n_probes}"
            )

        # Group probe indices by resolved domain key.
        by_domain: dict[str, list[int]] = {}
        for idx, d in enumerate(self.probe_domains):
            key = d if d is not None else "unknown"
            by_domain.setdefault(key, []).append(idx)

        out: dict[str, dict[str, float]] = {}
        for name in self.variant_names:
            vec = np.asarray(self.change_vectors[name], dtype=float)
            per_domain: dict[str, float] = {}
            for domain, indices in by_domain.items():
                sub = vec[indices]
                per_domain[domain] = float(np.linalg.norm(sub))
            out[name] = per_domain
        return out

    def complementarity(
        self, v1: str, v2: str, threshold: float = 0.3,
    ) -> ComplementarityResult:
        """Overlap / unique-domain decomposition for variant pair (v1, v2).

        A domain is "affected" by a variant iff its per-domain magnitude is
        more than `threshold * overall_magnitude`. Uses domain_heatmap()
        (so probe_domains must be populated).
        """
        if v1 not in self.variant_names:
            raise ValueError(f"unknown variant: {v1!r}")
        if v2 not in self.variant_names:
            raise ValueError(f"unknown variant: {v2!r}")

        heatmap = self.domain_heatmap()
        mag_v1 = self.magnitudes.get(v1, 0.0)
        mag_v2 = self.magnitudes.get(v2, 0.0)

        def _affected(name: str, overall_mag: float) -> set[str]:
            if overall_mag <= 0:
                return set()
            per_domain = heatmap[name]
            return {
                d for d, m in per_domain.items()
                if m / overall_mag > threshold
            }

        affected_v1 = _affected(v1, mag_v1)
        affected_v2 = _affected(v2, mag_v2)
        overlap = affected_v1 & affected_v2
        unique_v1 = affected_v1 - affected_v2
        unique_v2 = affected_v2 - affected_v1

        if mag_v2 > 0:
            ratio = float(mag_v1 / mag_v2)
        else:
            ratio = float("inf") if mag_v1 > 0 else float("nan")

        cos = self.cosine_matrix.get(v1, {}).get(v2, float("nan"))
        sel_cos = float("nan")
        if self.selective_cosine_matrix:
            sel_cos = self.selective_cosine_matrix.get(v1, {}).get(v2, float("nan"))

        return ComplementarityResult(
            v1=v1,
            v2=v2,
            cosine=float(cos),
            selective_cosine=float(sel_cos),
            magnitude_ratio=ratio,
            overlap_domains=tuple(sorted(overlap)),
            unique_v1_domains=tuple(sorted(unique_v1)),
            unique_v2_domains=tuple(sorted(unique_v2)),
            threshold=float(threshold),
        )

    def cluster(
        self, method: str = "average", distance_metric: str = "cosine",
    ) -> ClusterResult:
        """Hierarchical clustering of variants via scipy.cluster.hierarchy.

        Distance:
          - "cosine": 1 - GeoResult.cosine_matrix[i][j] (NaN treated as 1.0)
          - "euclidean": pdist on raw change_vectors
        Method: one of single, complete, average, ward.

        Raises:
            ValueError: n_variants < 2, bad method, or bad distance_metric.
            ImportError: scipy not installed (pip install lmdiff-kit[viz]).
        """
        if method not in ("single", "complete", "average", "ward"):
            raise ValueError(
                f"method must be one of single/complete/average/ward; got {method!r}"
            )
        if distance_metric not in ("cosine", "euclidean"):
            raise ValueError(
                f"distance_metric must be 'cosine' or 'euclidean'; got {distance_metric!r}"
            )

        labels = tuple(self.variant_names)
        n = len(labels)
        if n < 2:
            raise ValueError(f"cluster requires n_variants >= 2; got {n}")

        try:
            from scipy.cluster.hierarchy import linkage as _linkage  # type: ignore[import-not-found]
            from scipy.spatial.distance import pdist as _pdist  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "scipy required for hierarchical clustering. "
                "Install with: pip install lmdiff-kit[viz]"
            ) from exc

        if distance_metric == "cosine":
            condensed: list[float] = []
            for i in range(n):
                for j in range(i + 1, n):
                    c = self.cosine_matrix[labels[i]][labels[j]]
                    if c != c:  # NaN
                        d = 1.0
                    else:
                        d = 1.0 - float(c)
                    # Clamp: numerical drift can push 1 - cos slightly
                    # negative when cos was clamped to +1.
                    condensed.append(max(0.0, d))
            Z = _linkage(np.asarray(condensed, dtype=float), method=method)
        else:  # euclidean
            X = np.asarray(
                [self.change_vectors[name] for name in labels],
                dtype=float,
            )
            # linkage(observation matrix) goes via pdist internally; pass
            # explicitly so metric is unambiguous.
            D = _pdist(X, metric="euclidean")
            Z = _linkage(D, method=method)

        return ClusterResult(
            labels=labels,
            linkage_matrix=[[float(x) for x in row] for row in Z.tolist()],
            method=method,
            distance_metric=distance_metric,
            n_variants=n,
        )


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

        # Pre-compute per-probe token counts using the base tokenizer. CPU
        # only — no model forward pass. Needed for magnitudes_normalized
        # (schema v4) so per-task magnitudes from tasks with very different
        # prompt lengths can be compared on RMS-per-token terms.
        all_probe_tokens: list[int] = [
            len(base_engine.tokenizer.encode(p, add_special_tokens=False))
            for p in self.prompts
        ]

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

        # probe_domains: aligned with change_vectors after NaN filter (v3).
        # Stays () when caller passed a bare list[str] instead of a ProbeSet.
        probe_domains: tuple[str | None, ...] = ()
        if self.probe_set is not None:
            all_domains = [p.domain for p in self.probe_set]
            probe_domains = tuple(all_domains[i] for i in valid_indices)

        # Schema v4 (L-022): per-probe token counts + bulk-normalized magnitudes.
        avg_tokens_per_probe: tuple[float, ...] = tuple(
            float(all_probe_tokens[i]) for i in valid_indices
        )
        magnitudes_normalized: dict[str, float] = {}
        if avg_tokens_per_probe:
            mean_tokens = float(np.mean(avg_tokens_per_probe))
            denom = math.sqrt(n_valid * mean_tokens) if mean_tokens > 0 else 0.0
            if denom > 0:
                magnitudes_normalized = {
                    name: magnitudes[name] / denom for name in variant_names
                }

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
            probe_domains=probe_domains,
            avg_tokens_per_probe=avg_tokens_per_probe,
            magnitudes_normalized=magnitudes_normalized,
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
