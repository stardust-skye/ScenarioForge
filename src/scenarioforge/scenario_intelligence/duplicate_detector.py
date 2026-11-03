"""Metadata-only duplicate and redundancy detection.

Three distinct notions are kept apart:
  * Exact metadata duplicates: identical deterministic fingerprints (same selected metadata).
  * Near-duplicate metadata candidates: weighted similarity >= a configurable threshold.
  * Visually identical samples: OUT OF SCOPE. No pixels or point clouds are compared.

Similarity is a weighted mean over feature groups:
  * ``categories``: Jaccard similarity of category sets
  * ``category_counts``: cosine similarity of category count vectors
  * ``numeric``: cosine similarity of min-max normalised numeric features (valid only when both
    scenarios have every numeric feature; otherwise the group is skipped for that pair and the
    remaining weights are renormalised)
  * ``context``: 1 if both scenarios share the same location, else 0 (skipped if either unknown)

Limitations: consecutive keyframes in one scene are often similar by construction, so high
similarity is expected there and does not mean redundant information. Thresholds are heuristic;
cosine similarity on non-negative normalised features is insensitive to magnitude. Comparison is
O(n^2) in the number of scenarios.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import numpy as np

from scenarioforge.scenario_intelligence.schemas import ScenarioRecord

DEFAULT_FINGERPRINT_FEATURES = ("category_counts", "attribute_counts", "visibility_counts")
DEFAULT_NUMERIC_FEATURES = (
    "n_objects",
    "n_unique_instances",
    "ego_speed_mps",
    "n_objects_near_ego",
)


@dataclass(frozen=True)
class DuplicateConfig:
    fingerprint_features: tuple[str, ...] = DEFAULT_FINGERPRINT_FEATURES
    numeric_features: tuple[str, ...] = DEFAULT_NUMERIC_FEATURES
    weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "categories": 1.0,
            "category_counts": 1.0,
            "numeric": 1.0,
            "context": 0.5,
        }
    )
    similarity_threshold: float = 0.9
    same_scene_only: bool = False
    scene_scope: str = "all"
    min_temporal_separation_s: float = 0.0
    temporal_adjacency_s: float = 1.0
    exclude_temporally_adjacent: bool = False
    histogram_bins: int = 20

    def __post_init__(self) -> None:
        if not 0 < self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if (
            set(self.weights) - {"categories", "category_counts", "numeric", "context"}
            or any(not isfinite(w) or w < 0 for w in self.weights.values())
            or not any(self.weights.values())
        ):
            raise ValueError("similarity weights must be non-negative with one positive")
        if self.histogram_bins < 1:
            raise ValueError("histogram_bins must be at least 1")
        if self.scene_scope not in {"all", "same", "cross"}:
            raise ValueError("scene_scope must be 'all', 'same', or 'cross'")
        if self.same_scene_only and self.scene_scope == "cross":
            raise ValueError("same_scene_only conflicts with scene_scope='cross'")
        if not isfinite(self.min_temporal_separation_s) or self.min_temporal_separation_s < 0:
            raise ValueError("min_temporal_separation_s must be finite and non-negative")
        if not isfinite(self.temporal_adjacency_s) or self.temporal_adjacency_s < 0:
            raise ValueError("temporal_adjacency_s must be finite and non-negative")


def fingerprint(record: ScenarioRecord, features: Sequence[str]) -> str:
    """Deterministic SHA-256 over canonical JSON of the selected features (ids excluded)."""
    payload = {name: record.features.get(name) for name in features}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def exact_duplicate_groups(
    records: Sequence[ScenarioRecord], features: Sequence[str] = DEFAULT_FINGERPRINT_FEATURES
) -> list[list[str]]:
    """Groups (size >= 2) of sample ids sharing an identical metadata fingerprint."""
    buckets: dict[str, list[str]] = {}
    for record in records:
        buckets.setdefault(fingerprint(record, features), []).append(record.sample_id)
    return sorted(
        (sorted(ids) for ids in buckets.values() if len(ids) > 1), key=lambda g: (-len(g), g[0])
    )


@dataclass
class SimilarityIndex:
    """Pairwise similarity matrix aligned with ``sample_ids``."""

    sample_ids: list[str]
    matrix: np.ndarray
    component_similarities: dict[str, np.ndarray] = field(default_factory=dict)
    component_validity: dict[str, np.ndarray] = field(default_factory=dict)

    def similarity(self, a: str, b: str) -> float:
        index = {s: i for i, s in enumerate(self.sample_ids)}
        return float(self.matrix[index[a], index[b]])


def compute_similarity(
    records: Sequence[ScenarioRecord], config: DuplicateConfig | None = None
) -> SimilarityIndex:
    """Weighted metadata similarity for all scenario pairs (diagonal = 1)."""
    config = config or DuplicateConfig()
    n = len(records)
    ids = [r.sample_id for r in records]
    numerator = np.zeros((n, n))
    denominator = np.zeros((n, n))
    ones = np.ones((n, n), dtype=bool)
    component_similarities: dict[str, np.ndarray] = {}
    component_validity: dict[str, np.ndarray] = {}

    def add_component(name: str, scores: np.ndarray, valid: np.ndarray) -> None:
        component_similarities[name] = scores
        component_validity[name] = valid
        _accumulate(numerator, denominator, scores, valid, config.weights.get(name, 0.0))

    # categories: Jaccard on presence sets
    categories = sorted({c for r in records for c in (r.features.get("category_counts") or {})})
    column = {c: i for i, c in enumerate(categories)}
    presence = np.zeros((n, len(categories)))
    counts = np.zeros((n, len(categories)))
    for row, r in enumerate(records):
        for c, v in (r.features.get("category_counts") or {}).items():
            presence[row, column[c]] = 1.0
            counts[row, column[c]] = v
    intersection = presence @ presence.T
    sizes = presence.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - intersection
    jaccard = np.where(union > 0, intersection / np.where(union > 0, union, 1), 1.0)
    add_component("categories", jaccard, ones)
    add_component("category_counts", _cosine(counts), ones)

    # numeric: min-max normalised, valid only where both rows are complete
    if config.numeric_features:
        values = np.array(
            [[_num(r.features.get(f)) for f in config.numeric_features] for r in records], dtype=float
        )
        complete = ~np.isnan(values).any(axis=1)
        if complete.any():
            low = values[complete].min(axis=0)
            span = values[complete].max(axis=0) - low
            scaled = np.where(span > 0, (np.nan_to_num(values) - low) / np.where(span > 0, span, 1), 0.0)
            valid = complete[:, None] & complete[None, :]
            add_component("numeric", _cosine(scaled), valid)
        else:
            add_component("numeric", np.zeros((n, n)), np.zeros((n, n), dtype=bool))

    # context: same location
    locations = [r.features.get("location") for r in records]
    known = np.array([loc is not None for loc in locations])
    same = np.array([[a == b for b in locations] for a in locations])
    add_component("context", same.astype(float), known[:, None] & known[None, :])

    matrix = np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1), 0.0)
    np.fill_diagonal(matrix, 1.0)
    return SimilarityIndex(ids, matrix, component_similarities, component_validity)


def _accumulate(
    numerator: np.ndarray, denominator: np.ndarray, similarity: np.ndarray, valid: np.ndarray, weight: float
) -> None:
    if weight <= 0:
        return
    numerator += weight * np.where(valid, similarity, 0.0)
    denominator += weight * valid


def _cosine(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    product = (matrix @ matrix.T) / (safe[:, None] * safe[None, :])
    zero = norms == 0
    both_zero = zero[:, None] & zero[None, :]
    one_zero = zero[:, None] ^ zero[None, :]
    product = np.where(both_zero, 1.0, np.where(one_zero, 0.0, product))
    return np.clip(product, 0.0, 1.0)


def _num(value: Any) -> float:
    return float("nan") if value is None or isinstance(value, str) else float(value)


def detect_duplicates(
    records: Sequence[ScenarioRecord], config: DuplicateConfig | None = None
) -> dict[str, Any]:
    """Exact groups, near-duplicate candidate pairs, and the similarity distribution."""
    config = config or DuplicateConfig()
    if len(records) < 2:
        raise ValueError("Duplicate detection needs at least two scenarios")
    exact = exact_duplicate_groups(records, config.fingerprint_features)
    exact_members = {sid: i for i, group in enumerate(exact) for sid in group}
    index = compute_similarity(records, config)
    records_by_id = {record.sample_id: record for record in records}
    scene_order: dict[str, int] = {}
    by_scene: dict[str, list[ScenarioRecord]] = {}
    for record in records:
        by_scene.setdefault(record.scene_id, []).append(record)
    for scene_records in by_scene.values():
        for position, record in enumerate(
            sorted(scene_records, key=lambda r: (r.timestamp, r.sample_id))
        ):
            scene_order[record.sample_id] = position

    upper = np.triu_indices(len(records), k=1)
    sims = index.matrix[upper]
    candidates: list[dict[str, Any]] = []
    for i, j, sim in zip(upper[0], upper[1], sims):
        a, b = index.sample_ids[i], index.sample_ids[j]
        is_exact = a in exact_members and exact_members.get(a) == exact_members.get(b)
        if not is_exact and sim >= config.similarity_threshold:
            first, second = records_by_id[a], records_by_id[b]
            same_scene = first.scene_id == second.scene_id
            time_separation_s = abs(first.timestamp - second.timestamp) / 1_000_000 if same_scene else None
            order_gap = abs(scene_order[a] - scene_order[b]) if same_scene else None
            if not same_scene:
                relationship = "cross_scene"
            elif order_gap == 1:
                relationship = "consecutive"
            elif time_separation_s is not None and time_separation_s <= config.temporal_adjacency_s:
                relationship = "temporally_adjacent"
            else:
                relationship = "non_adjacent_same_scene"
            candidates.append({
                "sample_a": a,
                "sample_b": b,
                "similarity": round(float(sim), 6),
                "similarity_components": _pair_similarity_components(
                    index, i, j, config.weights
                ),
                "same_scene": same_scene,
                "temporal_relationship": relationship,
                "scene_sample_order_gap": order_gap,
                "time_separation_s": round(time_separation_s, 6) if time_separation_s is not None else None,
            })

    scene_scope = "same" if config.same_scene_only and config.scene_scope == "all" else config.scene_scope
    pairs = [
        pair for pair in candidates
        if (scene_scope == "all"
            or (scene_scope == "same" and pair["same_scene"])
            or (scene_scope == "cross" and not pair["same_scene"]))
        and not (
            pair["same_scene"]
            and pair["time_separation_s"] < config.min_temporal_separation_s
        )
        and not (
            config.exclude_temporally_adjacent
            and pair["temporal_relationship"] in {"consecutive", "temporally_adjacent"}
        )
    ]
    pairs.sort(key=lambda p: (-p["similarity"], p["sample_a"], p["sample_b"]))
    classifications = ("consecutive", "temporally_adjacent", "non_adjacent_same_scene", "cross_scene")
    candidate_counts = {kind: sum(p["temporal_relationship"] == kind for p in candidates) for kind in classifications}
    filtered_counts = {kind: sum(p["temporal_relationship"] == kind for p in pairs) for kind in classifications}

    histogram, edges = np.histogram(sims, bins=config.histogram_bins, range=(0.0, 1.0))
    return {
        "n_scenarios": len(records),
        "fingerprint_features": list(config.fingerprint_features),
        "exact_duplicate_groups": exact,
        "n_exact_duplicate_samples": sum(len(g) for g in exact),
        "similarity_threshold": config.similarity_threshold,
        "scene_scope": scene_scope,
        "min_temporal_separation_s": config.min_temporal_separation_s,
        "temporal_adjacency_s": config.temporal_adjacency_s,
        "exclude_temporally_adjacent": config.exclude_temporally_adjacent,
        "similarity_weights": dict(config.weights),
        "similarity_components": {
            "categories": "Jaccard similarity of category-presence sets",
            "category_counts": "Cosine similarity of per-category object counts",
            "numeric": {
                "formula": "Cosine similarity after per-feature min-max scaling",
                "features": list(config.numeric_features),
                "availability": "included only when both scenarios have all configured values",
            },
            "context": "1 for the same known location, otherwise 0; omitted if either is unknown",
        },
        "n_near_duplicate_pairs_unfiltered": len(candidates),
        "pair_classification_counts_unfiltered": candidate_counts,
        "pair_classification_counts": filtered_counts,
        "n_near_duplicate_pairs": len(pairs),
        "n_near_duplicate_pairs_same_scene": sum(1 for p in pairs if p["same_scene"]),
        "n_near_duplicate_pairs_cross_scene": filtered_counts["cross_scene"],
        "n_near_duplicate_pairs_consecutive": filtered_counts["consecutive"],
        "n_near_duplicate_pairs_temporally_adjacent": filtered_counts["temporally_adjacent"],
        "n_near_duplicate_pairs_non_adjacent_same_scene": filtered_counts["non_adjacent_same_scene"],
        "near_duplicate_pairs": pairs,
        "similarity_histogram": {"counts": histogram.tolist(), "edges": [round(float(e), 6) for e in edges]},
        "similarity_summary": {
            "mean": float(sims.mean()), "median": float(np.median(sims)), "max": float(sims.max()),
        },
        "scope_note": (
            "Metadata similarity only; no image or point-cloud comparison. Consecutive and "
            "temporally adjacent keyframes are similarity observations, not evidence that either "
            "sample is redundant training data."
        ),
    }


def _pair_similarity_components(
    index: SimilarityIndex, row_a: int, row_b: int, weights: Mapping[str, float]
) -> dict[str, dict[str, float | None]]:
    available_weight = sum(
        weights.get(name, 0.0)
        for name, valid in index.component_validity.items()
        if valid[row_a, row_b] and weights.get(name, 0.0) > 0
    )
    details: dict[str, dict[str, float | None]] = {}
    for name, scores in index.component_similarities.items():
        weight = weights.get(name, 0.0)
        valid = index.component_validity[name][row_a, row_b]
        score = float(scores[row_a, row_b]) if valid else None
        contribution = (
            weight * score / available_weight
            if score is not None and weight > 0 and available_weight > 0
            else 0.0
        )
        details[name] = {
            "similarity": round(score, 6) if score is not None else None,
            "weight": weight,
            "weighted_contribution": round(contribution, 6),
        }
    return details
