"""Compare an original scenario set with a curated subset.

Reading the numbers (no claim about model performance is made here):
  * JSD (category distribution): lower = the subset's object-category mix is closer to the
    original; higher = it deliberately departs from it (expected for rarity-seeking strategies).
  * normalized entropy: higher = more even category use. Rebalancing strategies should raise it.
  * category coverage: share of original categories still present in the subset.
  * duplicate candidate rate: near-duplicate pairs per possible pair; lower = less redundancy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from scenarioforge.scenario_intelligence.binning import BinningConfig, FeatureDiscretizer
from scenarioforge.scenario_intelligence.diversity_analyzer import (
    DiversityConfig,
    object_category_counts,
    scenario_presence_counts,
)
from scenarioforge.scenario_intelligence.duplicate_detector import SimilarityIndex
from scenarioforge.scenario_intelligence.metrics import (
    coverage,
    jensen_shannon_divergence,
    normalized_entropy,
    shannon_entropy,
    simpson_diversity,
)
from scenarioforge.scenario_intelligence.schemas import ScenarioRecord


def compare_subset(
    original: Sequence[ScenarioRecord],
    subset: Sequence[ScenarioRecord],
    *,
    similarity: SimilarityIndex | None = None,
    similarity_threshold: float = 0.9,
    diversity_config: DiversityConfig | None = None,
) -> dict[str, Any]:
    """Before/after metrics for a subset against the original scenario set."""
    if not original:
        raise ValueError("Original scenario set is empty")
    if not subset:
        raise ValueError("Curated subset is empty; nothing to compare")
    original_ids = {r.sample_id for r in original}
    if not {r.sample_id for r in subset} <= original_ids:
        raise ValueError("Subset contains sample ids that are not in the original set")
    config = diversity_config or DiversityConfig()

    before, after = object_category_counts(original), object_category_counts(subset)
    taxonomy = sorted(before)
    result: dict[str, Any] = {
        "selected_count": len(subset),
        "original_count": len(original),
        "selection_ratio": len(subset) / len(original),
        "category_distribution": {"original": dict(sorted(before.items())), "curated": dict(sorted(after.items()))},
        "category_jsd_bits": jensen_shannon_divergence(before, after),
        "normalized_entropy": {
            "original": normalized_entropy(before, len(taxonomy)),
            "curated": normalized_entropy(after, len(taxonomy)),
        },
        "shannon_entropy_bits": {"original": shannon_entropy(before), "curated": shannon_entropy(after)},
        "simpson_diversity": {"original": simpson_diversity(before), "curated": simpson_diversity(after)},
        "category_coverage": coverage(after, taxonomy),
        "scenario_presence_coverage": coverage(scenario_presence_counts(subset), taxonomy),
    }

    discretizer = FeatureDiscretizer(config.binned_features, config.binning).fit(original)
    feature_jsd: dict[str, float | None] = {}
    for feature in config.binned_features:
        original_bins = Counter(discretizer.transform(r)[feature] for r in original)
        subset_bins = Counter(discretizer.transform(r)[feature] for r in subset)
        feature_jsd[feature] = jensen_shannon_divergence(original_bins, subset_bins)
    result["feature_distribution_jsd_bits"] = feature_jsd

    if similarity is not None:
        result["duplicate_candidates"] = {
            "threshold": similarity_threshold,
            "original": _pair_stats(similarity, [r.sample_id for r in original], similarity_threshold),
            "curated": _pair_stats(similarity, [r.sample_id for r in subset], similarity_threshold),
        }
    return result


def _pair_stats(similarity: SimilarityIndex, ids: Sequence[str], threshold: float) -> dict[str, float | int]:
    positions = {s: i for i, s in enumerate(similarity.sample_ids)}
    rows = [positions[s] for s in ids]
    sub = similarity.matrix[np.ix_(rows, rows)]
    upper = sub[np.triu_indices(len(rows), k=1)]
    pairs = int((upper >= threshold).sum())
    possible = len(upper)
    return {"pairs": pairs, "possible_pairs": possible, "rate": pairs / possible if possible else 0.0}
