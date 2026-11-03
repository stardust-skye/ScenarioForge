"""Dataset diversity analysis over categorical and binned features.

Interpretation of reported values (see ``metrics.py`` for definitions):
  * normalized entropy: 1 means categories are used equally often, 0 means one category holds
    everything. It is relative to the observed support unless a taxonomy size is given.
  * Simpson diversity: probability that two random draws fall in different categories.
  * Herfindahl / top-k share / imbalance ratio: concentration and skew; high = dominated by few
    categories.
  * coverage: share of a feature's records with a known value, and of possible bins/categories
    that actually occur. Coverage describes presence, not representativeness.

Continuous features are discretised first (see ``binning.py``); entropy of a raw continuous
variable is not meaningful and is never computed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from scenarioforge.scenario_intelligence.binning import BinningConfig, FeatureDiscretizer
from scenarioforge.scenario_intelligence.metrics import distribution_summary
from scenarioforge.scenario_intelligence.rarity_analyzer import DEFAULT_BINNED_FEATURES
from scenarioforge.scenario_intelligence.scenario_features import CATEGORY_GROUPS
from scenarioforge.scenario_intelligence.schemas import MISSING_BIN, ScenarioRecord


@dataclass(frozen=True)
class DiversityConfig:
    binned_features: tuple[str, ...] = DEFAULT_BINNED_FEATURES
    binning: BinningConfig = field(default_factory=BinningConfig)
    top_k: int = 3
    taxonomy_size: int | None = None  # full nuScenes category count, if known


def object_category_counts(records: Sequence[ScenarioRecord]) -> Counter[str]:
    """Total annotated objects per category across scenarios."""
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.features.get("category_counts") or {})
    return counts


def scenario_presence_counts(records: Sequence[ScenarioRecord]) -> Counter[str]:
    """Number of scenarios containing each category at least once."""
    counts: Counter[str] = Counter()
    for record in records:
        counts.update((record.features.get("category_counts") or {}).keys())
    return counts


def group_counts(records: Sequence[ScenarioRecord]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for group, value in (record.features.get("group_counts") or {}).items():
            counts[group] += value
    return counts


def analyze_diversity(
    records: Sequence[ScenarioRecord], config: DiversityConfig | None = None
) -> dict[str, Any]:
    """Return a machine-readable diversity report for a scenario set."""
    config = config or DiversityConfig()
    if not records:
        raise ValueError("Cannot analyze diversity of an empty scenario set")
    summarize = lambda counts, support=None: distribution_summary(  # noqa: E731
        counts, support_size=support, top_k=config.top_k
    )

    objects = object_category_counts(records)
    presence = scenario_presence_counts(records)
    combinations = Counter(
        combination
        for r in records
        if (combination := r.features.get("category_combination"))
    )
    missing_combinations = len(records) - sum(combinations.values())
    groups = group_counts(records)

    discretizer = FeatureDiscretizer(config.binned_features, config.binning).fit(records)
    binned_counts: dict[str, Counter[str]] = {f: Counter() for f in config.binned_features}
    for record in records:
        for feature, label in discretizer.transform(record).items():
            binned_counts[feature][label] += 1

    features: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    for feature, counts in binned_counts.items():
        known = {label: c for label, c in counts.items() if label != MISSING_BIN}
        n_known = sum(known.values())
        features[feature] = {
            "distribution": dict(sorted(counts.items())),
            **summarize(known),
        }
        coverage[feature] = {
            "non_missing_fraction": n_known / len(records),
            "distinct_values_or_bins": len(known),
        }

    scene_entropy: dict[str, float] = {}
    by_scene: dict[str, Counter[str]] = {}
    for record in records:
        by_scene.setdefault(record.scene_id, Counter()).update(record.features.get("category_counts") or {})
    for scene_id, counts in sorted(by_scene.items()):
        scene_entropy[scene_id] = summarize(counts)["normalized_entropy"]

    return {
        "n_scenarios": len(records),
        "object_category_distribution": {
            "counts": dict(sorted(objects.items())),
            **summarize(objects, config.taxonomy_size),
        },
        "scenario_category_presence": {"counts": dict(sorted(presence.items())), **summarize(presence)},
        "category_combination_distribution": {
            "n_distinct_combinations": len(combinations),
            "singleton_combinations": sum(1 for c in combinations.values() if c == 1),
            "missing_count": missing_combinations,
            "coverage_fraction": (len(records) - missing_combinations) / len(records),
            **summarize(combinations),
        },
        "category_group_distribution": {
            "counts": {g: groups.get(g, 0) for g in CATEGORY_GROUPS},
            **summarize(groups),
        },
        "feature_diversity": features,
        "feature_coverage": coverage,
        "scene_category_normalized_entropy": scene_entropy,
        "bin_edges": discretizer.edges or {},
        "interpretation": {
            "normalized_entropy": "1 = perfectly even use of observed categories; 0 = one category only.",
            "simpson_diversity": "Probability two random draws differ; higher = more diverse.",
            "herfindahl_index": "Concentration; higher = dominated by few categories.",
            "coverage": "Presence of values only, not representativeness.",
        },
    }
