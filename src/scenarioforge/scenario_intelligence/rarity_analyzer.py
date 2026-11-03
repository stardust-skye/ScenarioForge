"""Frequency-based rarity analysis.

Rarity here means *statistically infrequent within this dataset*. It is not danger, importance, or
model difficulty: a rare scenario can be irrelevant and a common one can be critical.

Scoring: for an item with frequency f (number of scenarios exhibiting it),
``rarity = 1 / (f + epsilon)``. Per-scenario components are min-max normalised to [0, 1] across
scenarios and combined with configurable weights.

Components:
  * category    - the highest inverse frequency among the categories present in the scenario
  * combination - inverse frequency of the scenario's exact category (or group) combination
  * feature     - mean inverse frequency of the scenario's binned feature values

Limitations of inverse-frequency scoring: it depends on dataset size and binning; every
unique value scores as maximally rare (so noise looks rare); a single very rare category
dominates the category component; it ignores correlation between features; and with only the
mini split's 10 scenes, "rare" often reflects scene identity rather than a general phenomenon.

The default 5% rarity threshold and equal component weights are configurable heuristics, not
empirically validated thresholds. Component scores, raw inverse-frequency values, and weights
are included in the report.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from scenarioforge.scenario_intelligence.binning import BinningConfig, FeatureDiscretizer
from scenarioforge.scenario_intelligence.metrics import (
    inverse_frequency,
    min_max_normalize,
)
from scenarioforge.scenario_intelligence.schemas import MISSING_BIN, ScenarioRecord

DEFAULT_BINNED_FEATURES = (
    "n_objects",
    "n_pedestrians",
    "n_vehicles",
    "n_two_wheelers",
    "n_traffic_related",
    "n_unique_instances",
    "ego_speed_mps",
    "location",
)


@dataclass(frozen=True)
class RarityConfig:
    epsilon: float = 1.0
    rare_max_share: float = 0.05  # an item is "rare" if it occurs in <= this share of scenarios
    # "group" is the default: on nuScenes mini, "category" flags 322/404 scenarios as rare
    combination_level: str = "group"  # "category" or "group"
    binned_features: tuple[str, ...] = DEFAULT_BINNED_FEATURES
    binning: BinningConfig = field(default_factory=BinningConfig)
    weights: Mapping[str, float] = field(
        default_factory=lambda: {"category": 1.0, "combination": 1.0, "feature": 1.0}
    )

    def __post_init__(self) -> None:
        if not isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if self.combination_level not in {"category", "group"}:
            raise ValueError("combination_level must be 'category' or 'group'")
        if not isfinite(self.rare_max_share) or not 0 < self.rare_max_share <= 1:
            raise ValueError("rare_max_share must be in (0, 1]")
        if (
            set(self.weights) - {"category", "combination", "feature"}
            or any(not isfinite(w) or w < 0 for w in self.weights.values())
            or not any(self.weights.values())
        ):
            raise ValueError("rarity weights must be non-negative with at least one positive")


@dataclass
class ScenarioRarity:
    sample_id: str
    category: float | None
    combination: float | None
    feature: float | None
    combined: float | None
    raw: dict[str, float | None]
    is_rare_combination: bool


@dataclass
class RarityReport:
    n_scenarios: int
    config: RarityConfig
    category_frequency: dict[str, int]
    combination_frequency: dict[str, int]
    missing_combination_count: int
    feature_value_frequency: dict[str, dict[str, int]]
    rare_categories: list[str]
    rare_combinations: list[str]
    rare_feature_values: dict[str, list[str]]
    scenario_scores: dict[str, ScenarioRarity]
    scene_summary: dict[str, dict[str, Any]]
    bin_edges: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_scenarios": self.n_scenarios,
            "epsilon": self.config.epsilon,
            "rare_max_share": self.config.rare_max_share,
            "combination_level": self.config.combination_level,
            "weights": dict(self.config.weights),
            "bin_edges": self.bin_edges,
            "category_frequency": self.category_frequency,
            "combination_frequency": self.combination_frequency,
            "missing_combination_count": self.missing_combination_count,
            "feature_value_frequency": self.feature_value_frequency,
            "rare_categories": self.rare_categories,
            "rare_combinations": self.rare_combinations,
            "rare_feature_values": self.rare_feature_values,
            "scene_summary": self.scene_summary,
            "scenario_scores": {
                sid: {
                    "category": s.category,
                    "combination": s.combination,
                    "feature": s.feature,
                    "combined": s.combined,
                    "raw": s.raw,
                    "is_rare_combination": s.is_rare_combination,
                }
                for sid, s in self.scenario_scores.items()
            },
        }


def combination_key(record: ScenarioRecord, level: str) -> str | None:
    feature = "category_combination" if level == "category" else "group_combination"
    return record.features.get(feature) or None


def analyze_rarity(
    records: Sequence[ScenarioRecord], config: RarityConfig | None = None
) -> RarityReport:
    """Compute category, combination, and feature-value frequencies and rarity scores."""
    config = config or RarityConfig()
    if not records:
        raise ValueError("Cannot analyze rarity of an empty scenario set")
    n = len(records)
    threshold = config.rare_max_share * n

    category_frequency: Counter[str] = Counter()
    combination_frequency: Counter[str] = Counter()
    missing_combination_count = 0
    for record in records:
        category_frequency.update((record.features.get("category_counts") or {}).keys())
        combination = combination_key(record, config.combination_level)
        if combination is None:
            missing_combination_count += 1
        else:
            combination_frequency[combination] += 1

    discretizer = FeatureDiscretizer(config.binned_features, config.binning).fit(records)
    binned = {r.sample_id: discretizer.transform(r) for r in records}
    feature_frequency: dict[str, Counter[str]] = defaultdict(Counter)
    for labels in binned.values():
        for feature, label in labels.items():
            feature_frequency[feature][label] += 1

    raw_category: dict[str, float | None] = {}
    raw_combination: dict[str, float | None] = {}
    raw_feature: dict[str, float | None] = {}
    for record in records:
        counts = record.features.get("category_counts") or {}
        raw_category[record.sample_id] = (
            max(inverse_frequency(category_frequency[c], config.epsilon) for c in counts)
            if counts
            else None
        )
        combination = combination_key(record, config.combination_level)
        raw_combination[record.sample_id] = (
            inverse_frequency(combination_frequency[combination], config.epsilon)
            if combination is not None
            else None
        )
        known = [
            inverse_frequency(feature_frequency[f][label], config.epsilon)
            for f, label in binned[record.sample_id].items()
            if label != MISSING_BIN
        ]
        raw_feature[record.sample_id] = sum(known) / len(known) if known else None

    norm_category = min_max_normalize(raw_category)
    norm_combination = min_max_normalize(raw_combination)
    norm_feature = min_max_normalize(raw_feature)

    scores: dict[str, ScenarioRarity] = {}
    for record in records:
        sid = record.sample_id
        components = {
            "category": norm_category[sid],
            "combination": norm_combination[sid],
            "feature": norm_feature[sid],
        }
        combination = combination_key(record, config.combination_level)
        combo_count = combination_frequency[combination] if combination is not None else 0
        scores[sid] = ScenarioRarity(
            sample_id=sid,
            category=components["category"],
            combination=components["combination"],
            feature=components["feature"],
            combined=_weighted_mean(components, config.weights),
            raw={
                "category": raw_category[sid],
                "combination": raw_combination[sid],
                "feature": raw_feature[sid],
            },
            is_rare_combination=combination is not None and combo_count <= threshold,
        )

    scene_summary = _scene_summary(records, scores)
    return RarityReport(
        n_scenarios=n,
        config=config,
        category_frequency=dict(sorted(category_frequency.items())),
        combination_frequency=dict(
            sorted(combination_frequency.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        missing_combination_count=missing_combination_count,
        feature_value_frequency={f: dict(sorted(c.items())) for f, c in sorted(feature_frequency.items())},
        rare_categories=sorted(c for c, f in category_frequency.items() if f <= threshold),
        rare_combinations=sorted(c for c, f in combination_frequency.items() if f <= threshold),
        rare_feature_values={
            f: sorted(v for v, count in c.items() if count <= threshold and v != MISSING_BIN)
            for f, c in sorted(feature_frequency.items())
        },
        scenario_scores=scores,
        scene_summary=scene_summary,
        bin_edges=discretizer.edges or {},
    )


def _weighted_mean(
    components: Mapping[str, float | None], weights: Mapping[str, float]
) -> float | None:
    """Weighted mean over available components; weights renormalise when some are missing."""
    available = {k: v for k, v in components.items() if v is not None and weights.get(k, 0) > 0}
    total = sum(weights[k] for k in available)
    if total == 0:
        return None
    return sum(weights[k] * v for k, v in available.items()) / total


def _scene_summary(
    records: Sequence[ScenarioRecord], scores: Mapping[str, ScenarioRarity]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ScenarioRecord]] = defaultdict(list)
    for record in records:
        grouped[record.scene_id].append(record)
    summary: dict[str, dict[str, Any]] = {}
    for scene_id, items in sorted(grouped.items()):
        combined = [scores[r.sample_id].combined for r in items if scores[r.sample_id].combined is not None]
        summary[scene_id] = {
            "scene_name": items[0].scene_name,
            "n_samples": len(items),
            "n_rare_combination_samples": sum(1 for r in items if scores[r.sample_id].is_rare_combination),
            "mean_combined_rarity": sum(combined) / len(combined) if combined else None,
        }
    return summary
