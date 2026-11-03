"""Scenario-interest ranking.

The *scenario-interest score* is a transparent heuristic for prioritising which keyframes to
look at or curate first. It is NOT a safety, risk, or collision score and says nothing about model
performance.

Components, each min-max normalised to [0, 1] across the ranked set:
  * rarity                - combined frequency-based rarity (see ``rarity_analyzer``)
  * composition_diversity - normalised entropy of the category mix inside the scenario
  * complexity            - mean of normalised object count, unique-instance count, and
                            distinct-category count
  * preference            - weighted mean of user-chosen normalised features (default: none)

score = sum_k w_k * c_k / sum_k w_k over components that are available for the scenario; missing
components are excluded and the remaining weights renormalised, and the explanation says so.
Sorting is by descending score, then sample id, so ranking is fully deterministic.
Default positive component weights are equal by design, not because they have been calibrated.
Min-max normalization is relative to the supplied scenario set; constant components map to 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from scenarioforge.scenario_intelligence.metrics import min_max_normalize
from scenarioforge.scenario_intelligence.rarity_analyzer import RarityReport
from scenarioforge.scenario_intelligence.scenario_features import within_sample_entropy
from scenarioforge.scenario_intelligence.schemas import ScenarioRecord

COMPLEXITY_FEATURES = ("n_objects", "n_unique_instances", "n_distinct_categories")


@dataclass(frozen=True)
class RankingConfig:
    weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "rarity": 1.0,
            "composition_diversity": 1.0,
            "complexity": 1.0,
            "preference": 0.0,
        }
    )
    # feature name -> importance; e.g. {"n_pedestrians": 1.0} to favour pedestrian-rich scenes
    preferences: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.weights) - {"rarity", "composition_diversity", "complexity", "preference"}
        if unknown:
            raise ValueError(f"Unknown ranking components: {sorted(unknown)}")
        if any(not isfinite(w) or w < 0 for w in self.weights.values()) or not any(self.weights.values()):
            raise ValueError("Ranking weights must be non-negative with at least one positive")
        if any(not isfinite(w) or w < 0 for w in self.preferences.values()):
            raise ValueError("Preference weights must be finite and non-negative")
        if self.weights.get("preference", 0) > 0 and (
            not self.preferences or not any(self.preferences.values())
        ):
            raise ValueError("A positive 'preference' weight requires preference features")


@dataclass
class RankedScenario:
    rank: int
    sample_id: str
    scene_id: str
    score: float | None
    components: dict[str, float | None]
    contributions: dict[str, float]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "sample_id": self.sample_id,
            "scene_id": self.scene_id,
            "scenario_interest_score": self.score,
            "components": self.components,
            "contributions": self.contributions,
            "explanation": self.explanation,
        }


def rank_scenarios(
    records: Sequence[ScenarioRecord],
    rarity: RarityReport,
    config: RankingConfig | None = None,
) -> list[RankedScenario]:
    """Rank scenarios by scenario-interest score, returning component scores and explanations."""
    config = config or RankingConfig()
    if not records:
        return []

    rarity_scores = {sid: s.combined for sid, s in rarity.scenario_scores.items()}
    diversity = {
        r.sample_id: (
            within_sample_entropy(r.features["category_counts"])
            if r.features.get("category_counts") is not None
            else None
        )
        for r in records
    }
    complexity = _mean_normalized(records, COMPLEXITY_FEATURES)
    preference = _weighted_normalized(records, config.preferences)

    columns = {
        "rarity": min_max_normalize({r.sample_id: rarity_scores.get(r.sample_id) for r in records}),
        "composition_diversity": diversity,  # already in [0, 1]
        "complexity": complexity,
        "preference": preference,
    }
    scored: list[tuple[ScenarioRecord, float | None, dict[str, float | None], dict[str, float]]] = []
    for record in records:
        components = {name: col.get(record.sample_id) for name, col in columns.items()}
        active = {
            n: v for n, v in components.items() if v is not None and config.weights.get(n, 0) > 0
        }
        total = sum(config.weights[n] for n in active)
        contributions = {n: config.weights[n] * v / total for n, v in active.items()} if total else {}
        score = sum(contributions.values()) if total else None
        scored.append((record, score, components, contributions))

    scored.sort(key=lambda item: (item[1] is None, -(item[1] or 0.0), item[0].sample_id))
    return [
        RankedScenario(
            rank=position,
            sample_id=record.sample_id,
            scene_id=record.scene_id,
            score=score,
            components=components,
            contributions=contributions,
            explanation=_explain(components, contributions, config.weights),
        )
        for position, (record, score, components, contributions) in enumerate(scored, start=1)
    ]


def _mean_normalized(
    records: Sequence[ScenarioRecord], features: Sequence[str]
) -> dict[str, float | None]:
    normalized = {f: min_max_normalize({r.sample_id: r.features.get(f) for r in records}) for f in features}
    result: dict[str, float | None] = {}
    for record in records:
        values = [normalized[f][record.sample_id] for f in features]
        # Complexity is one composite component; it is unavailable unless all three documented
        # constituent features are present, rather than silently changing its definition per row.
        result[record.sample_id] = None if any(v is None for v in values) else sum(values) / len(values)
    return result


def _weighted_normalized(
    records: Sequence[ScenarioRecord], preferences: Mapping[str, float]
) -> dict[str, float | None]:
    if not preferences:
        return {r.sample_id: None for r in records}
    normalized = {
        f: min_max_normalize({r.sample_id: r.features.get(f) for r in records}) for f in preferences
    }
    total = sum(preferences.values())
    if total <= 0:
        return {r.sample_id: None for r in records}
    result: dict[str, float | None] = {}
    for record in records:
        values = [(preferences[f], normalized[f][record.sample_id]) for f in preferences]
        result[record.sample_id] = (
            None if any(v is None for _, v in values) else sum(w * v for w, v in values) / total
        )
    return result


def _explain(
    components: Mapping[str, float | None],
    contributions: Mapping[str, float],
    weights: Mapping[str, float],
) -> str:
    if not contributions:
        return "No scoring component was available for this scenario."
    parts = [
        f"{name}={components[name]:.3f} (weight {weights[name]:g}, adds {value:.3f})"
        for name, value in sorted(contributions.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    skipped = sorted(n for n, w in weights.items() if w > 0 and components.get(n) is None)
    note = f"; unavailable and excluded: {', '.join(skipped)}" if skipped else ""
    return "scenario-interest = " + " + ".join(parts) + note
