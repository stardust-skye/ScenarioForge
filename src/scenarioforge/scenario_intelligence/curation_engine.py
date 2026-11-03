"""Reproducible curation of metadata-defined subsets.

Strategies:
  * ``random``            - seeded uniform shuffle (baseline)
  * ``frequency``         - highest combined inverse-frequency rarity first
  * ``diversity``         - greedy: repeatedly add the scenario that most increases the Shannon
                            entropy of the selected set's object-category counts (ties by id)
  * ``weighted_interest`` - highest scenario-interest score first (configurable weights)

Constraints are never silently relaxed. Scenarios failing a constraint are counted per reason as
rejections; if fewer than the requested number can be selected, the result reports the shortfall
and its causes instead of padding the subset.

Determinism: input is sorted by sample id, randomness comes only from ``random.Random(seed)``,
and the manifest contains no wall-clock time, so identical inputs and config give an identical
manifest (and manifest hash).
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from scenarioforge.scenario_intelligence.duplicate_detector import SimilarityIndex
from scenarioforge.scenario_intelligence.metrics import shannon_entropy
from scenarioforge.scenario_intelligence.rarity_analyzer import RarityReport
from scenarioforge.scenario_intelligence.scenario_ranker import RankingConfig, rank_scenarios
from scenarioforge.scenario_intelligence.schemas import ScenarioRecord

STRATEGIES = ("random", "frequency", "diversity", "weighted_interest")
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class CurationConfig:
    strategy: str
    n_samples: int
    seed: int = 0
    require_groups: tuple[str, ...] = ()  # e.g. ("pedestrian",) -> need >=1 pedestrian
    min_features: Mapping[str, float] = field(default_factory=dict)
    max_features: Mapping[str, float] = field(default_factory=dict)
    max_near_duplicate_similarity: float | None = None  # None disables the limit
    ranking: RankingConfig = field(default_factory=RankingConfig)

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{self.strategy}'; choose from {STRATEGIES}")
        if self.n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        if self.max_near_duplicate_similarity is not None and not (
            0 < self.max_near_duplicate_similarity <= 1
        ):
            raise ValueError("max_near_duplicate_similarity must be in (0, 1]")


@dataclass
class CurationResult:
    config: CurationConfig
    selected: list[ScenarioRecord]
    selection_scores: dict[str, float | None]
    rejected_by_reason: dict[str, int]
    n_input: int
    n_eligible: int
    shortfall_reason: str | None
    manifest: dict[str, Any]


def curate(
    records: Sequence[ScenarioRecord],
    config: CurationConfig,
    *,
    dataset_source_id: str,
    rarity: RarityReport | None = None,
    similarity: SimilarityIndex | None = None,
) -> CurationResult:
    """Select a subset according to ``config``; see module docstring for guarantees."""
    ordered = sorted(records, key=lambda r: r.sample_id)
    if len({r.sample_id for r in ordered}) != len(ordered):
        raise ValueError("Input contains duplicate sample ids")
    if config.strategy in {"frequency", "weighted_interest"} and rarity is None:
        raise ValueError(f"Strategy '{config.strategy}' requires a rarity report")
    if config.max_near_duplicate_similarity is not None and similarity is None:
        raise ValueError("A near-duplicate limit requires a similarity index")

    eligible, rejected = _apply_filters(ordered, config)
    scores: dict[str, float | None] = {}

    if config.strategy == "random":
        order = list(eligible)
        random.Random(config.seed).shuffle(order)
        selected = _take_in_order(order, config, similarity)
    elif config.strategy == "frequency":
        assert rarity is not None
        scores = {r.sample_id: rarity.scenario_scores[r.sample_id].combined for r in eligible}
        order = sorted(eligible, key=lambda r: (scores[r.sample_id] is None, -(scores[r.sample_id] or 0), r.sample_id))
        selected = _take_in_order(order, config, similarity)
    elif config.strategy == "weighted_interest":
        assert rarity is not None
        ranked = rank_scenarios(eligible, rarity, config.ranking)
        scores = {r.sample_id: r.score for r in ranked}
        by_id = {r.sample_id: r for r in eligible}
        selected = _take_in_order([by_id[r.sample_id] for r in ranked], config, similarity)
    else:
        selected = _greedy_diverse(eligible, config, similarity)

    near_dup_blocked = _count_near_duplicate_blocked(eligible, selected, config, similarity)
    shortfall = None
    if len(selected) < config.n_samples:
        shortfall = (
            f"Requested {config.n_samples} but selected {len(selected)}: {len(eligible)} of "
            f"{len(ordered)} scenarios passed the constraints"
            + (f"; the near-duplicate limit excluded {near_dup_blocked} further eligible scenarios"
               if near_dup_blocked else "")
            + "."
        )

    manifest = build_manifest(
        ordered, selected, scores, config, dataset_source_id, rejected, len(eligible), shortfall
    )
    return CurationResult(config, selected, scores, rejected, len(ordered), len(eligible), shortfall, manifest)


def _apply_filters(
    records: Sequence[ScenarioRecord], config: CurationConfig
) -> tuple[list[ScenarioRecord], dict[str, int]]:
    rejected: Counter[str] = Counter()
    eligible: list[ScenarioRecord] = []
    for record in records:
        reason = _rejection_reason(record, config)
        if reason:
            rejected[reason] += 1
        else:
            eligible.append(record)
    return eligible, dict(sorted(rejected.items()))


def _rejection_reason(record: ScenarioRecord, config: CurationConfig) -> str | None:
    groups = record.features.get("group_counts") or {}
    for group in config.require_groups:
        if groups.get(group, 0) < 1:
            return f"missing_required_group:{group}"
    for feature, minimum in config.min_features.items():
        value = record.features.get(feature)
        if value is None:
            return f"missing_feature:{feature}"
        if value < minimum:
            return f"below_min:{feature}"
    for feature, maximum in config.max_features.items():
        value = record.features.get(feature)
        if value is None:
            return f"missing_feature:{feature}"
        if value > maximum:
            return f"above_max:{feature}"
    return None


def _too_similar(
    candidate: ScenarioRecord,
    selected: Sequence[ScenarioRecord],
    config: CurationConfig,
    similarity: SimilarityIndex | None,
) -> bool:
    limit = config.max_near_duplicate_similarity
    if limit is None or similarity is None or not selected:
        return False
    return any(similarity.similarity(candidate.sample_id, s.sample_id) >= limit for s in selected)


def _take_in_order(
    order: Sequence[ScenarioRecord], config: CurationConfig, similarity: SimilarityIndex | None
) -> list[ScenarioRecord]:
    selected: list[ScenarioRecord] = []
    for record in order:
        if len(selected) >= config.n_samples:
            break
        if not _too_similar(record, selected, config, similarity):
            selected.append(record)
    return selected


def _greedy_diverse(
    eligible: Sequence[ScenarioRecord], config: CurationConfig, similarity: SimilarityIndex | None
) -> list[ScenarioRecord]:
    selected: list[ScenarioRecord] = []
    totals: Counter[str] = Counter()
    remaining = list(eligible)
    while remaining and len(selected) < config.n_samples:
        best: ScenarioRecord | None = None
        best_gain = float("-inf")
        for candidate in remaining:  # remaining is sorted by id => deterministic ties
            if _too_similar(candidate, selected, config, similarity):
                continue
            trial = totals + Counter(candidate.features.get("category_counts") or {})
            gain = shannon_entropy(trial)
            if gain > best_gain + 1e-12:
                best, best_gain = candidate, gain
        if best is None:
            break
        selected.append(best)
        totals.update(best.features.get("category_counts") or {})
        remaining.remove(best)
    return selected


def _count_near_duplicate_blocked(
    eligible: Sequence[ScenarioRecord],
    selected: Sequence[ScenarioRecord],
    config: CurationConfig,
    similarity: SimilarityIndex | None,
) -> int:
    if config.max_near_duplicate_similarity is None or similarity is None:
        return 0
    chosen = {s.sample_id for s in selected}
    return sum(
        1 for r in eligible if r.sample_id not in chosen and _too_similar(r, selected, config, similarity)
    )


def _category_totals(records: Sequence[ScenarioRecord]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for record in records:
        totals.update(record.features.get("category_counts") or {})
    return dict(sorted(totals.items()))


def input_digest(records: Sequence[ScenarioRecord]) -> str:
    """SHA-256 over canonical JSON of all input scenarios (identifies the exact input data)."""
    payload = [r.to_dict() for r in sorted(records, key=lambda r: r.sample_id)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_manifest(
    all_records: Sequence[ScenarioRecord],
    selected: Sequence[ScenarioRecord],
    scores: Mapping[str, float | None],
    config: CurationConfig,
    dataset_source_id: str,
    rejected: Mapping[str, int],
    n_eligible: int,
    shortfall: str | None,
) -> dict[str, Any]:
    """Build the reproducibility manifest (no timestamps, so it is deterministic)."""
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_source_id": dataset_source_id,
        "scope": "keyframe-level metadata scenarios; sweeps not included",
        "input_digest_sha256": input_digest(all_records),
        "config": asdict(config),
        "counts": {
            "input": len(all_records),
            "eligible": n_eligible,
            "requested": config.n_samples,
            "selected": len(selected),
            "rejected_by_constraints": len(all_records) - n_eligible,
            "not_selected_eligible": n_eligible - len(selected),
            "selection_ratio": len(selected) / len(all_records) if all_records else None,
        },
        "rejected_by_reason": dict(rejected),
        "requested_count_satisfied": shortfall is None,
        "shortfall_reason": shortfall,
        "category_distribution_original": _category_totals(all_records),
        "category_distribution_selected": _category_totals(selected),
        "selected": [
            {
                "selection_rank": position,
                "sample_id": r.sample_id,
                "scene_id": r.scene_id,
                "scene_name": r.scene_name,
                "timestamp": r.timestamp,
                "selection_score": scores.get(r.sample_id),
            }
            for position, r in enumerate(selected, start=1)
        ],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return manifest


CSV_COLUMNS = (
    "selection_rank", "sample_id", "scene_id", "scene_name", "timestamp", "selection_score",
    "n_objects", "n_pedestrians", "n_vehicles", "category_combination",
)


def export_curation(result: CurationResult, json_path: Path, csv_path: Path) -> None:
    """Write the manifest JSON and a CSV of selected ids with key metadata."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for position, record in enumerate(result.selected, start=1):
            writer.writerow(
                [position, record.sample_id, record.scene_id, record.scene_name, record.timestamp,
                 result.selection_scores.get(record.sample_id)]
                + [record.features.get(c) for c in CSV_COLUMNS[6:]]
            )
