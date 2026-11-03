"""Tests for scenario intelligence.

All fixtures here are SYNTHETIC and exist only to exercise logic; they say nothing about nuScenes.
"""

from __future__ import annotations

import csv
import json
import math

import pytest

from scenarioforge.scenario_intelligence import reporting as rep
from scenarioforge.scenario_intelligence.cli import main
from scenarioforge.scenario_intelligence.config import ScenarioIntelligenceConfig
from scenarioforge.scenario_intelligence.curation_engine import CurationConfig, curate
from scenarioforge.scenario_intelligence.diversity_analyzer import analyze_diversity
from scenarioforge.scenario_intelligence.duplicate_detector import (
    DuplicateConfig,
    compute_similarity,
    detect_duplicates,
    fingerprint,
)
from scenarioforge.scenario_intelligence.metrics import (
    jensen_shannon_divergence,
    normalized_entropy,
    shannon_entropy,
    simpson_diversity,
)
from scenarioforge.scenario_intelligence.rarity_analyzer import analyze_rarity
from scenarioforge.scenario_intelligence.rarity_analyzer import RarityConfig
from scenarioforge.scenario_intelligence.scenario_extractor import extract_scenarios
from scenarioforge.scenario_intelligence.scenario_features import category_group
from scenarioforge.scenario_intelligence.scenario_ranker import RankingConfig, rank_scenarios
from scenarioforge.scenario_intelligence.schemas import (
    AnnotationInput,
    EgoPose,
    SampleInput,
    ScenarioRecord,
    SensorRefInput,
)


def _ann(i: int, category: str, xy=(0.0, 0.0)) -> AnnotationInput:
    return AnnotationInput(
        source_id=f"a{i}", category=category, instance_source_id=f"i{i}",
        translation=(xy[0], xy[1], 0.0), visibility_level="v80-100",
        attribute_names=(), num_lidar_pts=5, num_radar_pts=0,
    )


def _sample(sid, scene="s1", ts=0, cats=("vehicle.car",), pose=None, sensors=True) -> SampleInput:
    refs = (SensorRefInput("r1", "CAM_FRONT", "camera", sensors),) if sensors is not None else ()
    return SampleInput(
        sample_id=sid, scene_id=scene, scene_name=scene, timestamp=ts, scene_sample_count=3,
        scene_duration_s=1.0, location="loc",
        annotations=tuple(_ann(i, c, (10.0, 0.0)) for i, c in enumerate(cats)),
        sensor_refs=refs, ego_pose=pose, expected_channels=("CAM_FRONT",),
    )


def _pose(ts, x):
    return EgoPose(timestamp=ts, translation=(x, 0.0, 0.0), rotation=(1.0, 0.0, 0.0, 0.0))


def _records():
    cats = [("vehicle.car",)] * 6 + [("vehicle.car", "human.pedestrian.adult")] * 3 + [
        ("vehicle.bicycle", "human.pedestrian.adult", "movable_object.barrier")
    ]
    samples = [
        _sample(f"s{i:02d}", scene="a" if i < 5 else "b", ts=i * 500_000, cats=c, pose=_pose(i * 500_000, i * 5.0))
        for i, c in enumerate(cats)
    ]
    return extract_scenarios(samples)


# ---- metrics ----
def test_entropy_known_values():
    assert shannon_entropy({"a": 1, "b": 1}) == pytest.approx(1.0)
    assert normalized_entropy({"a": 5, "b": 5, "c": 5}) == pytest.approx(1.0)
    assert normalized_entropy({"a": 10}) == 0.0
    assert simpson_diversity({"a": 1, "b": 1}) == pytest.approx(0.5)


def test_jsd_bounds_and_symmetry():
    p, q = {"a": 1, "b": 0}, {"a": 0, "b": 1}
    assert jensen_shannon_divergence(p, q) == pytest.approx(1.0)
    assert jensen_shannon_divergence(p, p) == pytest.approx(0.0)
    assert jensen_shannon_divergence({"a": 3, "b": 1}, {"a": 1, "b": 3}) == pytest.approx(
        jensen_shannon_divergence({"a": 1, "b": 3}, {"a": 3, "b": 1})
    )


# ---- grouping and extraction ----
@pytest.mark.parametrize("category,group", [
    ("human.pedestrian.adult", "pedestrian"), ("vehicle.car", "vehicle"),
    ("vehicle.bicycle", "two_wheeler"), ("vehicle.motorcycle", "two_wheeler"),
    ("movable_object.barrier", "traffic_related"), ("static_object.bicycle_rack", "static_object"),
    ("animal", "animal"), ("something.else", "other"),
])
def test_category_group(category, group):
    assert category_group(category) == group


def test_extraction_features_and_speed():
    records = _records()
    first, second = records[0], records[1]
    assert first.features["ego_speed_mps"] is None  # no previous keyframe in scene
    assert second.features["ego_speed_mps"] == pytest.approx(10.0)  # 5 m in 0.5 s
    assert records[-1].features["n_two_wheelers"] == 1
    assert records[-1].features["group_combination"] == "pedestrian|traffic_related|two_wheeler"


def test_missing_metadata_is_none_not_zero():
    sample = _sample("x", pose=None)
    record = extract_scenarios([sample])[0]
    assert record.features["ego_x"] is None
    assert "ego_x" in record.missing_features


def test_duplicate_sample_ids_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        extract_scenarios([_sample("x"), _sample("x", ts=1)])


# ---- rarity / diversity ----
def test_rarity_orders_rare_category_higher():
    records = _records()
    report = analyze_rarity(records)
    scores = report.scenario_scores
    assert scores["s09"].category > scores["s00"].category
    assert report.category_frequency["vehicle.car"] > report.category_frequency["vehicle.bicycle"]


def test_diversity_report_deterministic():
    records = _records()
    assert json.dumps(analyze_diversity(records), sort_keys=True) == json.dumps(
        analyze_diversity(list(reversed(records))), sort_keys=True
    )


def test_missing_categories_are_not_scored_as_a_rare_combination():
    record = extract_scenarios([_sample("empty", cats=())])[0]
    rarity = analyze_rarity([record])
    diversity = analyze_diversity([record])

    score = rarity.scenario_scores["empty"]
    assert score.category is None
    assert score.combination is None
    assert score.raw["combination"] is None
    assert not score.is_rare_combination
    assert rarity.missing_combination_count == 1
    assert rarity.combination_frequency == {}
    combination_report = diversity["category_combination_distribution"]
    assert combination_report["missing_count"] == 1
    assert combination_report["coverage_fraction"] == 0.0
    assert combination_report["total"] == 0


def test_unknown_category_maps_to_other_group_and_stays_explicit():
    record = extract_scenarios([_sample("unknown", cats=("unmapped.category",))])[0]

    assert record.features["group_counts"] == {"other": 1}
    assert record.features["category_combination"] == "unmapped.category"
    assert analyze_diversity([record])["category_group_distribution"]["counts"]["other"] == 1


# ---- duplicates ----
def test_fingerprint_and_exact_duplicates():
    records = _records()
    cfg = DuplicateConfig()
    assert fingerprint(records[0], cfg.fingerprint_features) == fingerprint(records[2], cfg.fingerprint_features)
    report = detect_duplicates(records, cfg)
    assert report["n_exact_duplicate_samples"] >= 2


def test_similarity_scores_identical_composition_high_and_distinct_low():
    records = _records()
    index = compute_similarity(records, DuplicateConfig())
    assert index.similarity("s00", "s01") > index.similarity("s00", "s09")


def _similar_record(sample_id: str, scene_id: str, timestamp: int, attribute: str) -> ScenarioRecord:
    return ScenarioRecord(
        sample_id=sample_id,
        scene_id=scene_id,
        scene_name=scene_id,
        timestamp=timestamp,
        features={
            "category_counts": {"vehicle.car": 5},
            "attribute_counts": {attribute: 1},
            "visibility_counts": {"v80-100": 5},
            "location": "same-location",
            "n_objects": 5,
            "n_unique_instances": 5,
            "ego_speed_mps": 1.0,
            "n_objects_near_ego": 5,
        },
    )


def _temporal_records():
    return [
        _similar_record("a0", "scene-a", 0, "a0"),
        _similar_record("a1", "scene-a", 300_000, "a1"),
        _similar_record("a2", "scene-a", 600_000, "a2"),
        _similar_record("a3", "scene-a", 3_000_000, "a3"),
        _similar_record("b0", "scene-b", 0, "b0"),
    ]


def test_near_duplicate_pairs_are_classified_by_scene_and_time():
    report = detect_duplicates(_temporal_records(), DuplicateConfig(similarity_threshold=0.9))
    assert report["n_near_duplicate_pairs_unfiltered"] == 10
    assert report["pair_classification_counts_unfiltered"] == {
        "consecutive": 3,
        "temporally_adjacent": 1,
        "non_adjacent_same_scene": 2,
        "cross_scene": 4,
    }
    pair = next(p for p in report["near_duplicate_pairs"]
                if {p["sample_a"], p["sample_b"]} == {"a0", "a2"})
    assert pair["temporal_relationship"] == "temporally_adjacent"
    assert pair["scene_sample_order_gap"] == 2
    assert pair["time_separation_s"] == pytest.approx(0.6)
    assert set(pair["similarity_components"]) == {
        "categories", "category_counts", "numeric", "context"
    }
    assert sum(
        component["weighted_contribution"]
        for component in pair["similarity_components"].values()
    ) == pytest.approx(pair["similarity"])


def test_missing_numeric_and_context_components_are_explained_as_unavailable():
    first = _similar_record("first", "scene", 0, "first")
    second = _similar_record("second", "scene", 2_000_000, "second")
    first.features["ego_speed_mps"] = None
    second.features["ego_speed_mps"] = None
    second.features["location"] = None

    report = detect_duplicates([first, second])
    pair = report["near_duplicate_pairs"][0]

    assert pair["similarity_components"]["numeric"]["similarity"] is None
    assert pair["similarity_components"]["context"]["similarity"] is None
    assert sum(
        component["weighted_contribution"]
        for component in pair["similarity_components"].values()
    ) == pytest.approx(pair["similarity"])


def test_duplicate_scene_and_temporal_filters_do_not_change_scores():
    records = _temporal_records()
    all_pairs = detect_duplicates(records, DuplicateConfig())
    cross = detect_duplicates(records, DuplicateConfig(scene_scope="cross"))
    same = detect_duplicates(records, DuplicateConfig(scene_scope="same"))
    separated = detect_duplicates(records, DuplicateConfig(min_temporal_separation_s=0.5))
    assert cross["n_near_duplicate_pairs"] == 4
    assert same["n_near_duplicate_pairs"] == 6
    assert separated["n_near_duplicate_pairs"] == 8
    excluding_adjacent = detect_duplicates(
        records, DuplicateConfig(exclude_temporally_adjacent=True)
    )
    assert excluding_adjacent["n_near_duplicate_pairs"] == 6
    assert excluding_adjacent["exclude_temporally_adjacent"] is True
    assert all(
        p["temporal_relationship"] not in {"consecutive", "temporally_adjacent"}
        for p in excluding_adjacent["near_duplicate_pairs"]
    )
    assert cross["n_near_duplicate_pairs_unfiltered"] == all_pairs["n_near_duplicate_pairs_unfiltered"]
    score = {(p["sample_a"], p["sample_b"]): p["similarity"] for p in all_pairs["near_duplicate_pairs"]}
    assert all(score[(p["sample_a"], p["sample_b"])] == p["similarity"]
               for p in cross["near_duplicate_pairs"])
    tighter_adjacency = detect_duplicates(
        records, DuplicateConfig(temporal_adjacency_s=0.5)
    )
    assert tighter_adjacency["pair_classification_counts_unfiltered"]["temporally_adjacent"] == 0


def test_duplicate_config_rejects_invalid_scope_and_separation():
    with pytest.raises(ValueError, match="scene_scope"):
        DuplicateConfig(scene_scope="invalid")
    with pytest.raises(ValueError, match="finite and non-negative"):
        DuplicateConfig(min_temporal_separation_s=-1)


# ---- ranking ----
def test_ranking_consistent_and_explained():
    records = _records()
    rarity = analyze_rarity(records)
    first = rank_scenarios(records, rarity)
    second = rank_scenarios(list(reversed(records)), rarity)
    assert [r.sample_id for r in first] == [r.sample_id for r in second]
    assert [r.rank for r in first] == list(range(1, len(first) + 1))
    assert first[0].explanation
    assert first[0].sample_id == "s09"


def test_ranking_components_missing_values_and_preference_are_explicit():
    records = _records()
    records[0].features["n_unique_instances"] = None
    rarity = analyze_rarity(records)
    ranked = rank_scenarios(
        records,
        rarity,
        RankingConfig(
            weights={"rarity": 1.0, "composition_diversity": 0.0, "complexity": 1.0, "preference": 1.0},
            preferences={"n_pedestrians": 1.0},
        ),
    )
    missing = next(row for row in ranked if row.sample_id == records[0].sample_id)
    assert missing.components["complexity"] is None
    assert "unavailable and excluded: complexity" in missing.explanation
    assert set(missing.contributions) == {"rarity", "preference"}


def test_rarity_config_threshold_and_component_weights_are_configurable():
    records = _records()
    low_threshold = analyze_rarity(records, RarityConfig(rare_max_share=0.01))
    high_threshold = analyze_rarity(
        records,
        RarityConfig(rare_max_share=1.0, weights={"category": 2.0, "combination": 1.0, "feature": 0.0}),
    )
    assert len(low_threshold.rare_combinations) <= len(high_threshold.rare_combinations)
    assert high_threshold.config.weights["category"] == 2.0
    assert all("category" in score.raw and "combination" in score.raw for score in high_threshold.scenario_scores.values())


def test_rarity_and_ranking_reject_invalid_weights():
    with pytest.raises(ValueError, match="epsilon"):
        RarityConfig(epsilon=float("nan"))
    with pytest.raises(ValueError, match="weights"):
        RarityConfig(weights={"unknown": 1.0})
    with pytest.raises(ValueError, match="Preference weights"):
        RankingConfig(preferences={"n_objects": -1.0})
    with pytest.raises(ValueError, match="preference"):
        RankingConfig(weights={"preference": 1.0}, preferences={"n_objects": 0.0})


def test_json_style_config_overrides_thresholds_weights_and_duplicate_scope():
    config = ScenarioIntelligenceConfig.from_dict({
        "rarity": {
            "rare_max_share": 0.1,
            "weights": {"category": 2.0, "combination": 1.0, "feature": 0.0},
            "binning": {"strategy": "uniform", "n_bins": 3},
        },
        "duplicates": {"scene_scope": "cross", "min_temporal_separation_s": 0.5},
        "ranking": {
            "weights": {"rarity": 2.0, "preference": 1.0},
            "preferences": {"n_pedestrians": 1.0},
        },
    })
    assert config.rarity.rare_max_share == 0.1
    assert config.rarity.binning.strategy == "uniform"
    assert config.duplicates.scene_scope == "cross"
    assert config.duplicates.min_temporal_separation_s == 0.5
    assert config.ranking.weights["rarity"] == 2.0


# ---- curation ----
def _curate(strategy, n=4, seed=1, **kw):
    records = _records()
    cfg = CurationConfig(strategy=strategy, n_samples=n, seed=seed, **kw)
    return curate(records, cfg, dataset_source_id="synthetic", rarity=analyze_rarity(records),
                  similarity=compute_similarity(records, DuplicateConfig()))


@pytest.mark.parametrize("strategy", ["random", "frequency", "diversity", "weighted_interest"])
def test_curation_strategies_reproducible(strategy):
    a, b = _curate(strategy), _curate(strategy)
    assert [r.sample_id for r in a.selected] == [r.sample_id for r in b.selected]
    assert a.manifest["manifest_sha256"] == b.manifest["manifest_sha256"]
    assert len(a.selected) == 4
    assert len({r.sample_id for r in a.selected}) == 4
    assert a.manifest["config"]["strategy"] == strategy
    assert len(a.manifest["selected"]) == 4
    assert all({"sample_id", "scene_id"} <= row.keys() for row in a.manifest["selected"])


@pytest.mark.parametrize("strategy", ["random", "frequency", "diversity", "weighted_interest"])
def test_all_curation_strategies_respect_constraints_when_feasible(strategy):
    result = _curate(
        strategy,
        n=2,
        require_groups=("pedestrian",),
        min_features={"n_objects": 2},
        max_features={"n_objects": 2},
    )
    assert len(result.selected) == 2
    assert len({record.sample_id for record in result.selected}) == 2
    for record in result.selected:
        assert record.features["group_counts"]["pedestrian"] >= 1
        assert record.features["n_objects"] == 2


def test_random_seed_changes_selection():
    assert [r.sample_id for r in _curate("random", seed=1).selected] != [
        r.sample_id for r in _curate("random", seed=2).selected
    ]


def test_curation_constraints_and_shortfall():
    result = _curate("random", n=5, require_groups=("two_wheeler",))
    assert [r.sample_id for r in result.selected] == ["s09"]
    assert result.shortfall_reason and "Requested 5" in result.shortfall_reason


def test_curation_similarity_constraint_is_applied():
    records = _temporal_records()
    records[-1].features["location"] = "different-location"
    similarity = compute_similarity(records, DuplicateConfig())
    result = curate(
        records,
        CurationConfig(strategy="random", n_samples=2, seed=3,
                       max_near_duplicate_similarity=1.0),
        dataset_source_id="synthetic",
        rarity=analyze_rarity(records),
        similarity=similarity,
    )
    assert len(result.selected) == 2
    assert similarity.similarity(result.selected[0].sample_id, result.selected[1].sample_id) < 1.0


def test_invalid_curation_config():
    with pytest.raises(ValueError):
        CurationConfig(strategy="nope", n_samples=1)
    with pytest.raises(ValueError):
        CurationConfig(strategy="random", n_samples=0)


# ---- reporting / CLI ----
def test_scenario_round_trip(tmp_path):
    records = _records()
    path = rep.save_scenarios(records, tmp_path / "s.json", "synthetic")
    dataset_id, loaded = rep.load_scenarios(path)
    assert dataset_id == "synthetic" and [r.to_dict() for r in loaded] == [r.to_dict() for r in records]


def test_grouped_distribution_plot_is_created(tmp_path):
    path = rep.plot_grouped_distribution(
        {"vehicle.car": 5, "human.pedestrian.adult": 2},
        {"vehicle.car": 3, "human.pedestrian.adult": 4},
        "Synthetic distribution",
        tmp_path / "distribution.png",
    )
    assert path.is_file() and path.stat().st_size > 0


def test_cli_missing_input_returns_usage_error(tmp_path):
    assert main(["--output-dir", str(tmp_path), "--no-plots", "stats"]) == 2


def test_cli_analysis_pipeline_on_synthetic_file(tmp_path):
    rep.save_scenarios(_records(), tmp_path / rep.SCENARIO_FILE, "synthetic")
    for command in (["stats"], ["rarity"], ["diversity"], ["duplicates"], ["rank"],
                    ["curate", "--strategy", "diversity", "--n", "3"]):
        assert main(["--output-dir", str(tmp_path), "--no-plots", *command]) == 0, command
    manifest = tmp_path / "curation_diversity_n3_seed0_manifest.json"
    assert main(["--output-dir", str(tmp_path), "--no-plots", "compare", "--manifest", str(manifest)]) == 0
    report = json.loads((tmp_path / "comparison_curation_diversity_n3_seed0.json").read_text())
    assert not math.isnan(report["category_jsd_bits"])


def test_cli_duplicate_filters_expose_pair_classes(tmp_path):
    rep.save_scenarios(_temporal_records(), tmp_path / rep.SCENARIO_FILE, "synthetic")
    assert main([
        "--output-dir", str(tmp_path), "--no-plots", "duplicates", "--scene-scope", "cross",
    ]) == 0
    report = json.loads((tmp_path / "duplicate_report.json").read_text())
    assert report["n_near_duplicate_pairs"] == 4
    assert report["pair_classification_counts_unfiltered"]["consecutive"] == 3
    assert main([
        "--output-dir", str(tmp_path), "--no-plots", "duplicates",
        "--exclude-temporally-adjacent",
    ]) == 0
    report = json.loads((tmp_path / "duplicate_report.json").read_text())
    assert report["exclude_temporally_adjacent"] is True
    assert report["n_near_duplicate_pairs"] == 6
    with (tmp_path / "near_duplicate_pairs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and rows[0]["similarity_components_json"]


def test_cli_bad_weight_returns_usage_error(tmp_path):
    rep.save_scenarios(_records(), tmp_path / rep.SCENARIO_FILE, "synthetic")
    assert main(["--output-dir", str(tmp_path), "--no-plots", "rank", "--weight", "oops"]) == 2
