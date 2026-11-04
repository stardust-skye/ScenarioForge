from __future__ import annotations

import copy

import pytest
from PIL import Image

from scenarioforge.scenario_intelligence.experiment_curation import (
    balanced_scene_quotas,
    select_train_records,
    sha256_file,
    validate_training_scenarios,
)
from scenarioforge.scenario_intelligence.rarity_analyzer import analyze_rarity
from scenarioforge.scenario_intelligence.scenario_ranker import RankingConfig
from scenarioforge.scenario_intelligence.schemas import ScenarioRecord
from scenarioforge.vision_data.splits import (
    build_split_manifest,
    validate_split_manifest,
)


@pytest.fixture
def curation_inputs(tmp_path):
    rows = []
    records = []
    assignments = {"train": [], "validation": ["scene-val"], "test": ["scene-test"]}
    for scene_index in range(3):
        scene = f"scene-train-{scene_index}"
        assignments["train"].append(scene)
        for sample_index in range(2):
            token = f"train-{scene_index}-{sample_index}"
            relative_path = f"samples/{token}.png"
            path = tmp_path / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 10)).save(path)
            rows.append({
                "image_id": token,
                "sample_token": token,
                "scene_token": scene,
                "scene_name": scene,
                "image_path": relative_path,
                "width": "20",
                "height": "10",
                "image_status": "ok",
                "annotation_count": "0",
            })
            records.append(ScenarioRecord(
                sample_id=token,
                scene_id=scene,
                scene_name=scene,
                timestamp=sample_index,
                features={
                    "category_counts": {"vehicle.car": sample_index + 1}
                    if scene_index == sample_index else {"human.pedestrian.adult": 1},
                    "group_counts": {"vehicle": 1},
                    "group_combination": "vehicle",
                    "category_combination": "vehicle.car",
                    "n_objects": scene_index + sample_index + 1,
                    "n_unique_instances": scene_index + sample_index + 1,
                    "n_distinct_categories": 1,
                    "n_pedestrians": sample_index,
                    "n_vehicles": scene_index,
                    "n_two_wheelers": 0,
                    "n_traffic_related": 0,
                    "ego_speed_mps": float(scene_index * 10 + sample_index),
                    "location": "train-location",
                },
            ))
    for split, scene in (("validation", "scene-val"), ("test", "scene-test")):
        token = f"sample-{split}"
        relative_path = f"samples/{token}.png"
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 10)).save(path)
        rows.append({
            "image_id": token,
            "sample_token": token,
            "scene_token": scene,
            "scene_name": scene,
            "image_path": relative_path,
            "width": "20",
            "height": "10",
            "image_status": "ok",
            "annotation_count": "0",
        })
        records.append(ScenarioRecord(
            sample_id=token,
            scene_id=scene,
            scene_name=scene,
            timestamp=0,
            features={"category_counts": {"heldout.original": 1}, "n_objects": 1},
        ))
    split = build_split_manifest(
        rows,
        dataset_version="mini",
        scene_assignments=assignments,
    )
    train_records, audit = validate_training_scenarios(
        records,
        split_manifest=split,
        image_rows=rows,
        annotations=[],
        dataset_root=tmp_path,
    )
    return tmp_path, rows, records, split, train_records, audit


@pytest.mark.parametrize(
    "strategy", ["random", "frequency", "diversity", "weighted_interest"]
)
def test_heldout_metadata_cannot_change_training_scores_or_selection(
    curation_inputs, strategy
):
    root, rows, records, split, train_records, _ = curation_inputs
    modified = copy.deepcopy(records)
    heldout = [record for record in modified if record.sample_id in {
        "sample-validation", "sample-test"
    }]
    for record in heldout:
        record.features = {
            "category_counts": {"entirely.different": 10_000},
            "n_objects": 999_999,
            "location": "heldout-mutated",
        }
    changed_train_records, _ = validate_training_scenarios(
        modified,
        split_manifest=split,
        image_rows=rows,
        annotations=[],
        dataset_root=root,
    )
    rarity_a = analyze_rarity(train_records)
    rarity_b = analyze_rarity(changed_train_records)
    assert rarity_a.to_dict() == rarity_b.to_dict()
    quotas = balanced_scene_quotas(
        {f"scene-train-{index}": 2 for index in range(3)}, 3
    )
    selected_a, _ = select_train_records(
        train_records,
        strategy=strategy,
        quotas=quotas,
        seed=17,
        rarity=rarity_a,
        ranking=RankingConfig(),
    )
    selected_b, _ = select_train_records(
        changed_train_records,
        strategy=strategy,
        quotas=quotas,
        seed=17,
        rarity=rarity_b,
        ranking=RankingConfig(),
    )
    assert [record.sample_id for record in selected_a] == [
        record.sample_id for record in selected_b
    ]


def test_empty_images_remain_train_candidates_and_are_audited(curation_inputs):
    _, _, _, _, train_records, audit = curation_inputs
    assert len(train_records) == 6
    assert all(item["accepted_2d_annotation_count"] == 0 for item in audit)
    assert all(item["empty_2d_target"] for item in audit)


@pytest.mark.parametrize(
    "strategy", ["random", "frequency", "diversity", "weighted_interest"]
)
def test_random_seed_reproducibility_and_exact_budget(curation_inputs, strategy):
    _, _, _, _, train_records, _ = curation_inputs
    quotas = balanced_scene_quotas(
        {f"scene-train-{index}": 2 for index in range(3)}, 3
    )
    rarity = analyze_rarity(train_records)
    selected_a, _ = select_train_records(
        train_records,
        strategy=strategy,
        quotas=quotas,
        seed=42,
        rarity=rarity,
        ranking=RankingConfig(),
    )
    selected_b, _ = select_train_records(
        train_records,
        strategy=strategy,
        quotas=quotas,
        seed=42,
        rarity=rarity,
        ranking=RankingConfig(),
    )
    ids_a = [record.sample_id for record in selected_a]
    assert ids_a == [record.sample_id for record in selected_b]
    assert len(ids_a) == len(set(ids_a)) == 3
    assert all(record.sample_id.startswith("train-") for record in selected_a)


def test_full_reference_quota_contains_every_training_image(curation_inputs):
    _, _, _, _, train_records, _ = curation_inputs
    counts = {
        scene: sum(record.scene_id == scene for record in train_records)
        for scene in sorted({record.scene_id for record in train_records})
    }
    quotas = balanced_scene_quotas(counts, len(train_records))
    selected, _ = select_train_records(
        train_records,
        strategy="random",
        quotas=quotas,
        seed=42,
        rarity=analyze_rarity(train_records),
        ranking=RankingConfig(),
    )
    assert {record.sample_id for record in selected} == {
        record.sample_id for record in train_records
    }


@pytest.mark.parametrize("mutation, message", [
    ("duplicate", "duplicate sample token"),
    ("unknown", "unknown sample token"),
    ("missing", "missing training sample tokens"),
    ("scene_mismatch", "scene mismatch"),
])
def test_training_scenario_identity_errors_are_rejected(
    curation_inputs, mutation, message
):
    root, rows, records, split, _, _ = curation_inputs
    changed = copy.deepcopy(records)
    if mutation == "duplicate":
        changed.append(copy.deepcopy(changed[0]))
    elif mutation == "unknown":
        changed[0].sample_id = "not-in-image-manifest"
    elif mutation == "missing":
        changed = changed[:-1]
        changed = [record for record in changed if record.sample_id != "train-0-0"]
    else:
        changed[0].scene_id = "wrong-scene"
    with pytest.raises(ValueError, match=message):
        validate_training_scenarios(
            changed,
            split_manifest=split,
            image_rows=rows,
            annotations=[],
            dataset_root=root,
        )


def test_missing_training_image_is_rejected(curation_inputs):
    root, rows, records, split, _, _ = curation_inputs
    missing_root = root / "missing"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_training_scenarios(
            records,
            split_manifest=split,
            image_rows=rows,
            annotations=[],
            dataset_root=missing_root,
        )


def test_training_annotation_count_mismatch_is_rejected(curation_inputs):
    root, rows, records, _, _, _ = curation_inputs
    changed_rows = [dict(row) for row in rows]
    changed_rows[0]["annotation_count"] = "1"
    split = build_split_manifest(
        changed_rows,
        dataset_version="mini",
        scene_assignments={
            "train": ["scene-train-0", "scene-train-1", "scene-train-2"],
            "validation": ["scene-val"],
            "test": ["scene-test"],
        },
    )
    with pytest.raises(ValueError, match="Accepted annotation count mismatch"):
        validate_training_scenarios(
            records,
            split_manifest=split,
            image_rows=changed_rows,
            annotations=[],
            dataset_root=root,
        )


@pytest.mark.parametrize(
    "strategy", ["random", "frequency", "diversity", "weighted_interest"]
)
def test_all_strategies_obey_identical_scene_quotas(curation_inputs, strategy):
    _, _, _, _, train_records, _ = curation_inputs
    quotas = balanced_scene_quotas(
        {f"scene-train-{index}": 2 for index in range(3)}, 3
    )
    selected, _ = select_train_records(
        train_records,
        strategy=strategy,
        quotas=quotas,
        seed=3,
        rarity=analyze_rarity(train_records),
        ranking=RankingConfig(),
    )
    assert len(selected) == 3
    assert {
        scene: sum(record.scene_id == scene for record in selected)
        for scene in quotas
    } == quotas


def test_balanced_scene_quotas_respect_capacity_and_fail_when_impossible():
    assert balanced_scene_quotas({"a": 2, "b": 5, "c": 5}, 8) == {
        "a": 2, "b": 3, "c": 3
    }
    with pytest.raises(ValueError, match="impossible"):
        balanced_scene_quotas({"a": 2, "b": 1}, 4)


def test_changed_inventory_and_file_contents_change_provenance(curation_inputs, tmp_path):
    _, rows, _, split, _, _ = curation_inputs
    changed_rows = [dict(row) for row in rows]
    changed_rows[0]["image_path"] = "samples/changed.png"
    with pytest.raises(ValueError, match="inventory provenance"):
        validate_split_manifest(split, changed_rows, dataset_version="mini")

    source = tmp_path / "manifest.csv"
    source.write_text("original\n", encoding="utf-8")
    original_hash = sha256_file(source)
    source.write_text("modified\n", encoding="utf-8")
    assert sha256_file(source) != original_hash
