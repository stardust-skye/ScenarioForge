from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scenarioforge.vision_data.splits import (
    _validate_manifest_structure,
    build_split_manifest,
    load_split_manifest,
    validate_equal_training_budgets,
    validate_split_manifest,
    validate_training_selection,
    write_split_manifest,
)
from scenarioforge.vision_data.statistics import compute_split_statistics, write_split_statistics


def _images(scene_count: int = 10, samples_per_scene: int = 2):
    rows = []
    for scene_index in range(scene_count):
        for sample_index in range(samples_per_scene):
            token = f"sample-{scene_index}-{sample_index}"
            rows.append({
                "image_id": token,
                "sample_token": token,
                "scene_token": f"scene-{scene_index}",
                "image_path": f"samples/{token}.jpg",
                "width": "100",
                "height": "50",
                "image_status": "ok",
            })
    return rows


def _reseal_manifest(manifest):
    payload = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    manifest["manifest_digest"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return manifest


def test_scene_splits_are_deterministic_grouped_and_seeded():
    images = _images()
    first = build_split_manifest(images, dataset_version="mini", seed=42)
    repeated = build_split_manifest(images, dataset_version="mini", seed=42)
    different = build_split_manifest(images, dataset_version="mini", seed=17)

    assert first == repeated
    assert first["scene_tokens"] != different["scene_tokens"]
    assert validate_split_manifest(first, images, dataset_version="mini")["scene_overlap_count"] == 0
    assert sum(map(len, first["sample_tokens"].values())) == len(images)
    for split in ("train", "validation", "test"):
        assert len(first["scene_tokens"][split]) in (2, 6)
        assert len(first["sample_tokens"][split]) == 2 * len(first["scene_tokens"][split])


def test_explicit_assignments_cover_scenes_and_reject_invalid_membership():
    images = _images()
    assignments = {
        "train": [f"scene-{index}" for index in range(6)],
        "validation": ["scene-6", "scene-7"],
        "test": ["scene-8", "scene-9"],
    }
    manifest = build_split_manifest(
        images,
        dataset_version="mini",
        scene_assignments=assignments,
    )
    assert manifest["strategy"] == "explicit_scene_assignments"
    assert len(manifest["sample_tokens"]["train"]) == 12

    with pytest.raises(ValueError, match="assigned more than once"):
        build_split_manifest(
            images,
            dataset_version="mini",
            scene_assignments={
                "train": ["scene-0"],
                "validation": ["scene-0"],
                "test": [f"scene-{i}" for i in range(1, 10)],
            },
        )
    with pytest.raises(ValueError, match="unassigned"):
        build_split_manifest(
            images,
            dataset_version="mini",
            scene_assignments={
                "train": ["scene-0"],
                "validation": ["scene-1"],
                "test": ["scene-2"],
            },
        )


def test_insufficient_groups_and_invalid_ratios_fail_explicitly():
    with pytest.raises(ValueError, match="Cannot populate all requested splits"):
        build_split_manifest(_images(scene_count=2), dataset_version="mini")
    with pytest.raises(ValueError, match="sum to 1.0"):
        build_split_manifest(
            _images(),
            dataset_version="mini",
            ratios={"train": 0.7, "validation": 0.2, "test": 0.2},
        )
    with pytest.raises(ValueError, match="ratio must be positive"):
        build_split_manifest(
            _images(),
            dataset_version="mini",
            ratios={"train": float("nan"), "validation": 0.5, "test": 0.5},
        )


def test_split_manifest_is_saved_with_digest_and_detects_tampering(tmp_path):
    manifest = build_split_manifest(_images(), dataset_version="mini", seed=23)
    path = write_split_manifest(
        manifest,
        tmp_path / "splits.json",
        image_rows=_images(),
        dataset_version="mini",
    )
    assert load_split_manifest(
        path,
        image_rows=_images(),
        dataset_version="mini",
    ) == manifest

    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["sample_tokens"]["test"].append("extra")
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_split_manifest(path, image_rows=_images(), dataset_version="mini")


def test_authoritative_validation_rejects_resealed_scene_leakage_exploit():
    images = _images()
    manifest = build_split_manifest(images, dataset_version="mini", seed=42)
    forged = json.loads(json.dumps(manifest))
    train_sample = forged["sample_tokens"]["train"][0]
    test_sample = forged["sample_tokens"]["test"][0]
    forged["sample_tokens"]["train"][0] = test_sample
    forged["sample_tokens"]["test"][0] = train_sample
    _reseal_manifest(forged)

    _validate_manifest_structure(forged)
    assert forged["manifest_digest"] != manifest["manifest_digest"]
    with pytest.raises(ValueError, match="Sample membership does not match scene assignment"):
        validate_split_manifest(forged, images, dataset_version="mini")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unknown_sample", "Sample membership does not match scene assignment"),
        ("incorrect_sample_scene", "Sample membership does not match scene assignment"),
        ("scene_overlap", "Scenes overlap between splits"),
        ("unexpected_scene", "Scene coverage mismatch"),
        ("missing_scene", "Scene coverage mismatch"),
        ("duplicate_sample", "Samples overlap between splits"),
        ("missing_sample", "Sample membership does not match scene assignment"),
        ("unexpected_sample", "Sample membership does not match scene assignment"),
    ],
)
def test_authoritative_validation_rejects_invalid_inventory_membership(mutation, message):
    images = _images()
    manifest = build_split_manifest(images, dataset_version="mini", seed=42)
    forged = json.loads(json.dumps(manifest))
    if mutation == "unknown_sample":
        forged["sample_tokens"]["test"][0] = "unknown-sample"
    elif mutation == "incorrect_sample_scene":
        forged["sample_tokens"]["train"][0], forged["sample_tokens"]["test"][0] = (
            forged["sample_tokens"]["test"][0],
            forged["sample_tokens"]["train"][0],
        )
    elif mutation == "scene_overlap":
        forged["scene_tokens"]["train"].append(forged["scene_tokens"]["test"][0])
    elif mutation == "unexpected_scene":
        forged["scene_tokens"]["train"].append("unknown-scene")
    elif mutation == "missing_scene":
        forged["scene_tokens"]["train"].pop()
    elif mutation == "duplicate_sample":
        forged["sample_tokens"]["test"].append(forged["sample_tokens"]["train"][0])
    elif mutation == "missing_sample":
        forged["sample_tokens"]["train"].pop()
    else:
        forged["sample_tokens"]["test"].append("unexpected-sample")
    _reseal_manifest(forged)

    with pytest.raises(ValueError, match=message):
        validate_split_manifest(forged, images, dataset_version="mini")


def test_authoritative_validation_rejects_version_and_inventory_mismatches():
    images = _images()
    manifest = build_split_manifest(images, dataset_version="mini", seed=42)

    with pytest.raises(ValueError, match="dataset_version"):
        validate_split_manifest(manifest, images, dataset_version="different-mini")

    changed_inventory = [dict(row) for row in images]
    changed_inventory[0]["image_path"] = "samples/replaced.jpg"
    with pytest.raises(ValueError, match="inventory provenance"):
        validate_split_manifest(manifest, changed_inventory, dataset_version="mini")


def test_training_selection_cannot_contain_heldout_and_budgets_must_match():
    manifest = build_split_manifest(_images(), dataset_version="mini", seed=42)
    train_samples = manifest["sample_tokens"]["train"]
    images = _images()
    audit = validate_training_selection(
        manifest,
        train_samples[:2],
        image_rows=images,
        dataset_version="mini",
    )
    assert audit["heldout_samples_unchanged"]
    assert audit["training_budget"] == 2
    with pytest.raises(ValueError, match="non-training samples"):
        validate_training_selection(
            manifest,
            [manifest["sample_tokens"]["test"][0]],
            image_rows=images,
            dataset_version="mini",
        )
    with pytest.raises(ValueError, match="duplicate"):
        validate_training_selection(
            manifest,
            [train_samples[0], train_samples[0]],
            image_rows=images,
            dataset_version="mini",
        )
    with pytest.raises(ValueError, match="unequal training budgets"):
        validate_equal_training_budgets(
            manifest,
            {"random": train_samples[:2], "diverse": train_samples[:3]},
            image_rows=images,
            dataset_version="mini",
        )
    assert validate_equal_training_budgets(
        manifest,
        {"random": train_samples[:2], "diverse": train_samples[2:4]},
        image_rows=images,
        dataset_version="mini",
    )["common_training_budget"] == 2


def test_split_statistics_include_paths_boxes_empty_images_and_scene_rows(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    images = _images(scene_count=3, samples_per_scene=2)
    for row in images:
        path = root / row["image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    manifest = build_split_manifest(
        images,
        dataset_version="mini",
        scene_assignments={
            "train": ["scene-0"],
            "validation": ["scene-1"],
            "test": ["scene-2"],
        },
    )
    annotations = [{
        "source_annotation_token": "annotation-0",
        "image_id": "sample-0-0",
        "sample_token": "sample-0-0",
        "scene_token": "scene-0",
        "image_path": "samples/sample-0-0.jpg",
        "image_width": "100",
        "image_height": "50",
        "class_id": "1",
        "detection_category": "car",
        "x_min": "2",
        "y_min": "3",
        "x_max": "12",
        "y_max": "13",
    }]
    report = compute_split_statistics(
        images,
        annotations,
        manifest,
        root,
        dataset_version="mini",
    )
    train = report["splits"]["train"]
    assert train["scene_count"] == 1
    assert train["image_count"] == 2
    assert train["annotation_count"] == 1
    assert train["empty_image_count"] == 1
    assert train["class_distribution"] == {"car": 1}
    assert train["missing_image_path_count"] == 0
    assert report["leakage_validation"]["valid"]
    outputs = write_split_statistics(report, tmp_path / "reports")
    assert all(path.is_file() for path in outputs.values())


def test_split_statistics_report_missing_paths_invalid_and_unassigned_annotations(tmp_path):
    images = _images(scene_count=3, samples_per_scene=1)
    manifest = build_split_manifest(
        images,
        dataset_version="mini",
        scene_assignments={
            "train": ["scene-0"],
            "validation": ["scene-1"],
            "test": ["scene-2"],
        },
    )
    annotations = [
        {
            "source_annotation_token": "bad",
            "image_id": "sample-0-0",
            "sample_token": "sample-0-0",
            "scene_token": "scene-0",
            "image_path": "samples/sample-0-0.jpg",
            "image_width": "100",
            "image_height": "50",
            "class_id": "1",
            "detection_category": "car",
            "x_min": "-1",
            "y_min": "3",
            "x_max": "12",
            "y_max": "13",
        },
        {"source_annotation_token": "orphan", "sample_token": "unknown-sample"},
    ]
    report = compute_split_statistics(
        images,
        annotations,
        manifest,
        tmp_path,
        dataset_version="mini",
    )
    assert report["splits"]["train"]["missing_image_path_count"] == 1
    assert report["splits"]["train"]["invalid_annotation_count"] == 1
    assert report["unassigned_annotation_count"] == 1
    assert not report["leakage_validation"]["valid"]
