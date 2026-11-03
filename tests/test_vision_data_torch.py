from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="Install the separate .venv-ml environment for PyTorch tests")
from PIL import Image

from scenarioforge.vision_data.dataloader import (
    detection_collate_fn,
    make_detection_dataloader,
    make_split_datasets,
    select_training_subset,
)
from scenarioforge.vision_data.dataset import CameraDetectionDataset
from scenarioforge.vision_data.splits import build_split_manifest
from scenarioforge.vision_data.transforms import (
    RandomHorizontalFlip,
    build_eval_transforms,
    build_train_transforms,
)


IMAGE_COLUMNS = (
    "image_id", "sample_token", "scene_token", "scene_name", "sample_data_token",
    "image_path", "width", "height", "image_status", "annotation_count",
    "rejected_annotation_count",
)
ANNOTATION_COLUMNS = (
    "annotation_id", "source_annotation_token", "sample_token", "scene_token",
    "camera_sample_data_token", "image_id", "image_path", "image_width",
    "image_height", "original_category", "detection_category", "class_id",
    "x_min", "y_min", "x_max", "y_max", "visibility_token", "visibility_level",
    "center_3d", "dimensions_3d", "instance_token", "conversion_status",
    "rejection_reason", "is_clipped", "projected_point_count",
)


def _fixture(tmp_path: Path, *, invalid_box: bool = False, missing_image: bool = False):
    root = tmp_path / "dataset"
    first_path = root / "samples" / "first.jpg"
    second_path = root / "samples" / "second.jpg"
    first_path.parent.mkdir(parents=True)
    Image.new("RGB", (40, 20), color=(20, 40, 60)).save(first_path)
    if not missing_image:
        Image.new("RGB", (40, 20), color=(60, 40, 20)).save(second_path)

    image_rows = [
        {
            "image_id": "sample-first", "sample_token": "sample-first",
            "scene_token": "scene-first", "scene_name": "first",
            "sample_data_token": "camera-first", "image_path": "samples/first.jpg",
            "width": 40, "height": 20, "image_status": "ok",
            "annotation_count": 2, "rejected_annotation_count": 0,
        },
        {
            "image_id": "sample-second", "sample_token": "sample-second",
            "scene_token": "scene-second", "scene_name": "second",
            "sample_data_token": "camera-second", "image_path": "samples/second.jpg",
            "width": 40, "height": 20, "image_status": "ok",
            "annotation_count": 0, "rejected_annotation_count": 0,
        },
    ]
    image_csv = tmp_path / "image_manifest.csv"
    with image_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=IMAGE_COLUMNS)
        writer.writeheader()
        writer.writerows(image_rows)
    annotation_csv = tmp_path / "annotations.csv"
    annotation_rows = []
    for index in range(2):
        x_min = -1 if invalid_box else index + 2
        annotation_rows.append({
            "annotation_id": f"annotation-{index}",
            "source_annotation_token": f"annotation-{index}",
            "sample_token": "sample-first", "scene_token": "scene-first",
            "camera_sample_data_token": "camera-first",
            "image_id": "sample-first", "image_path": "samples/first.jpg",
            "image_width": 40, "image_height": 20,
            "original_category": "vehicle.car", "detection_category": "car",
            "class_id": 3, "x_min": x_min, "y_min": 3,
            "x_max": index + 12, "y_max": 13,
            "visibility_token": "4", "visibility_level": "v80-100",
            "center_3d": "[]", "dimensions_3d": "[]",
            "instance_token": f"instance-{index}", "conversion_status": "accepted",
            "rejection_reason": "", "is_clipped": "False", "projected_point_count": 8,
        })
    with annotation_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(annotation_rows)
    mapping = tmp_path / "category_mapping.json"
    mapping.write_text(json.dumps({
        "version": "fixture",
        "background_class_id": 0,
        "classes": [
            {"class_id": 3, "name": "car", "source_categories": ["vehicle.car"]},
            {"class_id": 9, "name": "truck", "source_categories": ["vehicle.truck"]},
        ],
    }), encoding="utf-8")
    return root, annotation_csv, image_csv, mapping


def _split_factory_fixture(tmp_path: Path):
    root = tmp_path / "dataset"
    vision = tmp_path / "vision"
    vision.mkdir()
    image_rows = []
    for scene_index in range(4):
        for sample_index in range(2):
            sample = f"sample-{scene_index}-{sample_index}"
            relative_path = f"samples/{sample}.jpg"
            image_path = root / relative_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40, 20), color=(scene_index * 20, sample_index * 20, 40)).save(image_path)
            image_rows.append({
                "image_id": sample,
                "sample_token": sample,
                "scene_token": f"scene-{scene_index}",
                "scene_name": f"scene-name-{scene_index}",
                "sample_data_token": f"camera-{sample}",
                "image_path": relative_path,
                "width": 40,
                "height": 20,
                "image_status": "ok",
                "annotation_count": 0,
                "rejected_annotation_count": 0,
            })

    image_csv = vision / "image_manifest.csv"
    with image_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=IMAGE_COLUMNS)
        writer.writeheader()
        writer.writerows(image_rows)
    annotation_csv = vision / "annotations.csv"
    with annotation_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
    mapping = vision / "category_mapping.json"
    mapping.write_text(json.dumps({
        "version": "fixture",
        "background_class_id": 0,
        "classes": [{"class_id": 3, "name": "car", "source_categories": ["vehicle.car"]}],
    }), encoding="utf-8")
    (vision / "dataset_manifest.json").write_text(
        json.dumps({"dataset_version": "fixture-mini"}),
        encoding="utf-8",
    )
    manifest = build_split_manifest(image_rows, dataset_version="fixture-mini", seed=42)
    return root, vision, image_rows, manifest


def _reseal_manifest(manifest):
    payload = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    manifest["manifest_digest"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return manifest


def test_split_dataset_factory_and_training_selection_reject_forged_scene_assignments(tmp_path):
    root, vision, image_rows, manifest = _split_factory_fixture(tmp_path)
    datasets = make_split_datasets(
        dataset_root=str(root),
        annotations_csv=str(vision / "annotations.csv"),
        image_manifest_csv=str(vision / "image_manifest.csv"),
        category_mapping_json=str(vision / "category_mapping.json"),
        split_manifest=manifest,
    )
    assert {split: len(dataset) for split, dataset in datasets.items()} == {
        "train": 4,
        "validation": 2,
        "test": 2,
    }

    forged = json.loads(json.dumps(manifest))
    train_sample = forged["sample_tokens"]["train"][0]
    test_sample = forged["sample_tokens"]["test"][0]
    forged["sample_tokens"]["train"][0] = test_sample
    forged["sample_tokens"]["test"][0] = train_sample
    _reseal_manifest(forged)

    with pytest.raises(ValueError, match="Sample membership does not match scene assignment"):
        make_split_datasets(
            dataset_root=str(root),
            annotations_csv=str(vision / "annotations.csv"),
            image_manifest_csv=str(vision / "image_manifest.csv"),
            category_mapping_json=str(vision / "category_mapping.json"),
            split_manifest=forged,
        )
    with pytest.raises(ValueError, match="Sample membership does not match scene assignment"):
        select_training_subset(
            datasets["train"],
            forged,
            [datasets["train"].sample_tokens[0]],
        )


def test_split_dataset_factory_fails_closed_without_dataset_manifest(tmp_path):
    root, vision, _, manifest = _split_factory_fixture(tmp_path)
    (vision / "dataset_manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="dataset_manifest.json"):
        make_split_datasets(
            dataset_root=str(root),
            annotations_csv=str(vision / "annotations.csv"),
            image_manifest_csv=str(vision / "image_manifest.csv"),
            category_mapping_json=str(vision / "category_mapping.json"),
            split_manifest=manifest,
        )


def test_dataset_loads_lazy_images_and_groups_multiple_boxes_with_reversible_labels(tmp_path):
    root, annotations, images, mapping = _fixture(tmp_path)
    dataset = CameraDetectionDataset(root, annotations, images, category_mapping_json=mapping)

    assert len(dataset) == 2
    image, target = dataset[0]
    assert image.shape == (3, 20, 40)
    assert image.dtype == torch.float32 and 0 <= image.min() <= image.max() <= 1
    assert target["boxes"].shape == (2, 4)
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].tolist() == [1, 1]
    assert target["labels"].dtype == torch.int64
    assert target["image_id"].dtype == torch.int64
    assert target["area"].tolist() == [100.0, 100.0]
    assert target["iscrowd"].tolist() == [0, 0]
    assert dataset.source_id_to_model_label == {3: 1, 9: 2}
    assert dataset.model_label_to_source_id == {1: 3, 2: 9}
    assert dataset[0][1]["image_id"].item() == target["image_id"].item()
    assert dataset[0][1]["image_id"].item() != dataset[1][1]["image_id"].item()


def test_empty_annotation_image_has_correct_detection_tensor_shapes(tmp_path):
    root, annotations, images, mapping = _fixture(tmp_path)
    dataset = CameraDetectionDataset(root, annotations, images, category_mapping_json=mapping)

    image, target = dataset[1]
    assert image.shape == (3, 20, 40)
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
    assert target["area"].shape == (0,)
    assert target["iscrowd"].shape == (0,)


def test_missing_image_and_invalid_boxes_fail_clearly(tmp_path):
    root, annotations, images, mapping = _fixture(tmp_path, missing_image=True)
    with pytest.raises(FileNotFoundError, match="sample-second"):
        CameraDetectionDataset(root, annotations, images, category_mapping_json=mapping)

    root, annotations, images, mapping = _fixture(tmp_path / "invalid", invalid_box=True)
    with pytest.raises(ValueError, match="box is invalid or outside image bounds"):
        CameraDetectionDataset(root, annotations, images, category_mapping_json=mapping)


def test_resize_updates_boxes_and_empty_targets_synchronously(tmp_path):
    root, annotations, images, mapping = _fixture(tmp_path)
    transform = build_eval_transforms((10, 20))
    dataset = CameraDetectionDataset(
        root, annotations, images, category_mapping_json=mapping, transform=transform
    )
    image, target = dataset[0]
    assert image.shape == (3, 10, 20)
    assert target["boxes"][0].tolist() == [1.0, 1.5, 6.0, 6.5]
    assert target["area"][0].item() == 25.0
    empty_image, empty_target = dataset[1]
    assert empty_image.shape == (3, 10, 20)
    assert empty_target["boxes"].shape == (0, 4)
    assert empty_target["area"].shape == (0,)


def test_horizontal_flip_updates_boxes_and_eval_is_deterministic(tmp_path):
    root, annotations, images, mapping = _fixture(tmp_path)
    dataset = CameraDetectionDataset(
        root,
        annotations,
        images,
        category_mapping_json=mapping,
        transform=build_train_transforms(horizontal_flip_probability=1.0),
    )
    _, target = dataset[0]
    assert target["boxes"][0].tolist() == [28.0, 3.0, 38.0, 13.0]

    eval_dataset = CameraDetectionDataset(
        root, annotations, images, category_mapping_json=mapping,
        transform=build_eval_transforms((10, 20)),
    )
    image_a, target_a = eval_dataset[0]
    image_b, target_b = eval_dataset[0]
    assert torch.equal(image_a, image_b)
    assert torch.equal(target_a["boxes"], target_b["boxes"])


def test_dataloader_collates_variable_targets_and_seeded_order(tmp_path):
    root, annotations, images, mapping = _fixture(tmp_path)
    dataset = CameraDetectionDataset(root, annotations, images, category_mapping_json=mapping)
    first_batch = detection_collate_fn([dataset[0], dataset[1]])
    assert len(first_batch[0]) == len(first_batch[1]) == 2
    assert [target["boxes"].shape[0] for target in first_batch[1]] == [2, 0]

    def order(seed: int):
        loader = make_detection_dataloader(dataset, batch_size=1, shuffle=True, seed=seed)
        return [target["image_id"].item() for _, targets in loader for target in targets]

    assert order(42) == order(42)
    assert len(order(42)) == 2
    assert make_detection_dataloader(dataset).num_workers == 0


def test_annotation_image_association_mismatch_is_rejected(tmp_path):
    root, annotations, images, mapping = _fixture(tmp_path)
    rows = list(csv.DictReader(annotations.open(encoding="utf-8")))
    rows[0]["scene_token"] = "wrong-scene"
    with annotations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="does not match image manifest"):
        CameraDetectionDataset(root, annotations, images, category_mapping_json=mapping)


def test_nuscenes_mini_dataset_and_dataloader_smoke():
    root = Path("data/downloaded-dataset")
    vision = Path("data/processed/vision")
    if not root.is_dir() or not (vision / "annotations.csv").is_file():
        pytest.skip("nuScenes mini and Part 3A exports are not available")
    dataset = CameraDetectionDataset(
        root,
        vision / "annotations.csv",
        vision / "image_manifest.csv",
        category_mapping_json=vision / "category_mapping.json",
        sample_tokens=[
            "00889f8a9549450aa2f32cf310a3e305",
            "0307afa1ec7e48ce90295acabf8cb6dd",
        ],
    )
    images, targets = next(iter(make_detection_dataloader(dataset, batch_size=2)))
    assert len(images) == len(targets) == 2
    assert all(image.dtype == torch.float32 and image.shape[0] == 3 for image in images)
    assert all(target["boxes"].shape[1] == 4 for target in targets)
