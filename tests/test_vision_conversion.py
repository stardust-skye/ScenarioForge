from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from scenarioforge.vision.category_mapping import CategoryMapping, DetectionClass
from scenarioforge.vision.config import ProjectionConfig
from scenarioforge.vision.converter import convert_camera_samples
from scenarioforge.vision.dataset_builder import write_conversion
from scenarioforge.vision.inspection import render_inspection

IDENTITY = [1.0, 0.0, 0.0, 0.0]
CALIBRATION = {
    "translation": [0.0, 0.0, 0.0],
    "rotation": IDENTITY,
    "camera_intrinsic": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
}
POSE = {"translation": [0.0, 0.0, 0.0], "rotation": IDENTITY}


def _source_annotation(token="ann-1", category="vehicle.car", center=(0, 0, 10), dimensions=(2, 2, 2)):
    return {
        "source_annotation_token": token,
        "instance_token": f"instance-{token}",
        "category": category,
        "translation": list(center),
        "size": list(dimensions),
        "rotation": IDENTITY,
        "visibility_token": "4",
    }


def _frame(root: Path, annotations=None, **overrides):
    image_path = root / "samples" / "CAM_FRONT" / "sample.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not overrides.get("omit_image"):
        Image.new("RGB", (100, 100), color=(30, 40, 50)).save(image_path)
    payload = {
        "sample_token": "sample-token",
        "scene_token": "scene-token",
        "scene_name": "scene-name",
        "sample_data_token": "camera-token",
        "image_path": image_path,
        "width": 100,
        "height": 100,
        "calibration": CALIBRATION,
        "ego_pose": POSE,
        "camera_error": None,
        "annotations": list(annotations if annotations is not None else [_source_annotation()]),
    }
    payload.update({key: value for key, value in overrides.items() if key != "omit_image"})
    return payload


def _convert(root: Path, frames, **kwargs):
    return convert_camera_samples(
        frames,
        dataset_root=root,
        dataset_version="v1.0-mini",
        visibility_levels={"4": "v80-100"},
        processing_timestamp="2026-01-01T00:00:00+00:00",
        **kwargs,
    )


def test_valid_conversion_preserves_provenance_and_mapping(tmp_path):
    result = _convert(tmp_path, [_frame(tmp_path)])

    assert len(result.accepted_annotations) == 1
    annotation = result.accepted_annotations[0]
    assert annotation.source_annotation_token == "ann-1"
    assert annotation.sample_token == "sample-token"
    assert annotation.scene_token == "scene-token"
    assert annotation.camera_sample_data_token == "camera-token"
    assert annotation.image_path == "samples/CAM_FRONT/sample.jpg"
    assert annotation.original_category == "vehicle.car"
    assert annotation.detection_category == "car"
    assert annotation.class_id == 1
    assert annotation.visibility_level == "v80-100"
    assert annotation.conversion_status == "accepted"
    assert result.report["total_source_annotations_considered"] == 1


def test_rejections_are_structured_for_geometry_category_and_missing_inputs(tmp_path):
    cases = [
        ({"annotations": [_source_annotation(center=(0, 0, -10))]}, "object_behind_camera"),
        ({"annotations": [_source_annotation(center=(20, 0, 10))]}, "outside_image"),
        ({"annotations": [_source_annotation(category="animal")]}, "unsupported_category"),
        ({"omit_image": True}, "missing_image"),
        ({"calibration": None}, "missing_calibration"),
        ({"ego_pose": None}, "missing_ego_pose"),
        ({"sample_data_token": None}, "missing_camera_sample_data"),
    ]
    for index, (overrides, expected_reason) in enumerate(cases):
        case_root = tmp_path / str(index)
        frame = _frame(case_root, **overrides)
        result = _convert(case_root, [frame])
        assert not result.accepted_annotations
        assert result.rejected_annotations[0].rejection_reason == expected_reason


def test_malformed_source_geometry_is_rejected_without_aborting_conversion(tmp_path):
    malformed = _source_annotation()
    malformed["translation"] = 10
    result = _convert(tmp_path, [_frame(tmp_path, [malformed])])

    assert not result.accepted_annotations
    assert result.rejected_annotations[0].rejection_reason == "invalid_annotation_geometry"


def test_custom_mapping_and_mapping_validation():
    mapping = CategoryMapping(
        version="custom-v2",
        classes=(DetectionClass(4, "auto", ("vehicle.car",)),),
    )
    assert mapping.by_source["vehicle.car"].class_id == 4
    assert mapping.to_dict()["classes"][0]["class_id"] == 4


def test_conversion_order_and_repeated_exports_are_deterministic(tmp_path):
    first = _frame(tmp_path, [_source_annotation("z")])
    second = _frame(tmp_path, [_source_annotation("a")])
    second["sample_token"] = "sample-a"
    second["scene_token"] = "scene-a"
    one = _convert(tmp_path, [first, second])
    two = _convert(tmp_path, [second, first])
    assert one.to_manifest() == two.to_manifest()

    output = tmp_path / "output"
    write_conversion(one, output)
    first_files = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    write_conversion(two, output)
    second_files = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    assert first_files == second_files
    assert json.loads((output / "annotations.json").read_text())["annotations"][0]["annotation_id"] == "a"


def test_report_sanity_counts_rejections_and_scene_counts(tmp_path):
    frames = [
        _frame(tmp_path, [_source_annotation("good")]),
        _frame(tmp_path, [_source_annotation("bad", center=(25, 0, 10))]),
    ]
    frames[1]["sample_token"] = "sample-2"
    result = _convert(tmp_path, frames)

    assert result.report["rejected_annotations_by_reason"] == {"outside_image": 1}
    assert result.report["rejected_annotations_by_reason_and_category"] == {
        "outside_image": {"car": 1}
    }
    assert result.report["sanity_checks"]["accepted_boxes_out_of_image_bounds"] == 0
    assert result.report["sanity_checks"]["accepted_zero_or_negative_area_boxes"] == 0
    assert result.report["per_scene_annotation_counts"]["scene-token"]["accepted_annotations"] == 1
    assert result.report["per_scene_annotation_counts"]["scene-token"]["rejected_annotations"] == 1


def test_report_breaks_rejections_down_by_reason_and_mapped_category(tmp_path):
    frame = _frame(
        tmp_path,
        [
            _source_annotation("outside", center=(25, 0, 10)),
            _source_annotation("unsupported", category="animal"),
        ],
    )
    result = _convert(tmp_path, [frame])

    assert result.report["rejected_annotations_by_reason_and_category"] == {
        "outside_image": {"car": 1},
        "unsupported_category": {"unmapped": 1},
    }


def test_csv_exports_and_visual_inspection_use_real_source_image(tmp_path):
    result = _convert(tmp_path, [_frame(tmp_path)])
    output = tmp_path / "vision"
    files = write_conversion(result, output)
    with files["annotations_csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["source_annotation_token"] == "ann-1"
    rendered = render_inspection(
        files["annotations_json"],
        tmp_path,
        output / "inspection",
        sample_token="sample-token",
    )
    assert len(rendered) == 1 and rendered[0].is_file()


def test_inspection_modes_select_empty_and_rejected_images(tmp_path):
    frame = _frame(
        tmp_path,
        [_source_annotation("out", center=(20, 0, 10))],
    )
    result = _convert(tmp_path, [frame])
    output = tmp_path / "vision"
    files = write_conversion(result, output)

    assert render_inspection(files["annotations_json"], tmp_path, output / "empty", mode="empty")
    assert render_inspection(files["annotations_json"], tmp_path, output / "rejected", mode="rejected")
