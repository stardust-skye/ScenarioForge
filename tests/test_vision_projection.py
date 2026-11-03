from __future__ import annotations

import math

import numpy as np
import pytest
from pyquaternion import Quaternion

from scenarioforge.vision.config import ProjectionConfig
from scenarioforge.vision.projection import (
    global_box_to_camera,
    project_annotation,
    project_box_to_image,
    transform_points,
)
from scenarioforge.vision.schemas import SourceAnnotation

INTRINSIC = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _annotation(
    center=(0.0, 0.0, 10.0),
    dimensions=(2.0, 2.0, 2.0),
    rotation=IDENTITY,
    token="ann-1",
):
    return SourceAnnotation(
        source_annotation_token=token,
        category="vehicle.car",
        translation=center,
        size=dimensions,
        rotation=rotation,
        visibility_token="4",
        visibility_level="v80-100",
    )


def _pose(translation=(0.0, 0.0, 0.0), rotation=IDENTITY):
    return {"translation": translation, "rotation": rotation}


def _calibration(translation=(0.0, 0.0, 0.0), rotation=IDENTITY):
    return {
        "translation": translation,
        "rotation": rotation,
        "camera_intrinsic": INTRINSIC.tolist(),
    }


def test_transform_points_identity_translation_and_inverse():
    points = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    assert np.array_equal(transform_points(points, (0, 0, 0), IDENTITY), points)
    translated = transform_points(points[:, :1], (1, 2, 3), IDENTITY)
    assert np.array_equal(translated[:, 0], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(
        transform_points(translated, (1, 2, 3), IDENTITY, inverse=True), points[:, :1]
    )


def test_transform_points_applies_quaternion_rotation():
    quarter_turn = Quaternion(axis=(0, 0, 1), angle=math.pi / 2)
    rotation = (quarter_turn.elements).tolist()
    transformed = transform_points(np.array([[1.0], [0.0], [0.0]]), (0, 0, 0), rotation)

    assert transformed[:, 0] == pytest.approx((0.0, 1.0, 0.0))


def test_global_box_transform_uses_global_to_ego_to_camera_order():
    annotation = _annotation(center=(11.0, 0.0, 10.0))
    camera_box = global_box_to_camera(
        annotation,
        _pose(translation=(10.0, 0.0, 0.0)),
        _calibration(),
    )

    assert camera_box.center == pytest.approx((1.0, 0.0, 10.0))


def test_project_box_uses_camera_intrinsics_and_bounds():
    projection, reason = project_annotation(
        _annotation(), _pose(), _calibration(), width=100, height=100
    )

    assert reason is None
    assert projection is not None
    assert (projection.x_min, projection.y_min, projection.x_max, projection.y_max) == pytest.approx(
        (38.8888889, 38.8888889, 61.1111111, 61.1111111)
    )
    assert projection.projected_point_count == 8
    assert not projection.is_clipped


def test_rotation_changes_projected_extent():
    rotation = Quaternion(axis=(0, 0, 1), angle=math.pi / 4).elements
    box = global_box_to_camera(_annotation(rotation=rotation), _pose(), _calibration())
    projection, reason = project_box_to_image(box.corners(), INTRINSIC, 100, 100)

    assert reason is None
    assert projection is not None
    assert projection.x_max - projection.x_min > 20


def test_behind_camera_box_is_rejected_without_projection():
    projection, reason = project_annotation(
        _annotation(center=(0.0, 0.0, -10.0)), _pose(), _calibration(), 100, 100
    )

    assert projection is None
    assert reason == "object_behind_camera"


def test_near_plane_crossing_is_clipped_before_projection():
    projection, reason = project_annotation(
        _annotation(center=(0.0, 0.0, 0.15), dimensions=(0.02, 0.02, 0.2)),
        _pose(),
        _calibration(),
        100,
        100,
        ProjectionConfig(max_box_area_fraction=1.0),
    )

    assert reason is None
    assert projection is not None
    assert projection.projected_point_count >= 3
    assert projection.x_min >= 0 and projection.y_min >= 0


def test_fully_outside_image_and_area_threshold_are_rejected():
    outside, outside_reason = project_annotation(
        _annotation(center=(20.0, 0.0, 10.0)), _pose(), _calibration(), 100, 100
    )
    too_small, too_small_reason = project_annotation(
        _annotation(),
        _pose(),
        _calibration(),
        100,
        100,
        ProjectionConfig(min_box_area_px=500.0),
    )

    assert outside is None and outside_reason == "outside_image"
    assert too_small is None and too_small_reason == "degenerate_box"


def test_partially_outside_image_is_clipped():
    projection, reason = project_annotation(
        _annotation(center=(4.8, 0.0, 10.0)), _pose(), _calibration(), 100, 100
    )

    assert reason is None
    assert projection is not None
    assert projection.is_clipped
    assert projection.x_max == 100
    assert projection.x_min < projection.x_max


@pytest.mark.parametrize(
    "points,translation,rotation",
    [
        (np.zeros((2, 2)), (0, 0, 0), IDENTITY),
        (np.zeros((3, 2)), (0, 0), IDENTITY),
        (np.array([[float("nan")], [0], [0]]), (0, 0, 0), IDENTITY),
    ],
)
def test_invalid_transform_inputs_raise(points, translation, rotation):
    with pytest.raises(ValueError):
        transform_points(points, translation, rotation)
