"""nuScenes box transforms and camera-plane projection using the official devkit geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Sequence

import numpy as np
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from scenarioforge.vision.config import ProjectionConfig
from scenarioforge.vision.schemas import SourceAnnotation

BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


@dataclass(frozen=True)
class BoxProjection:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    is_clipped: bool
    projected_point_count: int


def transform_points(
    points: np.ndarray,
    translation: Sequence[float],
    rotation: Sequence[float],
    *,
    inverse: bool = False,
) -> np.ndarray:
    """Transform 3xN points with a nuScenes (w,x,y,z) quaternion and translation."""
    points = np.asarray(points, dtype=float)
    offset = np.asarray(translation, dtype=float)
    if points.ndim != 2 or points.shape[0] != 3:
        raise ValueError("points must have shape (3, N)")
    if offset.shape != (3,) or len(rotation) != 4:
        raise ValueError("translation and rotation must have lengths 3 and 4")
    if not np.isfinite(points).all() or not np.isfinite(offset).all():
        raise ValueError("transform inputs must be finite")
    quaternion = Quaternion(rotation)
    if inverse:
        return quaternion.rotation_matrix.T @ (points - offset[:, None])
    return quaternion.rotation_matrix @ points + offset[:, None]


def global_box_to_camera(
    annotation: SourceAnnotation,
    ego_pose: dict[str, Any],
    calibration: dict[str, Any],
) -> Box:
    """Create the official nuScenes Box and apply global->ego->camera inverse transforms."""
    if annotation.translation is None or annotation.size is None or annotation.rotation is None:
        raise ValueError("annotation geometry is incomplete")
    center = np.asarray(annotation.translation, dtype=float)
    size = np.asarray(annotation.size, dtype=float)
    orientation = np.asarray(annotation.rotation, dtype=float)
    ego_translation = np.asarray(ego_pose["translation"], dtype=float)
    ego_rotation = ego_pose["rotation"]
    camera_translation = np.asarray(calibration["translation"], dtype=float)
    camera_rotation = calibration["rotation"]
    intrinsic = np.asarray(calibration["camera_intrinsic"], dtype=float)
    arrays = (center, size, orientation, ego_translation, camera_translation, intrinsic)
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("annotation, pose, or calibration contains non-finite values")
    if center.shape != (3,) or size.shape != (3,) or np.any(size <= 0):
        raise ValueError("annotation center and positive dimensions must have length 3")
    if orientation.shape != (4,):
        raise ValueError("annotation rotation must have length 4")
    if ego_translation.shape != (3,) or camera_translation.shape != (3,):
        raise ValueError("pose and calibration translation must have length 3")
    if intrinsic.shape != (3, 3):
        raise ValueError("camera intrinsic matrix must have shape (3, 3)")

    box = Box(
        center=center.tolist(),
        size=size.tolist(),
        orientation=Quaternion(orientation),
        name=annotation.category,
        token=annotation.source_annotation_token,
    )
    box.translate(-ego_translation)
    box.rotate(Quaternion(ego_rotation).inverse)
    box.translate(-camera_translation)
    box.rotate(Quaternion(camera_rotation).inverse)
    return box


def project_box_to_image(
    corners_camera: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    config: ProjectionConfig | None = None,
) -> tuple[BoxProjection | None, str | None]:
    """Clip cuboid edges at the near plane, project with devkit `view_points`, and validate."""
    config = config or ProjectionConfig()
    corners = np.asarray(corners_camera, dtype=float)
    camera_matrix = np.asarray(intrinsic, dtype=float)
    if width <= 0 or height <= 0:
        return None, "invalid_image_dimensions"
    if corners.shape != (3, 8) or camera_matrix.shape != (3, 3):
        return None, "invalid_projection_geometry"
    if not np.isfinite(corners).all() or not np.isfinite(camera_matrix).all():
        return None, "invalid_projection_geometry"
    if np.linalg.matrix_rank(camera_matrix) < 3:
        return None, "invalid_calibration"

    clipped_points: list[np.ndarray] = [
        corners[:, index].copy()
        for index in range(corners.shape[1])
        if corners[2, index] >= config.near_plane_m
    ]
    for start, end in BOX_EDGES:
        first, second = corners[:, start], corners[:, end]
        depth_first, depth_second = first[2], second[2]
        if (depth_first < config.near_plane_m <= depth_second) or (
            depth_second < config.near_plane_m <= depth_first
        ):
            fraction = (config.near_plane_m - depth_first) / (depth_second - depth_first)
            intersection = first + fraction * (second - first)
            intersection[2] = config.near_plane_m
            clipped_points.append(intersection)

    if not clipped_points:
        return None, "object_behind_camera"
    unique_points: list[np.ndarray] = []
    for point in clipped_points:
        if not any(np.allclose(point, existing, rtol=0.0, atol=1e-9) for existing in unique_points):
            unique_points.append(point)
    if len(unique_points) < config.min_projected_points:
        return None, "insufficient_projected_geometry"

    points_camera = np.column_stack(unique_points)
    depths = points_camera[2]
    if np.any(depths < config.near_plane_m) or np.any(depths <= 0):
        return None, "invalid_projection_depth"
    projected = view_points(points_camera, camera_matrix, normalize=True)
    if not np.isfinite(projected).all():
        return None, "invalid_projected_coordinates"

    raw_x_min, raw_y_min = float(projected[0].min()), float(projected[1].min())
    raw_x_max, raw_y_max = float(projected[0].max()), float(projected[1].max())
    x_min = min(max(raw_x_min, 0.0), float(width))
    y_min = min(max(raw_y_min, 0.0), float(height))
    x_max = min(max(raw_x_max, 0.0), float(width))
    y_max = min(max(raw_y_max, 0.0), float(height))
    if x_max <= x_min or y_max <= y_min:
        return None, "outside_image"

    area = (x_max - x_min) * (y_max - y_min)
    if not isfinite(area) or area < config.min_box_area_px:
        return None, "degenerate_box"
    if area / (width * height) > config.max_box_area_fraction:
        return None, "box_too_large"
    return (
        BoxProjection(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            is_clipped=(
                raw_x_min < 0 or raw_y_min < 0 or raw_x_max > width or raw_y_max > height
            ),
            projected_point_count=len(unique_points),
        ),
        None,
    )


def project_annotation(
    annotation: SourceAnnotation,
    ego_pose: dict[str, Any],
    calibration: dict[str, Any],
    width: int,
    height: int,
    config: ProjectionConfig | None = None,
) -> tuple[BoxProjection | None, str | None]:
    """Transform one global annotation box into camera coordinates and project it."""
    config = config or ProjectionConfig()
    try:
        box = global_box_to_camera(annotation, ego_pose, calibration)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return None, "invalid_annotation_geometry" if annotation.source_error is None else "invalid_source_annotation"
    return project_box_to_image(
        box.corners(),
        np.asarray(calibration["camera_intrinsic"], dtype=float),
        width,
        height,
        config,
    )
