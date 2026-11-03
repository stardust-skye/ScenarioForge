"""nuScenes mini adapter backed by the official nuScenes devkit."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from scenarioforge.datasets.base import DatasetAdapter


class NuScenesAdapter(DatasetAdapter):
    """Read nuScenes metadata and references without copying sensor payloads."""

    DATASET_NAME = "nuscenes"
    REQUIRED_METADATA_TABLES = (
        "attribute",
        "calibrated_sensor",
        "category",
        "ego_pose",
        "instance",
        "log",
        "map",
        "sample",
        "sample_annotation",
        "sample_data",
        "scene",
        "sensor",
        "visibility",
    )

    def __init__(
        self,
        dataroot: Path,
        version: str = "v1.0-mini",
        *,
        nuscenes_instance: Any | None = None,
    ) -> None:
        self.dataroot = dataroot.expanduser().resolve()
        self.version = version
        if nuscenes_instance is not None:
            self._nusc = nuscenes_instance
            return
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as exc:
            raise RuntimeError(
                "nuScenes devkit is missing. Install ScenarioForge dependencies with "
                "`python -m pip install -e .`."
            ) from exc
        self._nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=False)

    @property
    def dataset_record(self) -> dict[str, Any]:
        return {
            "source_id": f"{self.DATASET_NAME}:{self.version}",
            "name": self.DATASET_NAME,
            "version": self.version,
            "root_path": str(self.dataroot),
        }

    def validate_accessibility(self) -> list[str]:
        return self.check_accessibility(self.dataroot, self.version)

    @property
    def visibility_levels(self) -> dict[str, str]:
        """Return source visibility tokens mapped to their nuScenes level labels."""
        return {row["token"]: row["level"] for row in self._nusc.visibility}

    @staticmethod
    def check_accessibility(dataroot: Path, version: str) -> list[str]:
        """Check required metadata paths without initializing the devkit."""
        dataroot = dataroot.expanduser().resolve()
        problems: list[str] = []
        if not dataroot.is_dir():
            return [f"Dataset root does not exist or is not a directory: {dataroot}"]
        metadata_dir = dataroot / version
        if not metadata_dir.is_dir():
            problems.append(f"Metadata directory is missing: {metadata_dir}")
        else:
            for table in NuScenesAdapter.REQUIRED_METADATA_TABLES:
                metadata_file = metadata_dir / f"{table}.json"
                if not metadata_file.is_file():
                    problems.append(f"Required metadata file is missing: {metadata_file}")
        return problems

    def iter_scenes(self) -> Iterator[dict[str, Any]]:
        for scene in self._nusc.scene:
            yield {
                "source_id": scene["token"],
                "name": scene["name"],
                "description": scene.get("description", ""),
                "log_source_id": scene.get("log_token"),
                "declared_sample_count": scene["nbr_samples"],
                "first_sample_source_id": scene["first_sample_token"],
            }

    def iter_samples(self, scene: dict[str, Any]) -> Iterator[dict[str, Any]]:
        token = scene["first_sample_source_id"]
        seen: set[str] = set()
        while token:
            if token in seen:
                raise ValueError(f"Cycle detected in sample links for scene {scene['source_id']}")
            seen.add(token)
            sample = self._nusc.get("sample", token)
            if sample["scene_token"] != scene["source_id"]:
                raise ValueError(
                    f"Sample {token} points to scene {sample['scene_token']}, "
                    f"not {scene['source_id']}"
                )
            yield {
                "source_id": sample["token"],
                "scene_source_id": sample["scene_token"],
                "timestamp": sample["timestamp"],
                "previous_source_id": sample.get("prev") or None,
                "next_source_id": sample.get("next") or None,
                "data": sample["data"],
                "annotations": sample["anns"],
            }
            token = sample.get("next") or ""

    def iter_sensors(self) -> Iterator[dict[str, Any]]:
        sensors = {sensor["token"]: sensor for sensor in self._nusc.sensor}
        for calibrated in self._nusc.calibrated_sensor:
            sensor = sensors.get(calibrated["sensor_token"])
            if sensor is None:
                raise ValueError(
                    f"Calibration {calibrated['token']} references missing sensor "
                    f"{calibrated['sensor_token']}"
                )
            yield {
                "source_id": calibrated["token"],
                "channel": sensor["channel"],
                "modality": sensor["modality"],
                "calibration": {
                    "sensor_source_id": sensor["token"],
                    "translation": calibrated.get("translation"),
                    "rotation": calibrated.get("rotation"),
                    "camera_intrinsic": calibrated.get("camera_intrinsic"),
                },
            }

    def iter_sensor_data(self, sample: dict[str, Any]) -> Iterator[dict[str, Any]]:
        for channel, sample_data_token in sample["data"].items():
            item = self._nusc.get("sample_data", sample_data_token)
            calibrated_source_id = item["calibrated_sensor_token"]
            path = Path(self._nusc.get_sample_data_path(sample_data_token)).resolve()
            yield {
                "source_id": item["token"],
                "sample_source_id": sample["source_id"],
                "sensor_source_id": calibrated_source_id,
                "timestamp": item["timestamp"],
                "file_path": str(path),
                "is_key_frame": item["is_key_frame"],
                "metadata": {
                    "channel": channel,
                    "height": item.get("height"),
                    "width": item.get("width"),
                    "ego_pose_source_id": item.get("ego_pose_token"),
                    "previous_source_id": item.get("prev") or None,
                    "next_source_id": item.get("next") or None,
                },
            }

    def iter_annotations(self, sample: dict[str, Any]) -> Iterator[dict[str, Any]]:
        for annotation_token in sample["annotations"]:
            annotation = self._nusc.get("sample_annotation", annotation_token)
            instance = self._nusc.get("instance", annotation["instance_token"])
            category = self._nusc.get("category", instance["category_token"])
            yield {
                "source_id": annotation["token"],
                "sample_source_id": sample["source_id"],
                "category": category["name"],
                "instance_source_id": annotation.get("instance_token"),
                "attributes": {
                    "translation": annotation.get("translation"),
                    "size": annotation.get("size"),
                    "rotation": annotation.get("rotation"),
                    "visibility_token": annotation.get("visibility_token"),
                    "num_lidar_pts": annotation.get("num_lidar_pts"),
                    "num_radar_pts": annotation.get("num_radar_pts"),
                    "attribute_tokens": annotation.get("attribute_tokens", []),
                },
            }

    def iter_camera_samples(self, channel: str = "CAM_FRONT") -> Iterator[dict[str, Any]]:
        """Yield one camera frame and its source annotations per sample, including missing links."""
        scenes = {scene["token"]: scene for scene in self._nusc.scene}
        samples = sorted(
            self._nusc.sample,
            key=lambda row: (row["scene_token"], row["timestamp"], row["token"]),
        )
        for sample in samples:
            sample_data_token = sample.get("data", {}).get(channel)
            sample_data = None
            image_path = None
            calibration = None
            ego_pose = None
            camera_error = None
            if sample_data_token:
                try:
                    sample_data = self._nusc.get("sample_data", sample_data_token)
                except KeyError as exc:
                    camera_error = f"missing_camera_sample_data:{exc}"
                if sample_data is not None:
                    try:
                        calibration = self._nusc.get(
                            "calibrated_sensor", sample_data["calibrated_sensor_token"]
                        )
                    except KeyError as exc:
                        camera_error = f"missing_calibration:{exc}"
                    try:
                        ego_pose = self._nusc.get("ego_pose", sample_data["ego_pose_token"])
                    except KeyError as exc:
                        camera_error = f"missing_ego_pose:{exc}"
                    try:
                        image_path = Path(
                            self._nusc.get_sample_data_path(sample_data_token)
                        ).resolve()
                    except (KeyError, FileNotFoundError) as exc:
                        camera_error = f"missing_image:{exc}"

            annotations: list[dict[str, Any]] = []
            for annotation_token in sample.get("anns", []):
                try:
                    annotation = self._nusc.get("sample_annotation", annotation_token)
                    instance = self._nusc.get("instance", annotation["instance_token"])
                    category = self._nusc.get("category", instance["category_token"])
                    annotations.append({
                        "source_annotation_token": annotation_token,
                        "instance_token": annotation.get("instance_token"),
                        "category": category.get("name"),
                        "translation": annotation.get("translation"),
                        "size": annotation.get("size"),
                        "rotation": annotation.get("rotation"),
                        "visibility_token": annotation.get("visibility_token"),
                    })
                except KeyError as exc:
                    annotations.append({
                        "source_annotation_token": annotation_token,
                        "category": None,
                        "source_error": str(exc),
                    })

            scene = scenes.get(sample["scene_token"])
            yield {
                "sample_token": sample["token"],
                "scene_token": sample["scene_token"],
                "scene_name": scene.get("name") if scene else None,
                "sample_data_token": sample_data_token,
                "image_path": image_path,
                "width": sample_data.get("width") if sample_data else None,
                "height": sample_data.get("height") if sample_data else None,
                "calibration": calibration,
                "ego_pose": ego_pose,
                "camera_error": camera_error,
                "annotations": annotations,
            }

    def sensor_file_exists(self, file_path: Path) -> bool:
        return file_path.is_file()
