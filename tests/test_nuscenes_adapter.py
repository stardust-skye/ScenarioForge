from pathlib import Path

from scenarioforge.datasets.nuscenes_adapter import NuScenesAdapter


class FakeNuScenes:
    scene = [
        {
            "token": "scene-token",
            "name": "scene-001",
            "description": "fixture",
            "log_token": "log-token",
            "nbr_samples": 1,
            "first_sample_token": "sample-token",
        }
    ]
    sensor = [{"token": "sensor-token", "channel": "CAM_FRONT", "modality": "camera"}]
    calibrated_sensor = [
        {
            "token": "calibrated-token",
            "sensor_token": "sensor-token",
            "translation": [0, 0, 0],
            "rotation": [1, 0, 0, 0],
            "camera_intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }
    ]

    def get(self, table: str, token: str):
        records = {
            ("sample", "sample-token"): {
                "token": "sample-token",
                "scene_token": "scene-token",
                "timestamp": 1_000_000,
                "prev": "",
                "next": "",
                "data": {"CAM_FRONT": "sample-data-token"},
                "anns": ["annotation-token"],
            },
            ("sample_data", "sample-data-token"): {
                "token": "sample-data-token",
                "calibrated_sensor_token": "calibrated-token",
                "timestamp": 1_000_000,
                "is_key_frame": True,
                "height": 900,
                "width": 1600,
                "ego_pose_token": "pose-token",
                "prev": "",
                "next": "",
            },
            ("sample_annotation", "annotation-token"): {
                "token": "annotation-token",
                "instance_token": "instance-token",
                "translation": [1, 2, 3],
                "size": [2, 2, 4],
                "rotation": [1, 0, 0, 0],
                "visibility_token": "visibility-token",
                "num_lidar_pts": 0,
                "num_radar_pts": 0,
                "attribute_tokens": [],
            },
            ("instance", "instance-token"): {
                "token": "instance-token",
                "category_token": "category-token",
            },
            ("category", "category-token"): {"name": "vehicle.car"},
        }
        return records[(table, token)]

    def get_sample_data_path(self, token: str) -> str:
        return f"C:/fixture/{token}.jpg"


def test_adapter_normalizes_scene_sensor_and_annotation_records(tmp_path: Path) -> None:
    adapter = NuScenesAdapter(tmp_path, nuscenes_instance=FakeNuScenes())

    scene = next(adapter.iter_scenes())
    sample = next(adapter.iter_samples(scene))
    sensor = next(adapter.iter_sensors())
    sensor_data = next(adapter.iter_sensor_data(sample))
    annotation = next(adapter.iter_annotations(sample))

    assert scene["source_id"] == "scene-token"
    assert sample["timestamp"] == 1_000_000
    assert sensor["calibration"]["camera_intrinsic"][0] == [1, 0, 0]
    assert sensor_data["sensor_source_id"] == "calibrated-token"
    assert sensor_data["file_path"].endswith("sample-data-token.jpg")
    assert annotation["category"] == "vehicle.car"


def test_metadata_preflight_reports_missing_root(tmp_path: Path) -> None:
    problems = NuScenesAdapter.check_accessibility(tmp_path / "not-present", "v1.0-mini")

    assert len(problems) == 1
    assert "does not exist" in problems[0]
