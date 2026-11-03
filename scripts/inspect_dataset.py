"""Check nuScenes mini metadata and referenced files without database writes."""

import json
import logging
import sys
from pathlib import Path

from scenarioforge.cli import load_settings
from scenarioforge.config.logging_config import configure_logging
from scenarioforge.datasets.nuscenes_adapter import NuScenesAdapter


def main() -> int:
    settings = load_settings()
    configure_logging(settings.log_level)
    missing = NuScenesAdapter.check_accessibility(
        settings.dataset_root, settings.dataset_version
    )
    if missing:
        for problem in missing:
            logging.error(problem)
        return 2

    adapter = NuScenesAdapter(settings.dataset_root, settings.dataset_version)
    scene_count = sample_count = sensor_data_count = accessible_sensor_files = 0
    missing_sensor_files = 0
    annotation_count = 0
    sensor_channels = sorted({sensor["channel"] for sensor in adapter.iter_sensors()})
    for scene in adapter.iter_scenes():
        scene_count += 1
        for sample in adapter.iter_samples(scene):
            sample_count += 1
            annotation_count += sum(1 for _ in adapter.iter_annotations(sample))
            for reference in adapter.iter_sensor_data(sample):
                sensor_data_count += 1
                if adapter.sensor_file_exists(Path(reference["file_path"])):
                    accessible_sensor_files += 1
                else:
                    missing_sensor_files += 1

    print(
        json.dumps(
            {
                "dataset": adapter.dataset_record,
                "scenes": scene_count,
                "samples": sample_count,
                "sensor_channels": sensor_channels,
                "keyframe_sensor_data_references": sensor_data_count,
                "accessible_sensor_files": accessible_sensor_files,
                "missing_sensor_files": missing_sensor_files,
                "sweeps_included": False,
                "annotations": annotation_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
