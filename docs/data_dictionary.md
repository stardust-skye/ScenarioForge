# Data dictionary

All source identifiers are stored as strings and scoped by dataset. Database primary keys are internal integer IDs. Created/updated audit fields are UTC datetimes; source event timestamps are integer Unix microseconds.

| Entity | Important fields | Relationships and meaning |
|---|---|---|
| `datasets` | `source_id`, `name`, `version`, `root_path` | One row per registered source release. The nuScenes mini source ID is `nuscenes:v1.0-mini`. |
| `scenes` | `dataset_id`, `source_id`, `name`, `description`, `log_source_id`, `declared_sample_count` | A dataset owns scenes; a scene owns samples. `declared_sample_count` is the count supplied by source metadata. |
| `samples` | `dataset_id`, `scene_id`, `source_id`, `timestamp`, `previous_source_id`, `next_source_id` | A sample belongs to one scene and has timestamped sensor references and annotations. Link tokens remain source IDs for traceability. |
| `sensors` | `dataset_id`, `source_id`, `channel`, `modality`, `calibration` | A dataset-scoped calibrated sensor. `calibration` preserves available translation, rotation, camera intrinsics, and the underlying sensor token as JSON. |
| `sensor_data` | `dataset_id`, `sample_id`, `sensor_id`, `sensor_source_id`, `source_id`, `timestamp`, `file_path`, `is_key_frame`, `file_exists`, `metadata_json` | A sample's keyframe reference to a file, with dimensions, channel, and source pose/link metadata where available. nuScenes sweep chains are not ingested independently. `sensor_id` is nullable so a dangling source calibration reference can still be retained and reported. File bytes are not stored. |
| `annotations` | `dataset_id`, `sample_id`, `source_id`, `category`, `instance_source_id`, `attributes` | A sample annotation and its category; available box geometry, visibility, point counts, and attribute tokens remain in JSON. |
| `ingestion_issues` | `dataset_id`, `entity_type`, `source_id`, `code`, `severity`, `message`, `is_resolved` | A stable, dataset-scoped record of validation/accessibility problems. Issues are marked resolved when absent from a later completed ingestion pass. |

## Identifier conventions

- `source_id` is the stable identifier from the upstream dataset (nuScenes token), except the dataset record uses `nuscenes:<version>`.
- Relational `id` columns are internal database keys and must not be used to identify source records externally.
- Dataset-scoped unique constraints permit the same token format in different datasets without collisions.
- `sensor_data.sensor_source_id` preserves the upstream calibrated-sensor token even if no matching `sensors` row is available.
- For missing identifiers in an ingestion issue, `source_id` uses a deterministic `<missing-id:entity:occurrence>` marker so separate records are not collapsed.

## Timestamp conventions

- `samples.timestamp` and `sensor_data.timestamp` are source Unix timestamps in microseconds, as represented by nuScenes.
- Values must be non-negative. Invalid values are reported as ingestion issues and the invalid record is not persisted.
- Report timestamp coverage uses the minimum and maximum sample timestamps. Scene durations use each scene's maximum minus minimum sample timestamp and are reported in seconds. No conversion to local time is applied.

## Indexes and constraints

Each entity has a primary key and dataset-scoped source-ID uniqueness where applicable. Foreign keys preserve dataset/scene/sample relationships; repository methods additionally verify that parent rows belong to the same dataset. Indexes support scene/sample chronological lookup, per-sample sensor and annotation queries, channel lookup, category counts, and issue review.
