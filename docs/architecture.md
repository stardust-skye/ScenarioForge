# Architecture

## Components and responsibilities

- `config`: reads dataset, database, and logging settings from the environment.
- `datasets.base`: defines the dataset-independent adapter contract.
- `datasets.nuscenes_adapter`: uses the official devkit and translates nuScenes records into stable source IDs, sensor references, timestamps, calibration, and annotations. It does not copy sensor payloads.
- `schemas` and `ingestion.metadata_parser`: validate normalized records at the adapter boundary. Adapter-only traversal values are not persisted.
- `ingestion.loader`: traverses records, detects duplicate IDs and invalid/cross-record references, upserts valid rows, and persists issues rather than silently dropping problematic records.
- `database`: provides the SQLAlchemy schema, connection setup, and repository upserts. SQLite is the local default; a SQLAlchemy PostgreSQL URL can be configured.
- `analytics.basic_statistics`: queries persisted rows and writes JSON plus flattened CSV.
- `scripts`: expose inspect, ingest, and report workflows without requiring cloud services.

## Data flow

1. `inspect_dataset.py` validates the configured root and metadata tables, then counts scenes, samples, keyframe sensor references, accessible files, and missing sensor files without changing the database. It does not traverse sweep chains.
2. `ingest_dataset.py` initializes missing tables, opens the configured adapter, and validates data records as the adapter yields them.
3. Valid records are inserted or updated by `(dataset_id, source_id)`. Sensor-data file references remain paths; file absence is recorded both on the reference and in the issue table. Invalid records and dangling calibrated-sensor references are also written as issues.
4. The ingestion transaction rolls back on fatal traversal or persistence errors. Existing issues are only committed as resolved when the complete pass commits; issues observed again during that pass are active.
5. `report_statistics.py` reads those records and writes JSON and CSV summaries, including sensor channels and modalities. No statistics are fabricated for an empty database.

Each dataset ingestion is one SQL transaction. Record-level validation/reference problems are recoverable and recorded; iterator/traversal or database failures mark the summary `failed`, roll back the metadata transaction, and cause the CLI to return nonzero. A completed pass with recoverable issues is reported as `completed_with_issues`.

## Database relationships

- One `Dataset` owns many `Scene`, `Sensor`, and `IngestionIssue` records.
- One `Scene` owns many `Sample` records.
- One `Sample` owns many `SensorData` and `Annotation` records.
- One calibrated `Sensor` can have many `SensorData` references. If calibration metadata is missing, `SensorData` remains stored with its source sensor ID and a null relational sensor ID, plus an active issue.

Source identifiers are unique within each dataset. Sensor files are never stored as database blobs. See [data_dictionary.md](data_dictionary.md) for field and time conventions.

## Adapter design

Implementations of `DatasetAdapter` provide dataset identity, metadata accessibility checks, iterators for scenes, samples, calibrated sensors, sensor data, and annotations, and a sensor-file existence check. The nuScenes adapter emits keyframe references from `sample.data`; sweep `prev`/`next` chains are intentionally outside Part 1.6. The ingestion and database layers do not import nuScenes types. This keeps future adapters independent of nuScenes record formats.

The initial adapter is configured for `v1.0-mini`. The devkit reads metadata from the user-provided local root; ScenarioForge does not initiate downloads. The adapter exposes nuScenes tokens as source identifiers and preserves paths returned by the devkit.

## Engineering decisions and trade-offs

- **SQLite first, SQLAlchemy abstraction:** local setup works without a database service, while PostgreSQL can be selected with `DATABASE_URL`. This avoids requiring a second local engine/dialect for the current relational workload.
- **Normalized metadata, external sensor files:** relational queries work on metadata while large camera/LiDAR/radar assets stay in place.
- **Pydantic at ingestion boundaries:** malformed records are detected before reaching ORM rows and can be explained in the issue table.
- **Stable upserts instead of destructive reloads:** repeated ingestion updates matching source records and does not create duplicates. It does not currently delete database rows removed from a later version of the source dataset.
- **Repository ownership checks:** sample, sensor-data, and annotation writes verify that their referenced parent records belong to the supplied dataset.
- **Microsecond integer timestamps:** this matches nuScenes source precision and avoids lossy conversion; report duration values are converted to seconds.
- **Explicit scope:** no event mining, rare-scenario ranking, training, dashboard, or sensor-payload decoding is part of Part 1.
