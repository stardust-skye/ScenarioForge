"""Initialize the metadata schema and ingest nuScenes mini metadata."""

import json
import logging
import sys

from scenarioforge.cli import load_settings
from scenarioforge.config.logging_config import configure_logging
from scenarioforge.database.connection import (
    create_database_engine,
    create_session_factory,
    initialize_database,
)
from scenarioforge.datasets.nuscenes_adapter import NuScenesAdapter
from scenarioforge.ingestion.loader import ingest_dataset


def main() -> int:
    settings = load_settings()
    configure_logging(settings.log_level)
    problems = NuScenesAdapter.check_accessibility(
        settings.dataset_root, settings.dataset_version
    )
    if problems:
        for problem in problems:
            logging.error(problem)
        return 2

    engine = create_database_engine(settings)
    try:
        initialize_database(engine)
        adapter = NuScenesAdapter(settings.dataset_root, settings.dataset_version)
        summary = ingest_dataset(adapter, create_session_factory(engine))
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 1 if summary.status == "failed" else 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
