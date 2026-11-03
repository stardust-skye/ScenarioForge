"""Generate JSON and CSV statistics for an ingested dataset."""

import logging
import sys
from pathlib import Path

from scenarioforge.analytics.basic_statistics import export_statistics, generate_statistics
from scenarioforge.cli import load_settings
from scenarioforge.config.logging_config import configure_logging
from scenarioforge.database.connection import create_database_engine, create_session_factory


def main() -> int:
    settings = load_settings()
    configure_logging(settings.log_level)
    engine = create_database_engine(settings)
    try:
        factory = create_session_factory(engine)
        with factory() as session:
            report = generate_statistics(
                session, f"nuscenes:{settings.dataset_version}"
            )
        output_dir = settings.project_root / "data" / "processed"
        json_path = Path(output_dir / "dataset_statistics.json")
        csv_path = Path(output_dir / "dataset_statistics.csv")
        export_statistics(report, json_path, csv_path)
        logging.info("Wrote statistics to %s and %s", json_path, csv_path)
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
