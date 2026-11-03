import json
from pathlib import Path
from types import SimpleNamespace

import scripts.ingest_dataset as ingest_cli
from scenarioforge.ingestion.loader import IngestionSummary


def test_ingestion_cli_returns_nonzero_for_fatal_summary(monkeypatch, capsys) -> None:
    settings = SimpleNamespace(
        dataset_root=Path("dataset"),
        dataset_version="v1.0-mini",
        log_level="INFO",
    )
    engine = SimpleNamespace(disposed=False)
    engine.dispose = lambda: setattr(engine, "disposed", True)

    class FakeAdapter:
        @staticmethod
        def check_accessibility(root: Path, version: str) -> list[str]:
            assert root == settings.dataset_root
            assert version == settings.dataset_version
            return []

        def __init__(self, root: Path, version: str) -> None:
            assert root == settings.dataset_root
            assert version == settings.dataset_version

    summary = IngestionSummary(
        dataset_source_id="nuscenes:v1.0-mini",
        status="failed",
        fatal_error="RuntimeError: traversal failed",
    )
    monkeypatch.setattr(ingest_cli, "load_settings", lambda: settings)
    monkeypatch.setattr(ingest_cli, "configure_logging", lambda level: None)
    monkeypatch.setattr(ingest_cli, "create_database_engine", lambda configured: engine)
    monkeypatch.setattr(ingest_cli, "initialize_database", lambda configured: None)
    monkeypatch.setattr(ingest_cli, "create_session_factory", lambda configured: object())
    monkeypatch.setattr(ingest_cli, "NuScenesAdapter", FakeAdapter)
    monkeypatch.setattr(ingest_cli, "ingest_dataset", lambda adapter, factory: summary)

    assert ingest_cli.main() == 1
    assert engine.disposed is True
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
