"""SQLAlchemy engine, session, and schema initialization."""

from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from scenarioforge.config.settings import Settings
from scenarioforge.database.models import Base


def create_database_engine(settings: Settings) -> Engine:
    """Create an engine and ensure local SQLite database directories exist."""
    url = make_url(settings.database_url)
    connect_args: dict[str, object] = {}
    if url.drivername.startswith("sqlite"):
        database = url.database
        if database not in (None, "", ":memory:"):
            database_path = Path(database)
            if not database_path.is_absolute():
                database_path = settings.project_root / database_path
            database_path.parent.mkdir(parents=True, exist_ok=True)
            url = url.set(database=str(database_path))
        connect_args["check_same_thread"] = False

    engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
    if url.drivername.startswith("sqlite"):
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine


def _enable_sqlite_foreign_keys(connection: Any, record: Any) -> None:
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a typed SQLAlchemy session factory."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def initialize_database(engine: Engine) -> None:
    """Create missing tables without dropping or rewriting existing data."""
    Base.metadata.create_all(engine)
