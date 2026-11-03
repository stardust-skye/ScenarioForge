"""Consistent application logging configuration."""

import json
import logging
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """Render log records as JSON objects with stable core fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in (
            "event",
            "dataset_source_id",
            "entity_type",
            "source_id",
            "issue_code",
            "summary",
        ):
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure concise timestamped logs for command-line workflows."""
    resolved_level = getattr(logging, level.upper(), None)
    if not isinstance(resolved_level, int):
        raise ValueError(f"Unknown logging level: {level}")
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    logging.basicConfig(level=resolved_level, handlers=[handler])
