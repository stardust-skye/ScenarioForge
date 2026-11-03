import json
import logging

import pytest

from scenarioforge.config.logging_config import StructuredFormatter, configure_logging


def test_structured_formatter_includes_issue_context() -> None:
    record = logging.LogRecord(
        name="scenarioforge.ingestion",
        level=logging.WARNING,
        pathname=__file__,
        lineno=12,
        msg="invalid timestamp",
        args=(),
        exc_info=None,
    )
    record.issue_code = "invalid_timestamp"

    formatted = json.loads(StructuredFormatter().format(record))

    assert formatted["level"] == "WARNING"
    assert formatted["message"] == "invalid timestamp"
    assert formatted["issue_code"] == "invalid_timestamp"


def test_configure_logging_rejects_unknown_levels() -> None:
    with pytest.raises(ValueError, match="Unknown logging level"):
        configure_logging("not-a-level")
