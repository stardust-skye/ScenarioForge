"""Record validation and cross-record checks."""

from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

SchemaType = TypeVar("SchemaType", bound=BaseModel)


def validate_record(schema: type[SchemaType], record: dict[str, Any]) -> SchemaType:
    """Validate a normalized adapter record with a useful model error."""
    return schema.model_validate(record)


def validate_sensor_path(path_value: str) -> tuple[Path, bool]:
    """Return a normalized file path and whether it currently points to a file."""
    path = Path(path_value).expanduser()
    return path, path.is_file()


def validation_error_message(error: ValidationError) -> str:
    """Render Pydantic errors without hiding the invalid fields."""
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors()
    )
