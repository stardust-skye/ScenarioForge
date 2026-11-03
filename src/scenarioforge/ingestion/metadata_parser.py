"""Convert adapter dictionaries into validated, dataset-independent records."""

from typing import Any, TypeVar

from pydantic import BaseModel

from scenarioforge.ingestion.validators import validate_record
from scenarioforge.schemas.dataset import SampleSchema, SceneSchema

SchemaType = TypeVar("SchemaType", bound=BaseModel)

_TRAVERSAL_FIELDS: dict[type[BaseModel], frozenset[str]] = {
    SceneSchema: frozenset({"first_sample_source_id"}),
    SampleSchema: frozenset({"data", "annotations"}),
}


def parse_record(schema: type[SchemaType], record: dict[str, Any]) -> SchemaType:
    """Validate persisted fields and reject unknown adapter data.

    Scene/sample adapters carry a small set of fields required only for
    traversal. Those fields are explicitly allowed, then excluded from the
    persisted model. All other unknown keys reach Pydantic's ``extra=forbid``.
    """
    traversal_fields = _TRAVERSAL_FIELDS.get(schema, frozenset())
    return validate_record(
        schema,
        {key: value for key, value in record.items() if key not in traversal_fields},
    )
