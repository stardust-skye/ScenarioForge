"""Scene-grouped data splits and PyTorch-ready detection input."""

from scenarioforge.vision_data.splits import (
    DEFAULT_SPLIT_RATIOS,
    build_split_manifest,
    load_split_manifest,
    validate_split_manifest,
    validate_equal_training_budgets,
    validate_training_selection,
    write_split_manifest,
)

__all__ = [
    "DEFAULT_SPLIT_RATIOS",
    "build_split_manifest",
    "load_split_manifest",
    "validate_split_manifest",
    "validate_equal_training_budgets",
    "validate_training_selection",
    "write_split_manifest",
]
