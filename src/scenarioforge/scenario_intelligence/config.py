"""Centralised, JSON-overridable configuration for all scenario-intelligence analyses."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scenarioforge.scenario_intelligence.binning import BinningConfig
from scenarioforge.scenario_intelligence.diversity_analyzer import DiversityConfig
from scenarioforge.scenario_intelligence.duplicate_detector import DuplicateConfig
from scenarioforge.scenario_intelligence.rarity_analyzer import RarityConfig
from scenarioforge.scenario_intelligence.scenario_ranker import RankingConfig

OUTPUT_SUBDIR = ("data", "processed", "scenario_intelligence")


@dataclass(frozen=True)
class ScenarioIntelligenceConfig:
    near_ego_radius_m: float = 50.0
    rarity: RarityConfig = field(default_factory=RarityConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    duplicates: DuplicateConfig = field(default_factory=DuplicateConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScenarioIntelligenceConfig:
        """Build from a nested mapping; unknown keys raise instead of being ignored."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        sections = {
            "rarity": RarityConfig,
            "diversity": DiversityConfig,
            "duplicates": DuplicateConfig,
            "ranking": RankingConfig,
        }
        built: dict[str, Any] = {}
        if "near_ego_radius_m" in data:
            built["near_ego_radius_m"] = float(data["near_ego_radius_m"])
        for name, config_type in sections.items():
            if name in data:
                built[name] = _build(config_type, data[name])
        return cls(**built)

    @classmethod
    def from_file(cls, path: Path) -> ScenarioIntelligenceConfig:
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read configuration file {path}: {exc}") from exc


def _build(config_type: type, data: dict[str, Any]) -> Any:
    names = {f.name for f in dataclasses.fields(config_type)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Unknown keys for {config_type.__name__}: {sorted(unknown)}")
    values = dict(data)
    if "binning" in values:
        values["binning"] = _build(BinningConfig, values["binning"])
    for key in ("binned_features", "fingerprint_features", "numeric_features"):
        if key in values:
            values[key] = tuple(values[key])
    return config_type(**values)


def default_output_dir(project_root: Path) -> Path:
    return Path(project_root).joinpath(*OUTPUT_SUBDIR)
