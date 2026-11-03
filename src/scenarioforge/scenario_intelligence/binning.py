"""Configurable discretisation so categorical metrics can be applied to continuous features.

Entropy, Simpson, and frequency-based rarity are defined for categorical distributions. Numeric
features are therefore binned first. The strategy is explicit and configurable:

  * ``quantile`` (default): edges at equally spaced quantiles of the observed values, so bins
    hold similar numbers of scenarios. Duplicate edges collapse, so ties yield fewer bins.
  * ``uniform``: equal-width bins between the observed minimum and maximum.
  * ``fixed_edges``: user supplied edges per feature override either strategy.

Missing values always map to the ``missing`` bin rather than being imputed. String values are
treated as already categorical and pass through unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scenarioforge.scenario_intelligence.schemas import MISSING_BIN, ScenarioRecord


@dataclass(frozen=True)
class BinningConfig:
    n_bins: int = 4
    strategy: str = "quantile"
    fixed_edges: Mapping[str, Sequence[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_bins < 1:
            raise ValueError("n_bins must be at least 1")
        if self.strategy not in {"quantile", "uniform"}:
            raise ValueError(f"Unknown binning strategy: {self.strategy}")


def fit_edges(values: Iterable[float], config: BinningConfig, feature: str = "") -> list[float]:
    """Return strictly increasing interior bin edges for ``values``."""
    if feature in config.fixed_edges:
        return sorted({float(edge) for edge in config.fixed_edges[feature]})
    data = np.asarray(sorted(float(v) for v in values), dtype=float)
    if data.size == 0 or config.n_bins == 1:
        return []
    if config.strategy == "quantile":
        levels = np.linspace(0, 1, config.n_bins + 1)[1:-1]
        edges = np.quantile(data, levels)
    else:
        edges = np.linspace(data.min(), data.max(), config.n_bins + 1)[1:-1]
    return sorted({round(float(edge), 6) for edge in edges})


def assign_bin(value: Any, edges: Sequence[float]) -> str:
    """Map a value to a bin label; None -> ``missing``; strings pass through."""
    if value is None:
        return MISSING_BIN
    if isinstance(value, str):
        return value
    index = int(np.searchsorted(np.asarray(edges, dtype=float), float(value), side="right"))
    lower = "-inf" if index == 0 else f"{edges[index - 1]:g}"
    upper = "inf" if index >= len(edges) else f"{edges[index]:g}"
    return f"[{lower},{upper})"


class FeatureDiscretizer:
    """Fit bin edges on a record set, then label any record consistently."""

    def __init__(self, features: Sequence[str], config: BinningConfig | None = None) -> None:
        self.features = tuple(features)
        self.config = config or BinningConfig()
        self.edges: dict[str, list[float]] | None = None

    def fit(self, records: Sequence[ScenarioRecord]) -> FeatureDiscretizer:
        self.edges = {}
        for feature in self.features:
            numeric = [
                r.features[feature]
                for r in records
                if r.features.get(feature) is not None
                and not isinstance(r.features[feature], str)
            ]
            self.edges[feature] = fit_edges(numeric, self.config, feature)
        return self

    def transform(self, record: ScenarioRecord) -> dict[str, str]:
        if self.edges is None:
            raise RuntimeError("FeatureDiscretizer must be fitted before use")
        return {
            feature: assign_bin(record.features.get(feature), self.edges[feature])
            for feature in self.features
        }
