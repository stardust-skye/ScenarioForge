"""Persist and plot scenario-intelligence outputs. Plots only render data passed in from reports."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless; safe on CPU-only machines and CI
import matplotlib.pyplot as plt  # noqa: E402

from scenarioforge.scenario_intelligence.schemas import ScenarioRecord  # noqa: E402

SCENARIO_FILE = "scenarios.json"


def write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required input not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def write_flat_csv(payload: Mapping[str, Any], path: Path) -> Path:
    """Write a nested report as ``metric,value`` rows with dotted keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        for key, value in _flatten(payload):
            writer.writerow((key, value))
    return path


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, Any]] = []
        for key in sorted(value, key=str):
            rows.extend(_flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
        return rows
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (Mapping, list)):
        return [(prefix, json.dumps(value, sort_keys=True, default=str))]
    return [(prefix, json.dumps(value) if isinstance(value, (list, tuple)) else value)]


def save_scenarios(records: Sequence[ScenarioRecord], path: Path, dataset_source_id: str) -> Path:
    return write_json(
        {"dataset_source_id": dataset_source_id, "scenarios": [r.to_dict() for r in records]}, path
    )


def load_scenarios(path: Path) -> tuple[str, list[ScenarioRecord]]:
    payload = read_json(path)
    try:
        return payload["dataset_source_id"], [ScenarioRecord.from_dict(p) for p in payload["scenarios"]]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed scenario file {path}: {exc}") from exc


def _finish(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_bar(counts: Mapping[str, float], title: str, path: Path, *, top: int | None = None,
             horizontal: bool = True, xlabel: str = "count") -> Path:
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(items) + 1)) if horizontal else (8, 4))
    labels, values = [k for k, _ in items], [v for _, v in items]
    if horizontal:
        ax.barh(labels[::-1], values[::-1])
        ax.set_xlabel(xlabel)
    else:
        ax.bar(labels, values)
        ax.set_ylabel(xlabel)
        ax.tick_params(axis="x", rotation=60)
    ax.set_title(title)
    return _finish(fig, path)


def plot_histogram(values: Sequence[float], title: str, xlabel: str, path: Path, bins: int = 20) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("scenarios")
    return _finish(fig, path)


def plot_histogram_counts(counts: Sequence[int], edges: Sequence[float], title: str, xlabel: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(edges[:-1], counts, width=[b - a for a, b in zip(edges[:-1], edges[1:])], align="edge")
    ax.set_yscale("symlog")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("scenario pairs (symlog)")
    return _finish(fig, path)


def plot_grouped_distribution(original: Mapping[str, float], curated: Mapping[str, float],
                              title: str, path: Path) -> Path:
    keys = sorted(set(original) | set(curated))
    total_o, total_c = sum(original.values()) or 1, sum(curated.values()) or 1
    fig, ax = plt.subplots(figsize=(9, max(4, 0.34 * len(keys) + 1)))
    width = 0.38
    positions = list(range(len(keys)))
    original_values = [original.get(k, 0) / total_o for k in keys]
    curated_values = [curated.get(k, 0) / total_c for k in keys]
    ax.barh([p - width / 2 for p in positions], original_values, width, label="original")
    ax.barh([p + width / 2 for p in positions], curated_values, width, label="curated")
    ax.set_yticks(positions)
    ax.set_yticklabels(keys)
    ax.set_xlabel("share of annotated objects")
    ax.set_title(title)
    ax.legend()
    return _finish(fig, path)


def plot_stacked_ranking(rows: Sequence[Mapping[str, Any]], title: str, path: Path, top: int = 20) -> Path:
    rows = [r for r in rows if r["scenario_interest_score"] is not None][:top]
    names = sorted({n for r in rows for n in r["contributions"]})
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(rows) + 1.5)))
    left = [0.0] * len(rows)
    labels = [r["sample_id"][:8] for r in rows]
    for name in names:
        widths = [r["contributions"].get(name, 0.0) for r in rows]
        ax.barh(labels[::-1], widths[::-1], left=left[::-1], label=name)
        left = [a + b for a, b in zip(left, widths)]
    ax.set_xlabel("scenario-interest score (component contributions)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _finish(fig, path)
