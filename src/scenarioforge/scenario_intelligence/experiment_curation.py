"""Leakage-safe curation subsets for controlled training experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, UnidentifiedImageError

from scenarioforge.scenario_intelligence.curation_engine import input_digest
from scenarioforge.scenario_intelligence.metrics import shannon_entropy
from scenarioforge.scenario_intelligence.rarity_analyzer import RarityReport, analyze_rarity
from scenarioforge.scenario_intelligence.scenario_ranker import (
    RankingConfig,
    rank_scenarios,
)
from scenarioforge.scenario_intelligence.schemas import ScenarioRecord
from scenarioforge.vision_data.dataset import load_dataset_version
from scenarioforge.vision_data.splits import (
    load_split_manifest,
    validate_training_selection,
    validate_split_manifest,
)

DEFAULT_REDUCED_BUDGETS = (60, 120, 180)
SCENE_QUOTA_POLICY = "balanced_capacity_aware_equalization_v1"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_training_scenarios(
    records: Sequence[ScenarioRecord],
    *,
    split_manifest: dict[str, Any],
    image_rows: Sequence[dict[str, str]],
    annotations: Sequence[dict[str, str]],
    dataset_root: Path,
) -> tuple[list[ScenarioRecord], list[dict[str, Any]]]:
    """Validate all scenario identities and return only verified training records."""
    validate_split_manifest(
        split_manifest,
        image_rows,
        dataset_version=str(split_manifest.get("dataset_version", "")),
    )
    image_by_sample: dict[str, dict[str, str]] = {}
    for row in image_rows:
        token = row.get("sample_token", "")
        if not token or token in image_by_sample:
            raise ValueError(f"Image manifest has a missing or duplicate sample token: {token!r}")
        image_by_sample[token] = row

    record_by_sample: dict[str, ScenarioRecord] = {}
    for record in records:
        token = record.sample_id
        if not token or token in record_by_sample:
            raise ValueError(f"Scenario records have a missing or duplicate sample token: {token!r}")
        if token not in image_by_sample:
            raise ValueError(f"Scenario record references unknown sample token: {token}")
        if record.scene_id != image_by_sample[token].get("scene_token"):
            raise ValueError(f"Scenario/image scene mismatch for sample token {token}")
        record_by_sample[token] = record

    train_tokens = set(split_manifest["sample_tokens"]["train"])
    missing_train = sorted(train_tokens - set(record_by_sample))
    if missing_train:
        raise ValueError(f"Scenario records are missing training sample tokens: {missing_train[:10]}")

    selected_records = [record_by_sample[token] for token in sorted(train_tokens)]
    train_image_rows = {token: image_by_sample[token] for token in train_tokens}
    annotations_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_annotation_tokens: set[str] = set()
    known_samples = set(image_by_sample)
    for annotation in annotations:
        token = annotation.get("sample_token", "")
        if token not in known_samples:
            raise ValueError(f"Annotation references unknown sample token: {token!r}")
        if token not in train_tokens:
            continue
        image = train_image_rows[token]
        annotation_id = annotation.get("source_annotation_token", "")
        if not annotation_id or annotation_id in seen_annotation_tokens:
            raise ValueError(f"Training annotations have a missing or duplicate source token: {annotation_id!r}")
        seen_annotation_tokens.add(annotation_id)
        if annotation.get("conversion_status") != "accepted":
            raise ValueError(f"Training annotation {annotation_id} is not marked accepted")
        for annotation_key, image_key in (
            ("image_id", "image_id"),
            ("scene_token", "scene_token"),
            ("image_path", "image_path"),
            ("image_width", "width"),
            ("image_height", "height"),
        ):
            if annotation.get(annotation_key) != image.get(image_key):
                raise ValueError(
                    f"Training annotation {annotation_id} {annotation_key} does not match "
                    f"the image manifest for {token}"
                )
        try:
            x_min, y_min, x_max, y_max = (
                float(annotation[key]) for key in ("x_min", "y_min", "x_max", "y_max")
            )
            width, height = int(image["width"]), int(image["height"])
            class_id = int(annotation["class_id"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Training annotation {annotation_id} has invalid box/class data") from exc
        if (
            class_id <= 0
            or not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max))
            or not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height)
        ):
            raise ValueError(f"Training annotation {annotation_id} has an invalid box/class")
        annotations_by_sample[token].append(annotation)

    root = Path(dataset_root).expanduser().resolve()
    audit: list[dict[str, Any]] = []
    for token in sorted(train_tokens):
        image = train_image_rows[token]
        if image.get("image_status") != "ok" or not image.get("image_path"):
            raise ValueError(f"Training image {token} has invalid status/path")
        path = (root / image["image_path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Training image path escapes dataset root for {token}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Training image for {token} does not exist: {path}")
        try:
            with Image.open(path) as source:
                source.verify()
            with Image.open(path) as source:
                actual_size = source.size
        except (OSError, Image.DecompressionBombError, UnidentifiedImageError) as exc:
            raise ValueError(f"Training image for {token} is not a valid image: {path}") from exc
        expected_size = (int(image["width"]), int(image["height"]))
        if actual_size != expected_size:
            raise ValueError(
                f"Training image dimensions for {token} are {actual_size}, expected {expected_size}"
            )
        try:
            declared_count = int(image["annotation_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Image manifest has no valid annotation_count for {token}") from exc
        accepted_count = len(annotations_by_sample[token])
        if declared_count != accepted_count:
            raise ValueError(
                f"Accepted annotation count mismatch for {token}: "
                f"image manifest={declared_count}, annotations CSV={accepted_count}"
            )
        audit.append({
            "sample_token": token,
            "scene_token": image["scene_token"],
            "image_path": image["image_path"],
            "image_path_verified": True,
            "accepted_2d_annotation_count": accepted_count,
            "empty_2d_target": accepted_count == 0,
            "validation_status": "valid",
        })
    return selected_records, audit


def balanced_scene_quotas(scene_counts: dict[str, int], budget: int) -> dict[str, int]:
    """Allocate a budget as evenly as possible, respecting each scene's capacity."""
    if not scene_counts or any(count < 1 for count in scene_counts.values()):
        raise ValueError("Every training scene must have at least one candidate image")
    if budget < 1 or budget > sum(scene_counts.values()):
        raise ValueError(
            f"Budget {budget} is impossible for {sum(scene_counts.values())} training images"
        )
    quotas = {scene: 0 for scene in sorted(scene_counts)}
    for _ in range(budget):
        available = [scene for scene in quotas if quotas[scene] < scene_counts[scene]]
        if not available:
            raise ValueError(f"Scene quotas cannot satisfy budget {budget}: {scene_counts}")
        chosen = min(available, key=lambda scene: (quotas[scene], scene))
        quotas[chosen] += 1
    return quotas


def select_train_records(
    records: Sequence[ScenarioRecord],
    *,
    strategy: str,
    quotas: dict[str, int],
    seed: int,
    rarity: RarityReport,
    ranking: RankingConfig,
) -> tuple[list[ScenarioRecord], dict[str, float | None]]:
    """Apply one strategy while enforcing the same per-scene quotas."""
    if strategy not in {"random", "frequency", "diversity", "weighted_interest"}:
        raise ValueError(f"Unsupported experiment curation strategy: {strategy}")
    if set(quotas) != {record.scene_id for record in records}:
        raise ValueError("Scene quotas must cover exactly the training scenes")
    if sum(quotas.values()) < 1:
        raise ValueError("Scene quotas must select at least one sample")
    by_scene: dict[str, list[ScenarioRecord]] = defaultdict(list)
    for record in records:
        by_scene[record.scene_id].append(record)
    for scene, quota in quotas.items():
        if quota < 0 or quota > len(by_scene[scene]):
            raise ValueError(
                f"Impossible quota for scene {scene}: requested {quota}, available {len(by_scene[scene])}"
            )
    ordered = sorted(records, key=lambda item: item.sample_id)
    scores: dict[str, float | None] = {}

    if strategy == "random":
        selected: list[ScenarioRecord] = []
        rng = random.Random(seed)
        for scene in sorted(quotas):
            candidates = sorted(by_scene[scene], key=lambda item: item.sample_id)
            rng.shuffle(candidates)
            selected.extend(candidates[:quotas[scene]])
    elif strategy == "frequency":
        scores = {
            record.sample_id: rarity.scenario_scores[record.sample_id].combined
            for record in ordered
        }
        ranked = sorted(
            ordered,
            key=lambda item: (
                scores[item.sample_id] is None,
                -(scores[item.sample_id] or 0.0),
                item.sample_id,
            ),
        )
        selected = _fill_scene_quotas(ranked, quotas)
    elif strategy == "weighted_interest":
        ranked_scores = rank_scenarios(ordered, rarity, ranking)
        scores = {item.sample_id: item.score for item in ranked_scores}
        by_id = {item.sample_id: item for item in ordered}
        selected = _fill_scene_quotas(
            [by_id[item.sample_id] for item in ranked_scores], quotas
        )
    else:
        selected = []
        remaining = list(ordered)
        selected_by_scene: Counter[str] = Counter()
        category_totals: Counter[str] = Counter()
        while len(selected) < sum(quotas.values()):
            candidates = [
                candidate for candidate in remaining
                if selected_by_scene[candidate.scene_id] < quotas[candidate.scene_id]
            ]
            if not candidates:
                break
            best: ScenarioRecord | None = None
            best_gain = float("-inf")
            for candidate in candidates:
                trial = category_totals + Counter(
                    candidate.features.get("category_counts") or {}
                )
                gain = shannon_entropy(trial)
                if gain > best_gain + 1e-12:
                    best, best_gain = candidate, gain
            if best is None:
                break
            selected.append(best)
            selected_by_scene[best.scene_id] += 1
            category_totals.update(best.features.get("category_counts") or {})
            remaining.remove(best)
            scores[best.sample_id] = best_gain

    if len(selected) != sum(quotas.values()):
        raise ValueError(
            f"{strategy} selected {len(selected)} of {sum(quotas.values())} "
            "samples under the common scene quotas"
        )
    return selected, scores


def _fill_scene_quotas(
    ranked: Sequence[ScenarioRecord], quotas: dict[str, int]
) -> list[ScenarioRecord]:
    counts: Counter[str] = Counter()
    selected: list[ScenarioRecord] = []
    for record in ranked:
        if counts[record.scene_id] < quotas[record.scene_id]:
            selected.append(record)
            counts[record.scene_id] += 1
        if len(selected) == sum(quotas.values()):
            break
    return selected


def create_experiment_subsets(
    *,
    scenario_path: Path,
    vision_dir: Path,
    split_manifest_path: Path,
    dataset_root: Path,
    output_root: Path,
    seed: int = 42,
    reduced_budgets: Sequence[int] = DEFAULT_REDUCED_BUDGETS,
) -> Path:
    """Validate train-only inputs and create immutable curation subset artifacts."""
    from scenarioforge.scenario_intelligence.reporting import load_scenarios

    vision_dir = Path(vision_dir)
    scenario_path = Path(scenario_path)
    split_manifest_path = Path(split_manifest_path)
    dataset_root = Path(dataset_root)
    image_path = vision_dir / "image_manifest.csv"
    annotation_path = vision_dir / "annotations.csv"
    category_mapping_path = vision_dir / "category_mapping.json"
    dataset_manifest_path = vision_dir / "dataset_manifest.json"
    source_paths = {
        "split_manifest_sha256": split_manifest_path,
        "image_manifest_sha256": image_path,
        "annotation_manifest_sha256": annotation_path,
        "category_mapping_sha256": category_mapping_path,
        "dataset_manifest_sha256": dataset_manifest_path,
        "scenario_file_sha256": scenario_path,
    }
    source_hashes = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    image_rows = read_csv_rows(image_path)
    dataset_version = load_dataset_version(dataset_manifest_path)
    split_manifest = load_split_manifest(
        split_manifest_path,
        image_rows=image_rows,
        dataset_version=dataset_version,
    )
    if (
        split_manifest.get("strategy") != "seeded_scene_grouped"
        or split_manifest.get("seed") != 42
    ):
        raise ValueError("Curation requires the authoritative seeded scene split with seed 42")
    dataset_source_id, all_records = load_scenarios(scenario_path)
    annotation_rows = read_csv_rows(annotation_path)
    train_records, annotation_audit = validate_training_scenarios(
        all_records,
        split_manifest=split_manifest,
        image_rows=image_rows,
        annotations=annotation_rows,
        dataset_root=dataset_root,
    )
    expected_count = len(split_manifest["sample_tokens"]["train"])
    if expected_count != 242:
        raise ValueError(
            f"Expected the approved 242-image training partition, found {expected_count}"
        )
    budgets = tuple(int(value) for value in reduced_budgets)
    if len(set(budgets)) != len(budgets) or any(
        value < 1 or value >= expected_count for value in budgets
    ):
        raise ValueError("Reduced budgets must be unique and between 1 and 241")
    if not budgets:
        raise ValueError("At least one reduced budget is required")

    rarity = analyze_rarity(train_records)
    ranking = RankingConfig()
    scene_counts = dict(Counter(record.scene_id for record in train_records))
    timestamp = datetime.now(timezone.utc).isoformat()
    output_root = Path(output_root)
    run_dir = output_root / (
        f"curation_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    )

    subset_payloads: list[tuple[str, int, list[ScenarioRecord], dict[str, float | None], dict[str, int]]] = []
    for budget in budgets:
        quotas = balanced_scene_quotas(scene_counts, budget)
        for strategy in ("random", "frequency", "diversity", "weighted_interest"):
            selected, scores = select_train_records(
                train_records,
                strategy=strategy,
                quotas=quotas,
                seed=seed,
                rarity=rarity,
                ranking=ranking,
            )
            subset_payloads.append((strategy, budget, selected, scores, quotas))
    full_quotas = dict(sorted(scene_counts.items()))
    subset_payloads.append(
        ("full_reference", expected_count, train_records, {}, full_quotas)
    )

    validate_training_selection(
        split_manifest,
        [record.sample_id for record in train_records],
        image_rows=image_rows,
        dataset_version=dataset_version,
    )
    changed_sources = [
        name for name, path in source_paths.items()
        if sha256_file(path) != source_hashes[name]
    ]
    if changed_sources:
        raise ValueError(
            f"Curation source files changed during subset generation: {changed_sources}"
        )
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite curation output: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    common_provenance = {
        "dataset_version": dataset_version,
        "dataset_source_id": dataset_source_id,
        "split_manifest_digest": split_manifest["manifest_digest"],
        **source_hashes,
        "training_scenario_digest_sha256": input_digest(train_records),
        "seed": seed,
        "created_at_utc": timestamp,
        "scene_quota_policy": SCENE_QUOTA_POLICY,
        "scoring_configuration": {
            "rarity": {
                "config": {
                    "epsilon": rarity.config.epsilon,
                    "rare_max_share": rarity.config.rare_max_share,
                    "combination_level": rarity.config.combination_level,
                    "binned_features": list(rarity.config.binned_features),
                    "binning": {
                        "strategy": rarity.config.binning.strategy,
                        "n_bins": rarity.config.binning.n_bins,
                    },
                    "weights": dict(rarity.config.weights),
                },
                "category_frequency": rarity.category_frequency,
                "combination_frequency": rarity.combination_frequency,
                "feature_value_frequency": rarity.feature_value_frequency,
                "bin_edges": rarity.bin_edges,
            },
            "weighted_interest": {
                "weights": dict(ranking.weights),
                "preferences": dict(ranking.preferences),
            },
            "diversity_objective": "greedy Shannon entropy of cumulative object-category counts",
        },
        "training_image_validation_audit": annotation_audit,
        "empty_2d_target_count": sum(
            item["empty_2d_target"] for item in annotation_audit
        ),
    }

    manifest_paths: list[str] = []
    for strategy, budget, selected, scores, quotas in subset_payloads:
        name = f"{strategy}_n{budget}_seed{seed}"
        counts = Counter(record.scene_id for record in selected)
        sample_tokens = [record.sample_id for record in selected]
        validate_training_selection(
            split_manifest,
            sample_tokens,
            image_rows=image_rows,
            dataset_version=dataset_version,
        )
        if dict(sorted(counts.items())) != dict(sorted(quotas.items())):
            raise ValueError(f"Generated subset {name} violates the registered scene quotas")
        scene_tokens = sorted({record.scene_id for record in selected})
        token_scene = {record.sample_id: record.scene_id for record in train_records}
        manifest = {
            "schema_version": 1,
            **common_provenance,
            "selection_strategy": strategy,
            "budget": budget,
            "requested_budget": budget,
            "selected_count": len(selected),
            "selected_sample_tokens": sample_tokens,
            "selected_scene_tokens": scene_tokens,
            "per_scene_sample_counts": dict(sorted(counts.items())),
            "scene_quotas": quotas,
            "selected_annotation_counts": {
                record.sample_id: next(
                    item["accepted_2d_annotation_count"]
                    for item in annotation_audit
                    if item["sample_token"] == record.sample_id
                )
                for record in selected
            },
            "selection_scores": {
                token: scores.get(token) for token in sample_tokens
            },
            "requested_count_satisfied": len(selected) == budget,
            "manifest_sha256": "",
        }
        if len(sample_tokens) != budget or len(set(sample_tokens)) != budget:
            raise ValueError(f"Generated subset {name} does not satisfy its exact budget")
        manifest["manifest_sha256"] = hashlib.sha256(
            json.dumps(
                {key: value for key, value in manifest.items() if key != "manifest_sha256"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        (run_dir / f"{name}_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run_dir / f"{name}_selected.json").write_text(
            json.dumps(sample_tokens, indent=2) + "\n",
            encoding="utf-8",
        )
        with (run_dir / f"{name}_selected.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ("selection_rank", "sample_token", "scene_token", "accepted_2d_annotation_count", "selection_score")
            )
            annotation_counts = manifest["selected_annotation_counts"]
            for rank, record in enumerate(selected, start=1):
                writer.writerow((
                    rank,
                    record.sample_id,
                    token_scene[record.sample_id],
                    annotation_counts[record.sample_id],
                    scores.get(record.sample_id),
                ))
        manifest_paths.append(f"{name}_manifest.json")

    run_manifest = {
        **common_provenance,
        "run_directory": run_dir.name,
        "train_sample_count": len(train_records),
        "training_scene_counts": dict(sorted(scene_counts.items())),
        "subset_manifests": manifest_paths,
        "heldout_scenarios_excluded_before_scoring": len(all_records) - len(train_records),
        "test_evaluated": False,
    }
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir
