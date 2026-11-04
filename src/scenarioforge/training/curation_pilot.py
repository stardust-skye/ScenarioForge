"""Preflight and run the audited four-strategy Part 3F.4 curation pilot."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import time
import uuid
import warnings
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from scenarioforge.training.checkpointing import write_json
from scenarioforge.training.config import ExperimentConfig
from scenarioforge.training.experiment import DetectionExperiment
from scenarioforge.training.model_factory import (
    create_detection_model,
    pretrained_checkpoint_provenance,
)
from scenarioforge.training.seed import initialize_seed
from scenarioforge.scenario_intelligence.experiment_curation import (
    balanced_scene_quotas,
)
from scenarioforge.vision_data.dataloader import make_split_datasets, select_training_subset
from scenarioforge.vision_data.dataset import load_dataset_version
from scenarioforge.vision_data.splits import (
    load_split_manifest,
    validate_training_selection,
)
from scenarioforge.vision_data.transforms import build_eval_transforms

AUDITED_RUN_SHA256 = "b7fa8bc6276c626b7a03cc180e1988403dcc8b5b7b091a54495fb07bd92e217d"
AUDITED_CHILD_SHA256 = {
    "random": "85c331dcbd389cb4b48b70a9a19a3d5e57700bd611ce495b9d2ed4cf8712864c",
    "frequency": "bcfe9fc1a3274e72920a0f7e13b41c80b9f9d35c360584b09fdaea2989bd2927",
    "diversity": "580d64e61484b88560233c25d35d164622ff07c08ad5e2933809a0259b83b9c2",
    "weighted_interest": "fd8d61d4854544fbf70a6bba3c2c56617b79bcf480ac54bc4d137fa49a9e6959",
}
CONDITION_NAMES = {
    "random": "Random uniform",
    "frequency": "Frequency-based / inverse rarity",
    "diversity": "Diversity-aware",
    "weighted_interest": "Weighted-interest",
}
SCENE_QUOTA_POLICY = "balanced_capacity_aware_equalization_v1"
PILOT_BUDGET = 60
PILOT_SEED = 42
APPROVED_SEEDS = (42, 123, 2027, 31415, 271828)
AUDITED_CURATION_DIR = Path(
    "data/processed/curation_experiments/curation_20261103T205532Z_77d0c39b"
)


def preflight_curation_pilot(
    *,
    baseline_config_path: Path,
    curation_dir: Path,
    selection_seed: int,
    training_seed: int | None = None,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a seed block without creating experiment outputs or a model."""
    root = Path(repository_root or Path(__file__).resolve().parents[3]).resolve()
    if selection_seed not in APPROVED_SEEDS:
        raise ValueError(f"selection_seed must be one of {APPROVED_SEEDS}")
    training_seed = selection_seed if training_seed is None else training_seed
    if training_seed not in APPROVED_SEEDS:
        raise ValueError(f"training_seed must be one of {APPROVED_SEEDS}")
    if training_seed != selection_seed:
        raise ValueError(
            "The approved seed-block design requires training_seed to equal selection_seed"
        )

    baseline_path = _resolve_from_root(baseline_config_path, root)
    baseline = ExperimentConfig.from_json(baseline_path)
    baseline = replace(
        baseline,
        dataset_root=_resolve_from_root(baseline.dataset_root, root),
        vision_dir=_resolve_from_root(baseline.vision_dir, root),
        split_manifest=_resolve_from_root(baseline.split_manifest, root),
        seed=training_seed,
        selection_seed=selection_seed,
        training_seed=training_seed,
    )
    _validate_baseline_config(baseline, training_seed=training_seed)
    curation_path = Path(curation_dir)
    if not curation_path.is_absolute():
        curation_path = root / curation_path
    curation_path = curation_path.resolve()
    curation = _validate_curation_inputs(
        curation_dir=curation_path,
        config=baseline,
        repository_root=root,
        selection_seed=selection_seed,
    )
    checkpoint_provenance = pretrained_checkpoint_provenance(baseline.model)
    if checkpoint_provenance is None:
        raise ValueError("The pilot requires the pretrained COCO_V1 checkpoint")

    split_manifest = load_split_manifest(
        baseline.split_manifest,
        image_rows=_read_csv(baseline.vision_dir / "image_manifest.csv"),
        dataset_version=load_dataset_version(
            baseline.vision_dir / "dataset_manifest.json"
        ),
    )
    validation_tokens = tuple(split_manifest["sample_tokens"]["validation"])
    condition_results: list[dict[str, Any]] = []
    validation_tokens_by_strategy: dict[str, tuple[str, ...]] = {}
    evaluation_transform = build_eval_transforms(baseline.image_size)
    initial_model_fingerprint: str | None = None

    for strategy in AUDITED_CHILD_SHA256:
        condition = curation["conditions"][strategy]
        datasets = make_split_datasets(
            dataset_root=str(baseline.dataset_root),
            annotations_csv=str(baseline.vision_dir / "annotations.csv"),
            image_manifest_csv=str(baseline.vision_dir / "image_manifest.csv"),
            category_mapping_json=str(baseline.vision_dir / "category_mapping.json"),
            split_manifest=split_manifest,
            transforms={
                "train": evaluation_transform,
                "validation": evaluation_transform,
            },
            splits=("train", "validation"),
        )
        if set(datasets) != {"train", "validation"}:
            raise ValueError("Preflight must instantiate train and validation datasets only")
        train_dataset = select_training_subset(
            datasets["train"],
            split_manifest,
            condition["selected_sample_tokens"],
            image_rows=_read_csv(baseline.vision_dir / "image_manifest.csv"),
        )
        loaded_tokens = tuple(
            train_dataset.dataset.sample_tokens[index]
            for index in train_dataset.indices
        )
        if loaded_tokens != tuple(condition["selected_sample_tokens"]):
            raise ValueError(f"{strategy} dataset order differs from its manifest")
        if len(loaded_tokens) != PILOT_BUDGET or len(set(loaded_tokens)) != PILOT_BUDGET:
            raise ValueError(f"{strategy} did not load exactly 60 unique training samples")
        actual_validation_tokens = tuple(datasets["validation"].sample_tokens)
        if len(actual_validation_tokens) != 80 or set(actual_validation_tokens) != set(
            validation_tokens
        ):
            raise ValueError(f"{strategy} does not load the fixed 80-image validation split")
        validation_tokens_by_strategy[strategy] = actual_validation_tokens

        loaded_annotation_counts: dict[str, int] = {}
        for index, token in enumerate(loaded_tokens):
            _, target = train_dataset[index]
            box_count = int(target["boxes"].shape[0])
            if box_count != condition["annotation_counts"][token]:
                raise ValueError(
                    f"{strategy} sample {token} loaded {box_count} boxes; "
                    f"manifest records {condition['annotation_counts'][token]}"
                )
            loaded_annotation_counts[token] = box_count

        initialize_seed(training_seed, mode=baseline.reproducibility_mode)
        model = create_detection_model(
            baseline.model,
            datasets["train"].model_label_to_source_id,
        )
        if model.pretrained_weights != "COCO_V1":
            raise ValueError(f"{strategy} did not initialize from COCO_V1")
        output_classes = model.detector.roi_heads.box_predictor.cls_score.out_features
        if output_classes != 11:
            raise ValueError(f"{strategy} model has {output_classes} outputs; expected 11")
        fingerprint = _model_fingerprint(model)
        if initial_model_fingerprint is None:
            initial_model_fingerprint = fingerprint
        elif fingerprint != initial_model_fingerprint:
            raise ValueError(f"{strategy} model initialization differs within the seed block")
        del model
        gc.collect()

        condition_results.append({
            "strategy": strategy,
            "selection_seed": selection_seed,
            "training_seed": training_seed,
            "child_manifest_sha256": condition["manifest_sha256"],
            "selected_tokens": list(loaded_tokens),
            "selected_image_count": len(loaded_tokens),
            "selected_annotation_count": sum(loaded_annotation_counts.values()),
            "per_scene_sample_counts": condition["per_scene_sample_counts"],
            "validation_sample_count": len(actual_validation_tokens),
            "validation_tokens_sha256": _token_digest(actual_validation_tokens),
            "loaded_annotation_counts": loaded_annotation_counts,
        })

    if len(set(validation_tokens_by_strategy.values())) != 1:
        raise ValueError("The validation token order differs between strategies")
    control_payload = {
        key: value
        for key, value in baseline.to_dict().items()
        if key not in {"training_sample_tokens_file", "curation_provenance"}
    }
    return {
        "selection_seed": selection_seed,
        "training_seed": training_seed,
        "curation_directory": str(curation_path),
        "run_manifest_sha256": curation["run_manifest_sha256"],
        "split_manifest_digest": split_manifest["manifest_digest"],
        "image_inventory_digest": split_manifest["inventory_digest"],
        "validation_tokens_sha256": _token_digest(validation_tokens),
        "pretrained_checkpoint_provenance": checkpoint_provenance,
        "initial_model_state_sha256": initial_model_fingerprint,
        "controls_sha256": hashlib.sha256(
            json.dumps(control_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "test_dataset_instantiated": False,
        "model_constructed": True,
        "conditions": condition_results,
    }


def run_curation_pilot(
    *,
    baseline_config_path: Path,
    curation_dir: Path | None = None,
    output_root: Path,
    report_path: Path | None = None,
    selection_seed: int = PILOT_SEED,
    training_seed: int | None = None,
    strategy_order: tuple[str, ...] | None = None,
    repository_root: Path | None = None,
) -> Path:
    """Validate seed-specific curation inputs, then run each experiment sequentially."""
    root = Path(repository_root or Path(__file__).resolve().parents[3]).resolve()
    baseline_path = _resolve_from_root(baseline_config_path, root)
    if selection_seed not in APPROVED_SEEDS:
        raise ValueError(f"selection_seed must be one of {APPROVED_SEEDS}")
    training_seed = selection_seed if training_seed is None else training_seed
    if training_seed not in APPROVED_SEEDS:
        raise ValueError(f"training_seed must be one of {APPROVED_SEEDS}")
    if training_seed != selection_seed:
        raise ValueError(
            "The approved seed-block design requires training_seed to equal selection_seed"
        )
    strategy_order = _validate_strategy_order(strategy_order)
    if curation_dir is None:
        if selection_seed != PILOT_SEED:
            raise ValueError(
                "A seed-specific --curation-dir is required when selection_seed is not 42"
            )
        curation_dir = AUDITED_CURATION_DIR
    curation_dir = Path(curation_dir)
    if not curation_dir.is_absolute():
        curation_dir = root / curation_dir
    curation_dir = curation_dir.resolve()
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = output_root.resolve()
    if _is_relative_to(curation_dir, output_root) or _is_relative_to(output_root, curation_dir):
        raise ValueError("Pilot outputs must be isolated from the audited curation artifacts")

    baseline = ExperimentConfig.from_json(baseline_path)
    baseline = replace(
        baseline,
        dataset_root=_resolve_from_root(baseline.dataset_root, root),
        vision_dir=_resolve_from_root(baseline.vision_dir, root),
        split_manifest=_resolve_from_root(baseline.split_manifest, root),
        seed=training_seed,
        selection_seed=selection_seed,
        training_seed=training_seed,
    )
    _validate_baseline_config(baseline, training_seed=training_seed)
    if not torch.cuda.is_available():
        raise RuntimeError("The approved pilot requires CUDA; no CPU fallback is permitted")

    curation = _validate_curation_inputs(
        curation_dir=curation_dir,
        config=baseline,
        repository_root=root,
        selection_seed=selection_seed,
    )
    weight_provenance = pretrained_checkpoint_provenance(baseline.model)
    if weight_provenance is None:
        raise ValueError("The pilot requires the pretrained COCO_V1 checkpoint")

    if report_path is not None:
        report_path = Path(report_path)
        if not report_path.is_absolute():
            report_path = root / report_path
        report_path = report_path.resolve()
        if report_path.exists():
            raise FileExistsError(f"Refusing to overwrite pilot report: {report_path}")
    run_root = _create_unique_run_root(output_root)
    if report_path is None:
        report_path = run_root / "pilot_report.md"
    condition_runs: list[dict[str, Any]] = []
    experiments: list[tuple[str, DetectionExperiment, dict[str, Any]]] = []
    validation_tokens: tuple[str, ...] | None = None

    for strategy in strategy_order:
        condition = curation["conditions"][strategy]
        run_dir = run_root / _condition_slug(strategy)
        run_dir.mkdir(exist_ok=False)
        curation_provenance = {
            "strategy": strategy,
            "budget": PILOT_BUDGET,
            "selection_seed": selection_seed,
            "training_seed": training_seed,
            "seed": selection_seed,
            "selection_randomness": (
                "randomized_subset" if strategy == "random"
                else "deterministic_ranking_for_fixed_training_inputs"
            ),
            "selected_sample_tokens": condition["selected_sample_tokens"],
            "selected_scene_tokens": condition["selected_scene_tokens"],
            "per_scene_sample_counts": condition["per_scene_sample_counts"],
            "scene_quota_policy": condition["scene_quota_policy"],
            "child_manifest_path": str(condition["manifest_path"]),
            "child_manifest_sha256": condition["manifest_sha256"],
            "selected_tokens_path": str(condition["selected_tokens_path"]),
            "selected_tokens_sha256": condition["selected_tokens_sha256"],
            "run_manifest_path": str(curation["run_manifest_path"]),
            "run_manifest_sha256": curation["run_manifest_sha256"],
            "source_hashes": condition["source_hashes"],
        }
        config = replace(
            baseline,
            output_dir=run_root,
            training_sample_tokens_file=condition["selected_tokens_path"],
            training_sample_limit=None,
            validation_sample_limit=None,
            selection_seed=selection_seed,
            training_seed=training_seed,
            curation_provenance=curation_provenance,
            pretrained_checkpoint_provenance=weight_provenance,
        )
        experiment = DetectionExperiment(config, experiment_dir=run_dir)
        selected_actual = tuple(
            experiment.train_dataset.dataset.sample_tokens[index]
            for index in experiment.train_dataset.indices
        )
        if selected_actual != tuple(condition["selected_sample_tokens"]):
            raise ValueError(f"{strategy} dataset order differs from its audited selection")
        if len(selected_actual) != PILOT_BUDGET or len(set(selected_actual)) != PILOT_BUDGET:
            raise ValueError(f"{strategy} did not load exactly 60 unique training samples")
        if set(experiment.datasets) != {"train", "validation"}:
            raise ValueError("Training must instantiate train and validation datasets only")
        if "test" in experiment.datasets or len(experiment.validation_loader.dataset) != 80:
            raise ValueError("Pilot must use the full, fixed 80-image validation partition only")

        current_validation_tokens = tuple(experiment.datasets["validation"].sample_tokens)
        if len(current_validation_tokens) != 80:
            raise ValueError(f"{strategy} validation dataset does not contain 80 images")
        if validation_tokens is None:
            validation_tokens = current_validation_tokens
        elif current_validation_tokens != validation_tokens:
            raise ValueError("The validation dataset differs between pilot conditions")

        class_counts: Counter[str] = Counter()
        for sample_index, token in enumerate(selected_actual):
            _, target = experiment.train_dataset[sample_index]
            actual_boxes = int(target["boxes"].shape[0])
            expected_boxes = condition["annotation_counts"][token]
            if actual_boxes != expected_boxes:
                raise ValueError(
                    f"{strategy} sample {token} loaded {actual_boxes} boxes; "
                    f"the audited manifest records {expected_boxes}"
                )
            for label, count in zip(*torch.unique(target["labels"], return_counts=True)):
                class_name = experiment.model_label_to_name[int(label)]
                class_counts[class_name] += int(count)
        condition["class_counts"] = dict(sorted(class_counts.items()))
        condition["loaded_image_count"] = len(selected_actual)
        condition["loaded_box_count"] = sum(class_counts.values())
        experiments.append((strategy, experiment, condition))
        condition_runs.append({
            "strategy": strategy,
            "display_name": CONDITION_NAMES[strategy],
            "run_directory": str(run_dir),
            "status": "preflight_passed",
            "selected_image_count": len(selected_actual),
            "selected_annotation_count": condition["loaded_box_count"],
            "class_counts": condition["class_counts"],
            "curation_child_manifest_sha256": condition["manifest_sha256"],
            "selection_seed": selection_seed,
            "training_seed": training_seed,
            "selection_randomness": curation_provenance["selection_randomness"],
        })

    model_fingerprint: str | None = None
    for strategy, experiment, _ in experiments:
        _verify_pretrained_checkpoint_provenance(
            pretrained_checkpoint_provenance(experiment.config.model),
            weight_provenance,
        )
        initialize_seed(training_seed, mode=baseline.reproducibility_mode)
        model = create_detection_model(
            experiment.config.model,
            experiment.model_label_to_source_id,
        )
        if model.pretrained_weights != "COCO_V1":
            raise ValueError(f"{strategy} model did not use the COCO_V1 checkpoint")
        output_classes = model.detector.roi_heads.box_predictor.cls_score.out_features
        if output_classes != 11:
            raise ValueError(f"{strategy} model has {output_classes} outputs; expected 11")
        current_fingerprint = _model_fingerprint(model)
        if model_fingerprint is None:
            model_fingerprint = current_fingerprint
        elif current_fingerprint != model_fingerprint:
            raise ValueError("Fresh pilot model initial states are not identical")
        del model
        gc.collect()

    pilot_status: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_root),
        "baseline_config_path": str(baseline_path),
        "training_status": "running",
        "preflight_passed": True,
        "test_evaluated": False,
        "selection_seed": selection_seed,
        "training_seed": training_seed,
        "configuration": baseline.to_dict(),
        "pretrained_checkpoint_provenance": weight_provenance,
        "initial_model_state_sha256": model_fingerprint,
        "curation_run_manifest_sha256": curation["run_manifest_sha256"],
        "conditions": condition_runs,
    }
    status_path = run_root / "pilot_status.json"
    write_json(status_path, pilot_status)

    for index, (strategy, experiment, condition) in enumerate(experiments):
        run_record = condition_runs[index]
        model: torch.nn.Module | None = None
        try:
            _verify_pretrained_checkpoint_provenance(
                pretrained_checkpoint_provenance(experiment.config.model),
                weight_provenance,
            )
            initialize_seed(training_seed, mode=baseline.reproducibility_mode)
            model = experiment.create_model()
            if _model_fingerprint(model) != model_fingerprint:
                raise ValueError(f"{strategy} actual model initialization changed after preflight")
            run_record["model_state_sha256"] = model_fingerprint
            start_time = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                history = experiment.train()
            torch.cuda.synchronize(experiment.device)
            run_record["wall_clock_seconds"] = time.perf_counter() - start_time
            run_record["warnings"] = [
                {
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                }
                for warning in caught_warnings
            ]
            run_record["status"] = "completed"
            run_record["history"] = history
            summary_path = experiment.run_dir / "experiment_summary.json"
            run_record["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
            pilot_status["completed_conditions"] = index + 1
            write_json(status_path, pilot_status)
            for warning in caught_warnings:
                print(f"WARNING [{strategy}]: {warning.category.__name__}: {warning.message}")
            print(
                f"Completed {strategy}: {run_record['summary']['completed_epochs']} epochs; "
                f"best validation mAP={run_record['summary']['best_validation_map_50_95']}"
            )
        except Exception as exc:
            run_record["status"] = "failed"
            run_record["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            pilot_status["training_status"] = "failed"
            pilot_status["failed_strategy"] = strategy
            pilot_status["failure"] = run_record["failure"]
            write_json(status_path, pilot_status)
            _write_pilot_report(report_path, pilot_status)
            raise
        finally:
            experiment.trainer = None
            experiment.model = None
            model = None
            gc.collect()
            torch.cuda.empty_cache()

    pilot_status["training_status"] = "completed"
    pilot_status["test_evaluated"] = False
    write_json(status_path, pilot_status)
    _write_pilot_report(report_path, pilot_status)
    return run_root


def _validate_baseline_config(
    config: ExperimentConfig,
    *,
    training_seed: int = PILOT_SEED,
) -> None:
    expected = {
        "batch_size": 1,
        "num_workers": 0,
        "seed": training_seed,
        "epochs": 6,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "gradient_accumulation_steps": 1,
        "gradient_clip_norm": 1.0,
        "mixed_precision": True,
        "reproducibility_mode": "best_effort",
        "validation_interval": 1,
        "early_stopping_patience": None,
        "scheduler_step_size": 3,
        "scheduler_gamma": 0.1,
        "device": "cuda",
        "image_size": None,
        "training_sample_limit": None,
        "validation_sample_limit": None,
    }
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError(
                f"Baseline control {name} is {getattr(config, name)!r}; expected {value!r}"
            )
    if config.training_sample_tokens_file is not None:
        raise ValueError("Baseline config must not already select a curated subset")
    if (
        config.model.architecture != "fasterrcnn_mobilenet_v3_large_320_fpn"
        or config.model.pretrained_weights != "COCO_V1"
        or config.model.freeze_backbone_epochs != 0
    ):
        raise ValueError("Baseline detector or pretrained initialization differs from Part 3D.3")


def _validate_curation_inputs(
    *,
    curation_dir: Path,
    config: ExperimentConfig,
    repository_root: Path,
    selection_seed: int = PILOT_SEED,
) -> dict[str, Any]:
    run_manifest_path = curation_dir / "experiment_manifest.json"
    run_manifest_sha256 = _sha256_file(run_manifest_path)
    run_manifest = _read_json_object(run_manifest_path)
    is_original_audited_bundle = (
        selection_seed == PILOT_SEED
        and curation_dir.resolve()
        == (repository_root / AUDITED_CURATION_DIR).resolve()
    )
    if is_original_audited_bundle and run_manifest_sha256 != AUDITED_RUN_SHA256:
        raise ValueError("Run-level curation manifest hash differs from the independent audit")
    if (
        run_manifest.get("seed") != selection_seed
        or run_manifest.get("selection_seed", run_manifest.get("seed")) != selection_seed
        or run_manifest.get("training_seed", run_manifest.get("seed")) != selection_seed
        or run_manifest.get("train_sample_count") != 242
        or run_manifest.get("test_evaluated") is not False
    ):
        raise ValueError("Curation run manifest does not describe the approved train-only artifact")
    if run_manifest.get("dataset_version") != load_dataset_version(
        config.vision_dir / "dataset_manifest.json"
    ):
        raise ValueError("Curation parent-run dataset version differs from the active dataset")
    child_manifest_names = run_manifest.get("subset_manifests")
    if not isinstance(child_manifest_names, list) or not all(
        isinstance(name, str) for name in child_manifest_names
    ):
        raise ValueError("Curation run manifest is missing child-manifest provenance")
    expected_child_names = {
        f"{strategy}_n{PILOT_BUDGET}_seed{selection_seed}_manifest.json"
        for strategy in AUDITED_CHILD_SHA256
    }
    if not expected_child_names.issubset(set(child_manifest_names)):
        raise ValueError("Curation run manifest does not list all four required child manifests")

    split_path = config.split_manifest
    vision_dir = config.vision_dir
    source_paths = {
        "split_manifest_sha256": split_path,
        "image_manifest_sha256": vision_dir / "image_manifest.csv",
        "annotation_manifest_sha256": vision_dir / "annotations.csv",
        "category_mapping_sha256": vision_dir / "category_mapping.json",
        "dataset_manifest_sha256": vision_dir / "dataset_manifest.json",
        "scenario_file_sha256": repository_root
        / "data"
        / "processed"
        / "scenario_intelligence"
        / "scenarios.json",
    }
    expected_source_hashes = run_manifest.get("source_hashes")
    if not isinstance(expected_source_hashes, dict):
        expected_source_hashes = {
            key: run_manifest.get(key) for key in source_paths
        }
    for key, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required curation source is missing: {path}")
        if _sha256_file(path) != expected_source_hashes.get(key):
            raise ValueError(f"Curation source hash mismatch for {key}")

    image_rows = _read_csv(source_paths["image_manifest_sha256"])
    annotation_rows = _read_csv(source_paths["annotation_manifest_sha256"])
    dataset_version = load_dataset_version(source_paths["dataset_manifest_sha256"])
    split_manifest = load_split_manifest(
        split_path,
        image_rows=image_rows,
        dataset_version=dataset_version,
    )
    if (
        split_manifest.get("strategy") != "seeded_scene_grouped"
        or split_manifest.get("seed") != PILOT_SEED
        or split_manifest.get("manifest_digest") != run_manifest.get("split_manifest_digest")
    ):
        raise ValueError("The authoritative seed-42 split differs from the curation provenance")
    train_tokens = set(split_manifest["sample_tokens"]["train"])
    validation_tokens = set(split_manifest["sample_tokens"]["validation"])
    test_tokens = set(split_manifest["sample_tokens"]["test"])
    if (len(train_tokens), len(validation_tokens), len(test_tokens)) != (242, 80, 82):
        raise ValueError("The authoritative split sizes are not 242/80/82")

    image_by_sample = {row["sample_token"]: row for row in image_rows}
    annotations_by_sample: Counter[str] = Counter()
    for row in annotation_rows:
        token = row.get("sample_token", "")
        if token in train_tokens and row.get("conversion_status") == "accepted":
            annotations_by_sample[token] += 1
    expected_source_hashes = run_manifest.get("source_hashes", expected_source_hashes)
    if not isinstance(expected_source_hashes, dict):
        raise ValueError("Curation run manifest source hashes must be an object")
    training_scenes = {
        image_by_sample[token]["scene_token"] for token in train_tokens
    }
    expected_quotas = balanced_scene_quotas(
        dict(Counter(image_by_sample[token]["scene_token"] for token in train_tokens)),
        PILOT_BUDGET,
    )
    if set(expected_quotas.values()) != {10}:
        raise ValueError("Authoritative training scenes do not support the approved 10-per-scene quota")
    conditions: dict[str, Any] = {}
    for strategy in AUDITED_CHILD_SHA256:
        stem = f"{strategy}_n{PILOT_BUDGET}_seed{selection_seed}"
        manifest_path = curation_dir / f"{stem}_manifest.json"
        selected_tokens_path = curation_dir / f"{stem}_selected.json"
        if not manifest_path.is_file() or not selected_tokens_path.is_file():
            raise FileNotFoundError(f"Missing seed-{selection_seed} manifest for {strategy}")
        manifest = _read_json_object(manifest_path)
        actual_digest = _canonical_manifest_digest(manifest)
        known_audited_digest = (
            AUDITED_CHILD_SHA256[strategy]
            if is_original_audited_bundle
            else None
        )
        if (
            actual_digest != manifest.get("manifest_sha256")
            or (
                known_audited_digest is not None
                and actual_digest != known_audited_digest
            )
        ):
            raise ValueError(f"{strategy} child manifest hash is invalid")
        if (
            manifest.get("selection_strategy") != strategy
            or manifest.get("budget") != PILOT_BUDGET
            or manifest.get("selected_count") != PILOT_BUDGET
            or manifest.get("seed") != selection_seed
            or manifest.get("selection_seed", manifest.get("seed")) != selection_seed
            or manifest.get("training_seed", manifest.get("seed")) != selection_seed
            or manifest.get("dataset_version") != dataset_version
            or manifest.get("dataset_source_id") != run_manifest.get("dataset_source_id")
            or manifest.get("split_manifest_digest") != split_manifest["manifest_digest"]
            or manifest.get("scoring_configuration") != run_manifest.get(
                "scoring_configuration"
            )
            or manifest.get("requested_count_satisfied") is not True
            or manifest.get("scene_quota_policy") != SCENE_QUOTA_POLICY
        ):
            raise ValueError(f"{strategy} child manifest does not match the approved condition")
        if manifest.get("scene_quota_policy") != run_manifest.get("scene_quota_policy"):
            raise ValueError(f"{strategy} parent-run quota policy mismatch")
        for key, expected_hash in expected_source_hashes.items():
            if manifest.get(key) != expected_hash:
                raise ValueError(f"{strategy} child manifest source hash mismatch: {key}")
        selected_tokens = manifest.get("selected_sample_tokens")
        if (
            not isinstance(selected_tokens, list)
            or len(selected_tokens) != PILOT_BUDGET
            or not all(isinstance(token, str) and token for token in selected_tokens)
            or len(set(selected_tokens)) != PILOT_BUDGET
        ):
            raise ValueError(f"{strategy} does not select exactly 60 unique sample tokens")
        selected_file_tokens = json.loads(selected_tokens_path.read_text(encoding="utf-8"))
        if selected_file_tokens != selected_tokens:
            raise ValueError(f"{strategy} selected-token file differs from its child manifest")
        validate_training_selection(
            split_manifest,
            selected_tokens,
            image_rows=image_rows,
            dataset_version=dataset_version,
        )
        if set(selected_tokens) & (validation_tokens | test_tokens):
            raise ValueError(f"{strategy} selection contains held-out sample tokens")
        if manifest.get("training_scenario_digest_sha256") != run_manifest.get(
            "training_scenario_digest_sha256"
        ):
            raise ValueError(f"{strategy} parent-run scenario provenance mismatch")
        per_scene = manifest.get("per_scene_sample_counts")
        scene_quotas = manifest.get("scene_quotas")
        if not isinstance(per_scene, dict) or not isinstance(scene_quotas, dict):
            raise ValueError(f"{strategy} is missing scene-quota assignments")
        actual_scene_counts = dict(sorted(Counter(
            image_by_sample[token]["scene_token"] for token in selected_tokens
        ).items()))
        if (
            per_scene != actual_scene_counts
            or per_scene != scene_quotas
            or per_scene != expected_quotas
            or scene_quotas != expected_quotas
            or set(per_scene) != training_scenes
            or manifest.get("selected_scene_tokens")
            != sorted({image_by_sample[token]["scene_token"] for token in selected_tokens})
        ):
            raise ValueError(f"{strategy} does not satisfy the approved common scene quotas")
        annotation_counts = manifest.get("selected_annotation_counts")
        if not isinstance(annotation_counts, dict):
            raise ValueError(f"{strategy} is missing annotation-count provenance")
        for token in selected_tokens:
            if token not in image_by_sample:
                raise ValueError(f"{strategy} contains an unknown sample token: {token}")
            image_path = (
                config.dataset_root / image_by_sample[token]["image_path"]
            ).resolve()
            if not _is_relative_to(image_path, config.dataset_root.resolve()):
                raise ValueError(f"{strategy} image path escapes the dataset root: {token}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Selected image file is missing for {token}: {image_path}")
            if image_by_sample[token].get("scene_token") not in training_scenes:
                raise ValueError(f"{strategy} has an invalid training-scene assignment: {token}")
            if int(image_by_sample[token]["annotation_count"]) != annotations_by_sample[token]:
                raise ValueError(f"Training annotation inventory mismatch for {token}")
            if annotation_counts.get(token) != annotations_by_sample[token]:
                raise ValueError(f"{strategy} annotation count mismatch for {token}")
        conditions[strategy] = {
            "manifest_path": manifest_path.resolve(),
            "manifest_sha256": actual_digest,
            "selected_tokens_path": selected_tokens_path.resolve(),
            "selected_tokens_sha256": _sha256_file(selected_tokens_path),
            "selected_sample_tokens": selected_tokens,
            "selected_scene_tokens": manifest.get("selected_scene_tokens"),
            "per_scene_sample_counts": per_scene,
            "scene_quota_policy": manifest["scene_quota_policy"],
            "annotation_counts": annotation_counts,
            "source_hashes": {
                key: manifest[key]
                for key in sorted(expected_source_hashes)
            },
            "selection_seed": selection_seed,
        }
    return {
        "run_manifest_path": run_manifest_path.resolve(),
        "run_manifest_sha256": run_manifest_sha256,
        "conditions": conditions,
    }


def _write_pilot_report(path: Path, status: dict[str, Any]) -> None:
    rows = status["conditions"]
    lines = [
        "# Part 3F.4 - Controlled Curation Pilot Report",
        "",
        f"**Status:** {status['training_status']}  ",
        f"**Test partition evaluated:** {status.get('test_evaluated', False)}  ",
        f"**Preflight passed:** {status.get('preflight_passed', False)}",
        "",
        "## Objective and controls",
        "",
        "Compare four train-only, scene-balanced 60-image subsets using the same "
        "Faster R-CNN MobileNetV3 Large 320 FPN architecture, COCO_V1 initialization, "
        f"selection/training seed {status.get('selection_seed', PILOT_SEED)}, six epochs, "
        "batch size 1, AdamW, StepLR, AMP, preprocessing, and the "
        "same 80-image validation partition. This single-seed feasibility pilot does "
        "not establish statistical significance or general improvement.",
        "",
        "The training command instantiated only train and validation datasets. "
        "No test dataset was instantiated or evaluated.",
        "",
        "## Preflight and provenance",
        "",
        f"- Curation run-manifest SHA-256: `{status.get('curation_run_manifest_sha256', 'not recorded')}`",
        f"- COCO_V1 checkpoint SHA-256: `{status.get('pretrained_checkpoint_provenance', {}).get('sha256', 'not recorded')}`",
        f"- Initial model-state SHA-256: `{status.get('initial_model_state_sha256', 'not recorded')}`",
        "- Each completed checkpoint contains curation child/run-manifest hashes and the pretrained checkpoint hash in metadata.",
        f"- Unique non-overwriting run directory: `{status.get('run_directory', 'not recorded')}`",
        "- Reproduction command requires the matching `--selection-seed` and explicit `--curation-dir` for non-42 manifests.",
        "",
        "## Results",
        "",
        "| Strategy | Status | Images / boxes | Best epoch | Best val mAP@[0.50:0.95] | AP50 | AP75 | Runtime (s) | Peak CUDA allocation (bytes) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"- **{row['display_name']}** child-manifest SHA-256: "
            f"`{row.get('curation_child_manifest_sha256', 'not recorded')}`; "
            f"output: `{row.get('run_directory', 'not recorded')}`"
        )
    lines.append("")
    for row in rows:
        summary = row.get("summary", {})
        metrics = _best_validation_metrics(row)
        lines.append(
            "| {name} | {status} | {images} / {boxes} | {epoch} | {map} | {ap50} | "
            "{ap75} | {runtime} | {memory} |".format(
                name=row["display_name"],
                status=row["status"],
                images=row.get("selected_image_count", 60),
                boxes=row.get("selected_annotation_count", "n/a"),
                epoch=summary.get("best_validation_epoch", "n/a"),
                map=_format_metric(metrics.get("map_50_95")),
                ap50=_format_metric(metrics.get("map_50")),
                ap75=_format_metric(metrics.get("map_75")),
                runtime=_format_metric(
                    row.get("wall_clock_seconds", summary.get("training_duration_seconds"))
                ),
                memory=summary.get("peak_cuda_memory_allocated_bytes", "n/a"),
            )
        )
    lines.extend(["", "## Per-class validation metrics", ""])
    for row in rows:
        metrics = _best_validation_metrics(row)
        per_class = metrics.get("per_class", {})
        if not per_class:
            continue
        lines.extend([
            f"### {row['display_name']}",
            "",
            "| Class | Validation GT support | AP | Precision (IoU 0.50) | Recall (IoU 0.50) |",
            "|---|---:|---:|---:|---:|",
        ])
        for name, values in per_class.items():
            lines.append(
                f"| {name} | {values.get('ground_truth_objects', '—')} | "
                f"{_format_metric(values.get('average_precision'))} | "
                f"{_format_metric(values.get('precision_iou50'))} | "
                f"{_format_metric(values.get('recall_iou50'))} |"
            )
        lines.append("")
    lines.extend(["## Selected training-set class composition", ""])
    for row in rows:
        class_counts = row.get("class_counts")
        if not class_counts:
            continue
        lines.extend([
            f"### {row['display_name']}",
            "",
            "| Class | Accepted training boxes |",
            "|---|---:|",
        ])
        lines.extend(f"| {name} | {count} |" for name, count in class_counts.items())
        lines.append("")
    lines.extend(["## Loss curves and epoch timings", ""])
    lines.extend([
        "Training total loss and its components are recorded per epoch. The existing "
        "trainer does not compute validation loss; validation mAP is shown instead, "
        "and no validation-loss values are inferred.",
        "",
    ])
    for row in rows:
        history = row.get("history", [])
        if not history:
            continue
        lines.extend([
            f"### {row['display_name']}",
            "",
            "| Epoch | Train total loss | Validation mAP@[0.50:0.95] | Epoch seconds |",
            "|---:|---:|---:|---:|",
        ])
        for epoch in history:
            validation = epoch.get("validation_metrics") or {}
            lines.append(
                f"| {epoch.get('epoch')} | {_format_metric(epoch.get('training_loss'))} | "
                f"{_format_metric(validation.get('map_50_95'))} | "
                f"{_format_metric(epoch.get('epoch_duration_seconds'))} |"
            )
        lines.append("")
    lines.extend([
        "## Warnings, failures, and limitations",
        "",
    ])
    for row in rows:
        warning_counts = Counter(
            (warning["category"], warning["message"])
            for warning in row.get("warnings", [])
        )
        for (category, message), count in sorted(warning_counts.items()):
            lines.append(
                f"- **{row['display_name']} warning ({count} occurrences):** "
                f"{category}: {message}"
            )
        if row.get("failure"):
            lines.append(
                f"- **{row['display_name']} failed:** {row['failure']['type']}: "
                f"{row['failure']['message']}"
            )
    if status.get("failure"):
        lines.append(
            f"- Pilot stopped at **{status.get('failed_strategy')}**; no subsequent "
            "strategy was launched."
        )
    lines.extend([
        "- One seed and a small nuScenes mini validation set do not support significance claims or broad generalization.",
        "- Metadata-derived curation diversity is not visual diversity; subset composition is not detection performance.",
        "- Validation AP for a class without ground-truth support is undefined, not zero.",
        "- The historical 242-image six-epoch baseline (best validation mAP 0.038926 at epoch 4, peak allocation 435,321,856 bytes, approximately 810 seconds) is a non-matched historical reference, not an independent replicate.",
        "",
        "## Readiness for the multi-seed matrix",
        "",
        (
            "The pilot completed with all four runs; it validates execution and provenance "
            "plumbing only. Repeat the controlled matrix with multiple seeds before drawing "
            "comparative or general conclusions."
            if status["training_status"] == "completed"
            else "The pilot did not complete all four runs. Diagnose the recorded failure before proceeding to a multi-seed matrix."
        ),
        "",
        "Detailed per-run artifacts, checkpoints, and provenance are under the unique run directory recorded in `pilot_status.json`.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _best_validation_metrics(run_record: dict[str, Any]) -> dict[str, Any]:
    summary = run_record.get("summary", {})
    best_epoch = summary.get("best_validation_epoch")
    for record in run_record.get("history", []):
        if record.get("epoch") == best_epoch:
            return record.get("validation_metrics") or {}
    return {}


def _canonical_manifest_digest(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_strategy_order(strategy_order: tuple[str, ...] | None) -> tuple[str, ...]:
    strategies = tuple(AUDITED_CHILD_SHA256)
    if strategy_order is None:
        return strategies
    requested = tuple(strategy_order)
    if len(requested) != len(strategies) or set(requested) != set(strategies):
        raise ValueError(
            "strategy_order must contain each approved strategy exactly once: "
            + ", ".join(strategies)
        )
    return requested


def _token_digest(tokens: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(list(tokens), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _verify_pretrained_checkpoint_provenance(
    actual: dict[str, str] | None,
    expected: dict[str, str] | None,
) -> None:
    if expected is None or actual != expected:
        raise ValueError("Cached pretrained checkpoint provenance differs between conditions")


def _model_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_from_root(path: Path, root: Path) -> Path:
    path = Path(path)
    return (path if path.is_absolute() else root / path).resolve()


def _condition_slug(strategy: str) -> str:
    return {
        "random": "random_uniform",
        "frequency": "frequency_based",
        "diversity": "diversity_aware",
        "weighted_interest": "weighted_interest",
    }[strategy]


def _create_unique_run_root(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = output_root / f"part3f4_{timestamp}_{uuid.uuid4().hex[:8]}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError("Could not allocate a unique Part 3F.4 output directory")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _format_metric(value: Any) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
