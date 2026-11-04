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
from scenarioforge.vision_data.dataset import load_dataset_version
from scenarioforge.vision_data.splits import (
    load_split_manifest,
    validate_training_selection,
)

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


def run_curation_pilot(
    *,
    baseline_config_path: Path,
    curation_dir: Path,
    output_root: Path,
    report_path: Path,
    repository_root: Path | None = None,
) -> Path:
    """Validate all pilot inputs, then run each fresh experiment sequentially."""
    root = Path(repository_root or Path(__file__).resolve().parents[3]).resolve()
    baseline_path = _resolve_from_root(baseline_config_path, root)
    curation_dir = Path(curation_dir)
    if not curation_dir.is_absolute():
        curation_dir = root / curation_dir
    curation_dir = curation_dir.resolve()
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = output_root.resolve()
    report_path = Path(report_path)
    if not report_path.is_absolute():
        report_path = root / report_path
    report_path = report_path.resolve()

    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite pilot report: {report_path}")
    if _is_relative_to(curation_dir, output_root) or _is_relative_to(output_root, curation_dir):
        raise ValueError("Pilot outputs must be isolated from the audited curation artifacts")

    baseline = ExperimentConfig.from_json(baseline_path)
    baseline = replace(
        baseline,
        dataset_root=_resolve_from_root(baseline.dataset_root, root),
        vision_dir=_resolve_from_root(baseline.vision_dir, root),
        split_manifest=_resolve_from_root(baseline.split_manifest, root),
    )
    _validate_baseline_config(baseline)
    if not torch.cuda.is_available():
        raise RuntimeError("The approved pilot requires CUDA; no CPU fallback is permitted")

    curation = _validate_curation_inputs(
        curation_dir=curation_dir,
        config=baseline,
        repository_root=root,
    )
    weight_provenance = pretrained_checkpoint_provenance(baseline.model)
    if weight_provenance is None:
        raise ValueError("The pilot requires the pretrained COCO_V1 checkpoint")

    run_root = _create_unique_run_root(output_root)
    condition_runs: list[dict[str, Any]] = []
    experiments: list[tuple[str, DetectionExperiment, dict[str, Any]]] = []
    validation_tokens: tuple[str, ...] | None = None

    for strategy in AUDITED_CHILD_SHA256:
        condition = curation["conditions"][strategy]
        run_dir = run_root / _condition_slug(strategy)
        run_dir.mkdir(exist_ok=False)
        curation_provenance = {
            "strategy": strategy,
            "budget": PILOT_BUDGET,
            "seed": PILOT_SEED,
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
        })

    model_fingerprint: str | None = None
    for strategy, experiment, _ in experiments:
        if pretrained_checkpoint_provenance(experiment.config.model) != weight_provenance:
            raise ValueError("Cached pretrained checkpoint changed during preflight")
        initialize_seed(PILOT_SEED, mode=baseline.reproducibility_mode)
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
            if pretrained_checkpoint_provenance(experiment.config.model) != weight_provenance:
                raise ValueError("Cached pretrained checkpoint changed before model construction")
            initialize_seed(PILOT_SEED, mode=baseline.reproducibility_mode)
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


def _validate_baseline_config(config: ExperimentConfig) -> None:
    expected = {
        "batch_size": 1,
        "num_workers": 0,
        "seed": PILOT_SEED,
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
) -> dict[str, Any]:
    run_manifest_path = curation_dir / "experiment_manifest.json"
    run_manifest_sha256 = _sha256_file(run_manifest_path)
    if run_manifest_sha256 != AUDITED_RUN_SHA256:
        raise ValueError("Run-level curation manifest hash differs from the independent audit")
    run_manifest = _read_json_object(run_manifest_path)
    if run_manifest.get("train_sample_count") != 242 or run_manifest.get("test_evaluated") is not False:
        raise ValueError("Curation run manifest does not describe the approved train-only artifact")

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
    conditions: dict[str, Any] = {}
    for strategy, expected_digest in AUDITED_CHILD_SHA256.items():
        stem = f"{strategy}_n60_seed42"
        manifest_path = curation_dir / f"{stem}_manifest.json"
        selected_tokens_path = curation_dir / f"{stem}_selected.json"
        manifest = _read_json_object(manifest_path)
        actual_digest = _canonical_manifest_digest(manifest)
        if actual_digest != manifest.get("manifest_sha256") or actual_digest != expected_digest:
            raise ValueError(f"{strategy} child manifest hash does not match the independent audit")
        if (
            manifest.get("selection_strategy") != strategy
            or manifest.get("budget") != PILOT_BUDGET
            or manifest.get("selected_count") != PILOT_BUDGET
            or manifest.get("seed") != PILOT_SEED
            or manifest.get("dataset_version") != dataset_version
            or manifest.get("split_manifest_digest") != split_manifest["manifest_digest"]
            or manifest.get("requested_count_satisfied") is not True
        ):
            raise ValueError(f"{strategy} child manifest does not match the approved condition")
        for key, expected_hash in expected_source_hashes.items():
            if manifest.get(key) != expected_hash:
                raise ValueError(f"{strategy} child manifest source hash mismatch: {key}")
        selected_tokens = manifest.get("selected_sample_tokens")
        if (
            not isinstance(selected_tokens, list)
            or len(selected_tokens) != PILOT_BUDGET
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
        if manifest.get("scene_quota_policy") != SCENE_QUOTA_POLICY:
            raise ValueError(f"{strategy} uses a different scene-quota policy")
        per_scene = manifest.get("per_scene_sample_counts")
        scene_quotas = manifest.get("scene_quotas")
        actual_scene_counts = dict(sorted(Counter(
            image_by_sample[token]["scene_token"] for token in selected_tokens
        ).items()))
        if (
            per_scene != actual_scene_counts
            or per_scene != scene_quotas
            or set(per_scene.values()) != {10}
        ):
            raise ValueError(f"{strategy} does not satisfy the audited common scene quotas")
        annotation_counts = manifest.get("selected_annotation_counts")
        if not isinstance(annotation_counts, dict):
            raise ValueError(f"{strategy} is missing annotation-count provenance")
        for token in selected_tokens:
            if token not in image_by_sample:
                raise ValueError(f"{strategy} contains an unknown sample token: {token}")
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
        "seed 42, six epochs, batch size 1, AdamW, StepLR, AMP, preprocessing, and the "
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
        "- Reproduction command: `python -m scenarioforge.training.cli curation-pilot` (defaults to the audited manifests and Part 3D.3 baseline config).",
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
