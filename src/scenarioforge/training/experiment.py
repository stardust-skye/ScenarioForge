"""Dataset-integrated experiment setup and artifact orchestration."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from scenarioforge.training.checkpointing import load_checkpoint, save_checkpoint, write_json
from scenarioforge.training.config import ExperimentConfig
from scenarioforge.training.model_factory import DetectionModelAdapter, create_detection_model
from scenarioforge.training.seed import (
    capture_rng_state,
    environment_metadata,
    initialize_seed,
    restore_rng_state,
)
from scenarioforge.training.trainer import DetectionTrainer
from scenarioforge.vision_data.dataloader import (
    make_detection_dataloader,
    make_split_datasets,
    select_training_subset,
)
from scenarioforge.vision_data.dataset import load_dataset_version
from scenarioforge.vision_data.splits import load_split_manifest
from scenarioforge.vision_data.statistics import load_csv_rows
from scenarioforge.vision_data.transforms import build_eval_transforms, build_train_transforms


class DetectionExperiment:
    """Prepare data, models, and artifacts while keeping the test set out of fitting."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        experiment_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.device = _select_device(config.device)
        initialize_seed(config.seed, mode=config.reproducibility_mode)
        self.run_dir = (
            Path(experiment_dir)
            if experiment_dir is not None
            else config.output_dir / _new_experiment_id()
        )
        self.experiment_id = self.run_dir.name
        self.vision_dir = config.vision_dir
        self.image_rows = load_csv_rows(self.vision_dir / "image_manifest.csv")
        self.dataset_version = load_dataset_version(
            self.vision_dir / "dataset_manifest.json"
        )
        category_mapping_path = self.vision_dir / "category_mapping.json"
        if not category_mapping_path.is_file():
            raise FileNotFoundError(
                "Training requires the authoritative category_mapping.json; "
                "class IDs are not inferred from held-out annotations"
            )
        self.split_manifest = load_split_manifest(
            config.split_manifest,
            image_rows=self.image_rows,
            dataset_version=self.dataset_version,
        )
        image_size = config.image_size
        self.transforms = {
            "train": build_train_transforms(image_size),
            "validation": build_eval_transforms(image_size),
            "test": build_eval_transforms(image_size),
        }
        self.datasets = make_split_datasets(
            dataset_root=str(config.dataset_root),
            annotations_csv=str(self.vision_dir / "annotations.csv"),
            image_manifest_csv=str(self.vision_dir / "image_manifest.csv"),
            category_mapping_json=str(self.vision_dir / "category_mapping.json"),
            split_manifest=self.split_manifest,
            transforms=self.transforms,
            splits=("train", "validation"),
        )
        train_dataset = self.datasets["train"]
        selected_tokens = self._selected_training_tokens(train_dataset.sample_tokens)
        if not selected_tokens:
            raise ValueError("Training selection must contain at least one sample")
        self.train_dataset = select_training_subset(
            train_dataset,
            self.split_manifest,
            selected_tokens,
            image_rows=self.image_rows,
        )
        self.selection_audit = train_dataset.training_selection_audit
        if self.selection_audit is None:
            raise RuntimeError("Training sample selection was not audited")
        validation_dataset = self.datasets["validation"]
        if config.validation_sample_limit is not None:
            validation_dataset = Subset(
                validation_dataset,
                range(min(config.validation_sample_limit, len(validation_dataset))),
            )
        self.train_loader = make_detection_dataloader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            seed=config.seed,
            pin_memory=self.device.type == "cuda",
        )
        self.validation_loader = make_detection_dataloader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            seed=config.seed,
            pin_memory=self.device.type == "cuda",
        )
        self.model_label_to_name = self._model_label_names()
        self.model_label_to_source_id = dict(
            train_dataset.model_label_to_source_id
        )
        self.metadata = {
            "experiment_id": self.experiment_id,
            "dataset_version": self.dataset_version,
            "split_manifest_digest": self.split_manifest["manifest_digest"],
            "image_inventory_digest": self.split_manifest["inventory_digest"],
            "model_architecture": config.model.architecture,
            "model_configuration": {
                **config.model.__dict__,
                "foreground_class_count": len(self.model_label_to_source_id),
                "background_label": 0,
            },
            "model_label_to_source_id": self.model_label_to_source_id,
            "training_selection": self.selection_audit,
            "seed": config.seed,
            "reproducibility_mode": config.reproducibility_mode,
        }
        if config.curation_provenance is not None:
            self.metadata["curation_provenance"] = json.loads(
                json.dumps(config.curation_provenance)
            )
        if config.pretrained_checkpoint_provenance is not None:
            self.metadata["pretrained_checkpoint_provenance"] = json.loads(
                json.dumps(config.pretrained_checkpoint_provenance)
            )
        runtime_environment = environment_metadata(self.device)
        self.metadata["runtime_environment"] = runtime_environment
        config_json = json.dumps(
            config.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        self.metadata["configuration_digest"] = hashlib.sha256(
            config_json.encode("utf-8")
        ).hexdigest()
        provenance = {
            key: self.metadata[key]
            for key in (
                "configuration_digest",
                "dataset_version",
                "split_manifest_digest",
                "image_inventory_digest",
                "training_selection",
                "model_label_to_source_id",
                "runtime_environment",
            )
        }
        for key in ("curation_provenance", "pretrained_checkpoint_provenance"):
            if key in self.metadata:
                provenance[key] = self.metadata[key]
        self.metadata["provenance_digest"] = hashlib.sha256(
            json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.model: DetectionModelAdapter | None = None
        self.trainer: DetectionTrainer | None = None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.run_dir / "config.json"
        environment_path = self.run_dir / "environment.json"
        if config_path.is_file():
            saved_config = ExperimentConfig.from_json(config_path)
            if saved_config.to_dict() != config.to_dict():
                raise ValueError("Existing experiment configuration does not match this run")
        else:
            config.write_json(config_path)
        environment_payload = {
            **runtime_environment,
            **self.metadata,
        }
        if environment_path.is_file():
            saved_environment = json.loads(environment_path.read_text(encoding="utf-8"))
            if saved_environment.get("provenance_digest") != self.metadata["provenance_digest"]:
                raise ValueError("Existing experiment provenance does not match this run")
        else:
            write_json(environment_path, environment_payload)

    @classmethod
    def from_config_file(cls, path: Path) -> DetectionExperiment:
        return cls(ExperimentConfig.from_json(path))

    @classmethod
    def from_experiment_dir(cls, path: Path) -> DetectionExperiment:
        path = Path(path)
        config = ExperimentConfig.from_json(path / "config.json")
        return cls(config, experiment_dir=path)

    def inspect_batch(self) -> dict[str, Any]:
        """Return the actual data contract for the first train batch."""
        images, targets = next(iter(self.train_loader))
        per_image = [
            {
                "image_shape": list(image.shape),
                "image_dtype": str(image.dtype),
                "image_device": str(image.device),
                "target_keys": sorted(target),
                "boxes_shape": list(target["boxes"].shape),
                "boxes_dtype": str(target["boxes"].dtype),
                "labels_shape": list(target["labels"].shape),
                "labels_dtype": str(target["labels"].dtype),
                "box_count": len(target["boxes"]),
            }
            for image, target in zip(images, targets)
        ]
        return {
            "batch_size": len(images),
            "device": self.device.type,
            "images": per_image,
            "cuda_allocated_bytes": (
                torch.cuda.memory_allocated(self.device)
                if self.device.type == "cuda"
                else 0
            ),
            "cuda_reserved_bytes": (
                torch.cuda.memory_reserved(self.device)
                if self.device.type == "cuda"
                else 0
            ),
        }

    def create_model(self) -> DetectionModelAdapter:
        if self.model is None:
            self.model = create_detection_model(
                self.config.model,
                self.model_label_to_source_id,
            ).to(self.device)
        return self.model

    def create_trainer(self) -> DetectionTrainer:
        if self.trainer is None:
            self.trainer = DetectionTrainer(
                self.create_model(),
                self.config,
                device=self.device,
                run_dir=self.run_dir,
                metadata=self.metadata,
                model_label_to_name=self.model_label_to_name,
            )
        return self.trainer

    def run_one_batch_smoke_test(self) -> dict[str, Any]:
        """Smoke-test a disposable model without changing the experiment trajectory."""
        rng_state = capture_rng_state()
        loader_generator_state = self.train_loader.generator.get_state()
        try:
            model = create_detection_model(
                self.config.model, self.model_label_to_source_id
            ).to(self.device)
            trainer = DetectionTrainer(
                model,
                self.config,
                device=self.device,
                run_dir=self.run_dir,
                metadata=self.metadata,
                model_label_to_name=self.model_label_to_name,
            )
            images, targets = next(iter(self.train_loader))
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            losses = trainer.train_one_batch(images, targets)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                peak_memory = torch.cuda.max_memory_allocated(self.device)
            else:
                peak_memory = None
            checkpoint_path = self.run_dir / "checkpoint_smoke.pth"
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=0,
                best_validation_ap=None,
                metadata=self.metadata,
                global_step=trainer.global_step,
                train_loader_generator_state=loader_generator_state,
            )
            loaded = load_checkpoint(
                checkpoint_path,
                model=model,
                map_location=self.device,
            )
            if loaded["metadata"] != self.metadata:
                raise ValueError("Checkpoint metadata did not round-trip")
            return {
                "losses": losses,
                "checkpoint": str(checkpoint_path),
                "checkpoint_reload_verified": True,
                "device": str(self.device),
                "cuda_peak_allocated_bytes": peak_memory,
            }
        finally:
            restore_rng_state(rng_state)
            self.train_loader.generator.set_state(loader_generator_state)

    def train(self, *, resume_from: Path | None = None) -> list[dict[str, Any]]:
        trainer = self.create_trainer()
        if resume_from is not None:
            trainer.resume(resume_from)
        return trainer.fit(self.train_loader, self.validation_loader)

    def evaluate_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        split_name: str,
        test_authorization: Path | None = None,
    ) -> dict[str, Any]:
        if split_name not in ("validation", "test"):
            raise ValueError("Evaluation split must be 'validation' or 'test'")
        if split_name == "test":
            summary_path = self.run_dir / "experiment_summary.json"
            if not summary_path.is_file():
                raise ValueError("Test evaluation is blocked until training has completed")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not summary.get("selection_and_checkpoint_frozen"):
                raise ValueError("Test evaluation requires a frozen experiment checkpoint")
            if summary.get("test_evaluated"):
                raise ValueError("Final test evaluation has already been recorded for this experiment")
            best_checkpoint = self.run_dir / "checkpoint_best.pth"
            if Path(checkpoint_path).resolve() != best_checkpoint.resolve():
                raise ValueError("Test evaluation must use this run's checkpoint_best.pth")
            checkpoint_digest = _file_sha256(best_checkpoint)
            if summary.get("best_checkpoint_sha256") != checkpoint_digest:
                raise ValueError("Selected checkpoint changed after validation-based selection")
            self._validate_test_authorization(test_authorization, checkpoint_digest)
            write_json(
                self.run_dir / "test_evaluation_authorization.json",
                json.loads(Path(test_authorization).read_text(encoding="utf-8")),
            )
        model = self.create_model()
        checkpoint = load_checkpoint(
            checkpoint_path,
            model=model,
            map_location=self.device,
            expected_metadata={
                key: self.metadata[key]
                for key in (
                    "experiment_id",
                    "split_manifest_digest",
                    "model_label_to_source_id",
                    "configuration_digest",
                    "provenance_digest",
                )
            },
        )
        dataset = self.datasets.get(split_name)
        if dataset is None:
            dataset = make_split_datasets(
                dataset_root=str(self.config.dataset_root),
                annotations_csv=str(self.vision_dir / "annotations.csv"),
                image_manifest_csv=str(self.vision_dir / "image_manifest.csv"),
                category_mapping_json=str(self.vision_dir / "category_mapping.json"),
                split_manifest=self.split_manifest,
                transforms=self.transforms,
                splits=(split_name,),
            )[split_name]
            self.datasets[split_name] = dataset
        if split_name == "validation" and self.config.validation_sample_limit is not None:
            dataset = Subset(
                dataset,
                range(min(self.config.validation_sample_limit, len(dataset))),
            )
        loader = make_detection_dataloader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            seed=self.config.seed,
            pin_memory=self.device.type == "cuda",
        )
        from scenarioforge.training.evaluator import DetectionEvaluator

        metrics = DetectionEvaluator(
            model_label_to_name=self.model_label_to_name
        ).evaluate(
            model,
            loader,
            device=self.device,
            split_name=split_name,
        )
        write_json(self.run_dir / f"{split_name}_metrics.json", metrics)
        if split_name == "test":
            summary["test_evaluated"] = True
            write_json(self.run_dir / "experiment_summary.json", summary)
        return metrics

    def _validate_test_authorization(
        self, authorization_path: Path | None, checkpoint_digest: str
    ) -> None:
        if authorization_path is None:
            raise ValueError(
                "Test evaluation requires an explicit --test-authorization JSON file"
            )
        payload = json.loads(Path(authorization_path).read_text(encoding="utf-8"))
        required = (
            "experiment_id",
            "checkpoint_sha256",
            "purpose",
            "authorized_by",
            "authorized_at",
            "provenance_digest",
            "split_manifest_digest",
        )
        if not isinstance(payload, dict) or any(
            not isinstance(payload.get(key), str) or not payload[key].strip()
            for key in required
        ):
            raise ValueError(
                "Test authorization must include experiment_id, checkpoint_sha256, "
                "purpose, authorized_by, and authorized_at"
            )
        if payload["experiment_id"] != self.experiment_id:
            raise ValueError("Test authorization experiment_id does not match this run")
        if payload["checkpoint_sha256"] != checkpoint_digest:
            raise ValueError(
                "Test authorization checkpoint_sha256 does not match the selected checkpoint"
            )
        if payload["provenance_digest"] != self.metadata["provenance_digest"]:
            raise ValueError("Test authorization provenance_digest does not match this experiment")
        if payload["split_manifest_digest"] != self.metadata["split_manifest_digest"]:
            raise ValueError("Test authorization split_manifest_digest does not match this split")
        try:
            authorized_at = datetime.fromisoformat(
                payload["authorized_at"].replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                "Test authorization authorized_at must be an ISO-8601 timestamp"
            ) from exc
        if authorized_at.tzinfo is None:
            raise ValueError("Test authorization authorized_at must include a timezone")

    def _selected_training_tokens(self, train_tokens: tuple[str, ...]) -> list[str]:
        if self.config.training_sample_tokens_file is not None:
            payload = json.loads(
                self.config.training_sample_tokens_file.read_text(encoding="utf-8")
            )
            if not isinstance(payload, list) or not all(
                isinstance(token, str) for token in payload
            ):
                raise ValueError("Training sample token file must contain a JSON string array")
            return payload
        if self.config.training_sample_limit is not None:
            return list(train_tokens[: self.config.training_sample_limit])
        return list(train_tokens)

    def _model_label_names(self) -> dict[int, str]:
        mapping = json.loads(
            (self.vision_dir / "category_mapping.json").read_text(encoding="utf-8")
        )
        source_names = {
            int(item["class_id"]): str(item["name"])
            for item in mapping["classes"]
        }
        return {
            label: source_names[source_id]
            for label, source_id in self.datasets["train"].model_label_to_source_id.items()
        }


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback was applied")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _new_experiment_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
