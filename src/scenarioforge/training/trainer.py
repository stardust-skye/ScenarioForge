"""Memory-conscious training loop for detector adapters."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from scenarioforge.training.checkpointing import load_checkpoint, save_checkpoint, write_json
from scenarioforge.training.config import ExperimentConfig
from scenarioforge.training.evaluator import DetectionEvaluator
from scenarioforge.training.model_factory import DetectionModelAdapter, validate_loss_dictionary


class DetectionTrainer:
    """Train on a train loader and select checkpoints using validation only."""

    def __init__(
        self,
        model: DetectionModelAdapter,
        config: ExperimentConfig,
        *,
        device: torch.device,
        run_dir: Path,
        metadata: dict[str, Any],
        model_label_to_name: dict[int, str],
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=config.mixed_precision and device.type == "cuda",
        )
        self.evaluator = DetectionEvaluator(model_label_to_name=model_label_to_name)
        self.start_epoch = 0
        self.global_step = 0
        self.best_validation_ap: float | None = None
        self.best_validation_epoch: int | None = None
        self.stale_epochs = 0
        self.history: list[dict[str, Any]] = []
        self._active_train_loader = None
        self._restored_train_loader_generator_state: Tensor | None = None

    def train_one_batch(
        self,
        images: tuple[Tensor, ...] | list[Tensor],
        targets: tuple[dict[str, Tensor], ...] | list[dict[str, Tensor]],
    ) -> dict[str, float]:
        """Perform one optimizer step for a forward/backward smoke test."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            losses = self._compute_loss(images, targets)
            total_loss = sum(losses.values())
            self.scaler.scale(total_loss).backward()
            self._optimizer_step()
        except torch.cuda.OutOfMemoryError as exc:
            self.optimizer.zero_grad(set_to_none=True)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise _memory_error() from exc
        except RuntimeError as exc:
            if self.config.reproducibility_mode == "strict" and "deterministic" in str(exc).casefold():
                raise RuntimeError(
                    "Strict reproducibility cannot continue because an operation lacks "
                    "a deterministic implementation on this runtime."
                ) from exc
            raise
        return {name: float(value.detach().cpu()) for name, value in losses.items()} | {
            "total_loss": float(total_loss.detach().cpu())
        }

    def resume(self, checkpoint_path: Path) -> None:
        if (self.run_dir / "experiment_summary.json").is_file():
            raise ValueError(
                "Completed experiment artifacts are immutable; resume into a new experiment"
            )
        raw_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if not isinstance(raw_checkpoint, dict) or raw_checkpoint.get(
            "checkpoint_version"
        ) != 2:
            raise ValueError(
                "This checkpoint predates complete resume state and cannot be resumed safely"
            )
        checkpoint_epoch = int(raw_checkpoint["epoch"])
        loaded_history = self._read_history()
        history_epochs = [record["epoch"] for record in loaded_history]
        if history_epochs != list(range(1, checkpoint_epoch + 1)):
            raise ValueError(
                "Training history must end at the checkpoint epoch; refusing to overwrite "
                "or silently restart historical results"
            )
        best_epoch = raw_checkpoint.get("best_validation_epoch")
        best_score = raw_checkpoint.get("best_validation_ap")
        if best_score is not None:
            if not isinstance(best_epoch, int) or not 1 <= best_epoch <= checkpoint_epoch:
                raise ValueError("Checkpoint best-validation epoch is inconsistent")
            best_record = loaded_history[best_epoch - 1]
            best_metrics = best_record["validation_metrics"]
            if (
                best_metrics is None
                or best_metrics.get("map_50_95") != best_score
                or not best_record["best_checkpoint_updated"]
            ):
                raise ValueError("Checkpoint best metric does not match training history")
            best_path = self.run_dir / "checkpoint_best.pth"
            if not best_path.is_file():
                raise ValueError("Checkpoint best metric has no matching checkpoint_best.pth")
            best_checkpoint = torch.load(
                best_path, map_location="cpu", weights_only=True
            )
            if (
                best_checkpoint.get("checkpoint_version") != 2
                or best_checkpoint.get("epoch") != best_epoch
                or best_checkpoint.get("best_validation_ap") != best_score
                or best_checkpoint.get("metadata") != raw_checkpoint.get("metadata")
            ):
                raise ValueError("checkpoint_best.pth does not match saved best-metric provenance")
        expected_metadata = {
            key: self.metadata[key]
            for key in (
                "experiment_id",
                "split_manifest_digest",
                "model_label_to_source_id",
                "configuration_digest",
                "provenance_digest",
            )
            if key in self.metadata
        }
        checkpoint = load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
            expected_metadata=expected_metadata,
            restore_rng=True,
        )
        if checkpoint.get("optimizer_state_dict") is None or checkpoint.get(
            "scheduler_state_dict"
        ) is None:
            raise ValueError("Checkpoint is missing optimizer or scheduler state")
        self.start_epoch = checkpoint_epoch
        self.global_step = int(checkpoint["global_step"])
        self.best_validation_ap = checkpoint.get("best_validation_ap")
        self.best_validation_epoch = checkpoint.get("best_validation_epoch")
        self.stale_epochs = int(checkpoint.get("stale_epochs", 0))
        self._restored_train_loader_generator_state = checkpoint.get(
            "train_loader_generator_state"
        )
        self.history = loaded_history

    def fit(self, train_loader, validation_loader) -> list[dict[str, Any]]:
        if self.start_epoch == 0 and any(
            (self.run_dir / name).exists()
            for name in ("training_history.csv", "checkpoint_last.pth")
        ):
            raise ValueError(
                "Existing training artifacts require explicit checkpoint resume; "
                "refusing to overwrite historical results"
            )
        if self.start_epoch >= self.config.epochs:
            raise ValueError(
                f"Checkpoint is at epoch {self.start_epoch}; configured total epochs "
                f"is {self.config.epochs}, so there is no remaining work to resume."
            )
        self._active_train_loader = train_loader
        if self._restored_train_loader_generator_state is not None:
            generator = getattr(train_loader, "generator", None)
            if generator is not None:
                generator.set_state(self._restored_train_loader_generator_state.cpu())
        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_started = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            generator = getattr(train_loader, "generator", None)
            if generator is not None:
                generator.manual_seed((self.config.seed + epoch) % (2**63 - 1))
            self.model.train()
            self._set_backbone_trainable(
                epoch >= self.config.model.freeze_backbone_epochs
            )
            train_losses = self._train_epoch(train_loader)
            should_validate = (
                (epoch + 1) % self.config.validation_interval == 0
                or epoch + 1 == self.config.epochs
            )
            validation = None
            improved = False
            if should_validate:
                validation = self.evaluator.evaluate(
                    self.model,
                    validation_loader,
                    device=self.device,
                    split_name="validation",
                )
                score = validation["map_50_95"]
                if score is None:
                    raise ValueError("Validation mAP is undefined; no ground-truth objects were evaluated")
                improved = self.best_validation_ap is None or (
                    score > self.best_validation_ap + 1e-12
                )
                if improved:
                    self.best_validation_ap = score
                    self.best_validation_epoch = epoch + 1
                    self.stale_epochs = 0
                else:
                    self.stale_epochs += 1
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    peak_memory = torch.cuda.max_memory_allocated(self.device)
                else:
                    peak_memory = None
                epoch_duration = time.perf_counter() - epoch_started
                learning_rate = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()
                record = {
                    "epoch": epoch + 1,
                    "learning_rate": learning_rate,
                    "epoch_duration_seconds": epoch_duration,
                    "cuda_peak_allocated_bytes": peak_memory,
                    "training_loss": train_losses["total_loss"],
                    "training_loss_components": train_losses,
                    "validation_metrics": validation,
                "best_checkpoint_updated": improved,
            }
            self.history.append(record)
            self._write_history()
            if improved:
                self._save("checkpoint_best.pth", epoch + 1)
            self._save("checkpoint_last.pth", epoch + 1)
            if (
                should_validate
                and self.config.early_stopping_patience is not None
                and self.stale_epochs >= self.config.early_stopping_patience
            ):
                break
        if self.best_validation_ap is None:
            raise ValueError("Training completed without a validation checkpoint")
        validation_results = [
            record["validation_metrics"]
            for record in self.history
            if record["validation_metrics"] is not None
        ]
        write_json(self.run_dir / "validation_metrics.json", {
            "split": "validation",
            "evaluations": validation_results,
        })
        best_checkpoint = self.run_dir / "checkpoint_best.pth"
        write_json(self.run_dir / "experiment_summary.json", {
            **self.metadata,
            "completed_epochs": self.history[-1]["epoch"] if self.history else self.start_epoch,
            "global_step": self.global_step,
            "best_validation_map_50_95": self.best_validation_ap,
            "best_validation_epoch": self.best_validation_epoch,
            "best_checkpoint_sha256": _file_sha256(best_checkpoint),
            "training_duration_seconds": sum(
                record["epoch_duration_seconds"] for record in self.history
            ),
            "peak_cuda_memory_allocated_bytes": max(
                (
                    record["cuda_peak_allocated_bytes"]
                    for record in self.history
                    if record["cuda_peak_allocated_bytes"] is not None
                ),
                default=None,
            ),
            "selection_and_checkpoint_frozen": True,
            "test_evaluated": False,
        })
        return self.history

    def _train_epoch(self, data_loader) -> dict[str, float]:
        totals: dict[str, float] = {}
        batch_count = len(data_loader)
        if batch_count < 1:
            raise ValueError("Cannot train on an empty data loader")
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, targets) in enumerate(data_loader):
            try:
                with self._autocast():
                    losses = self._compute_loss(images, targets)
                    loss = sum(losses.values())
                validate_loss_dictionary(losses)
                group_size = min(
                    self.config.gradient_accumulation_steps,
                    batch_count - (batch_index // self.config.gradient_accumulation_steps)
                    * self.config.gradient_accumulation_steps,
                )
                self.scaler.scale(loss / group_size).backward()
                for name, value in losses.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
                totals["total_loss"] = totals.get("total_loss", 0.0) + float(loss.detach().cpu())
                if (
                    (batch_index + 1) % self.config.gradient_accumulation_steps == 0
                    or batch_index + 1 == batch_count
                ):
                    self._optimizer_step()
                    self.optimizer.zero_grad(set_to_none=True)
            except torch.cuda.OutOfMemoryError as exc:
                self.optimizer.zero_grad(set_to_none=True)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                raise _memory_error() from exc
            except RuntimeError as exc:
                if (
                    self.config.reproducibility_mode == "strict"
                    and "deterministic" in str(exc).casefold()
                ):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError(
                        "Strict reproducibility cannot continue because an operation "
                        "lacks a deterministic implementation on this runtime."
                    ) from exc
                raise
        return {name: value / batch_count for name, value in totals.items()}

    def _compute_loss(self, images, targets) -> dict[str, Tensor]:
        images_on_device = [
            image.to(self.device, non_blocking=self.device.type == "cuda") for image in images
        ]
        targets_on_device = [
            {
                key: value.to(self.device, non_blocking=self.device.type == "cuda")
                if isinstance(value, Tensor)
                else value
                for key, value in target.items()
            }
            for target in targets
        ]
        with self._autocast():
            losses = self.model.compute_loss(images_on_device, targets_on_device)
        validate_loss_dictionary(losses)
        return losses

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.config.mixed_precision and self.device.type == "cuda",
        )

    def _optimizer_step(self) -> None:
        if self.config.gradient_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_norm
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.global_step += 1

    def _set_backbone_trainable(self, trainable: bool) -> None:
        setter = getattr(self.model, "set_backbone_trainable", None)
        if setter is not None:
            setter(trainable)

    def _save(self, filename: str, epoch: int) -> Path:
        return save_checkpoint(
            self.run_dir / filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            best_validation_ap=self.best_validation_ap,
            metadata=self.metadata,
            global_step=self.global_step,
            best_validation_epoch=self.best_validation_epoch,
            stale_epochs=self.stale_epochs,
            train_loader_generator_state=(
                self._active_train_loader.generator.get_state()
                if getattr(self._active_train_loader, "generator", None) is not None
                else None
            ),
        )

    def _read_history(self) -> list[dict[str, Any]]:
        history_path = self.run_dir / "training_history.csv"
        if not history_path.is_file():
            return []
        with history_path.open(newline="", encoding="utf-8") as handle:
            return [
                {
                    "epoch": int(row["epoch"]),
                    "learning_rate": float(row["learning_rate"]),
                    "epoch_duration_seconds": (
                        float(row["epoch_duration_seconds"])
                        if row.get("epoch_duration_seconds")
                        else None
                    ),
                    "cuda_peak_allocated_bytes": (
                        int(row["cuda_peak_allocated_bytes"])
                        if row.get("cuda_peak_allocated_bytes")
                        else None
                    ),
                    "training_loss": float(row["training_loss"]),
                    "training_loss_components": json.loads(
                        row["training_loss_components"]
                    ),
                    "validation_metrics": (
                        json.loads(row["validation_metrics"])
                        if row["validation_metrics"]
                        else None
                    ),
                    "best_checkpoint_updated": (
                        row["best_checkpoint_updated"].casefold() == "true"
                    ),
                }
                for row in csv.DictReader(handle)
            ]

    def _write_history(self) -> None:
        path = self.run_dir / "training_history.csv"
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "epoch",
                    "learning_rate",
                    "epoch_duration_seconds",
                    "cuda_peak_allocated_bytes",
                    "training_loss",
                    "training_loss_components",
                    "validation_metrics",
                    "best_checkpoint_updated",
                ),
            )
            writer.writeheader()
            for record in self.history:
                writer.writerow({
                    **record,
                    "training_loss_components": _json_cell(record["training_loss_components"]),
                    "validation_metrics": _json_cell(record["validation_metrics"]),
                })
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _json_cell(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True)


def _memory_error() -> RuntimeError:
    return RuntimeError(
        "CUDA out of memory during training. No automatic CPU fallback or "
        "configuration reduction was applied. Try a smaller explicit image_size, "
        "batch_size=1, or gradient accumulation."
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
