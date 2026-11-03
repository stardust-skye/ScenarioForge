from __future__ import annotations

import inspect
import csv
import json
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from scenarioforge.training.checkpointing import load_checkpoint
from scenarioforge.training.config import ExperimentConfig, ModelConfig
from scenarioforge.training.evaluator import DetectionEvaluator
from scenarioforge.training.experiment import DetectionExperiment
from scenarioforge.training.model_factory import (
    create_detection_model,
    validate_loss_dictionary,
)
from scenarioforge.training.seed import initialize_seed
from scenarioforge.training.trainer import DetectionTrainer
from scenarioforge.vision_data.dataloader import detection_collate_fn, make_detection_dataloader
from scenarioforge.vision_data.splits import (
    build_split_manifest,
    validate_training_selection,
)


class _TinyDetector(nn.Module):
    architecture = "tiny-test"

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.model_label_to_source_id = {1: 1, 2: 2}

    def compute_loss(self, _images, _targets):
        del _images, _targets
        return {"loss_classifier": self.weight.square()}

    def predict(self, images):
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=image.device),
                "labels": torch.tensor([1], dtype=torch.int64, device=image.device),
                "scores": torch.tensor([0.9], device=image.device),
            }
            for image in images
        ]


class _StochasticTinyDetector(_TinyDetector):
    def compute_loss(self, images, _targets):
        import numpy as np

        factor = (
            images[0].mean()
            + random.random()
            + float(np.random.random())
            + float(torch.rand(()))
            + 1.0
        )
        return {"loss_classifier": self.weight.square() * factor}


class _IndexedDetectionDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        image = torch.full((3, 32, 32), float(index), dtype=torch.float32)
        target = _batch()[1][0].copy()
        target["image_id"] = torch.tensor(index + 1, dtype=torch.int64)
        return image, target


class _OneBatchLoader:
    def __init__(self, batch):
        self.batch = batch
        self.generator = torch.Generator()
        self.generator.manual_seed(0)

    def __len__(self):
        return 1

    def __iter__(self):
        return iter((self.batch,))


class _InterruptAfterFirstEpoch:
    def __init__(self, loader):
        self.loader = loader
        self.generator = loader.generator
        self.iterations = 0

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.iterations += 1
        if self.iterations == 2:
            raise RuntimeError("simulated process interruption")
        return iter(self.loader)


def _config(tmp_path, **overrides):
    values = {
        "output_dir": tmp_path,
        "device": "cpu",
        "epochs": 1,
        "mixed_precision": False,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def _batch():
    image = torch.zeros((3, 32, 32), dtype=torch.float32)
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "image_id": torch.tensor(1, dtype=torch.int64),
        "area": torch.tensor([100.0], dtype=torch.float32),
        "iscrowd": torch.tensor([0], dtype=torch.int64),
    }
    return (image,), (target,)


def test_config_round_trip_and_validation(tmp_path):
    config = _config(tmp_path, image_size=(320, 640))
    path = config.write_json(tmp_path / "config.json")
    loaded = ExperimentConfig.from_json(path)
    assert loaded.to_dict() == config.to_dict()
    with pytest.raises(ValueError, match="Unknown"):
        ExperimentConfig.from_dict({"unknown_option": True})
    with pytest.raises(ValueError, match="non-negative"):
        ModelConfig(freeze_backbone_epochs=-1)


def test_seed_initializes_python_numpy_torch_and_loader_generator():
    import random
    import numpy as np

    first_generator = initialize_seed(1234)
    first = (random.random(), float(np.random.random()), float(torch.rand(())))
    second_generator = initialize_seed(1234)
    second = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert first == second
    assert torch.equal(first_generator.get_state(), second_generator.get_state())


def test_reproducibility_modes_are_explicit_and_recordable():
    initialize_seed(123, mode="strict")
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    initialize_seed(123, mode="best_effort")
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    with pytest.raises(ValueError, match="reproducibility_mode"):
        _config(Path("."), reproducibility_mode="unknown")
    initialize_seed(123, mode="strict")


def test_model_construction_maps_ten_foreground_classes_and_forward_pass():
    mapping = {label: label * 10 for label in range(1, 11)}
    model = create_detection_model(ModelConfig(pretrained_weights=None), mapping)
    assert model.detector.roi_heads.box_predictor.cls_score.out_features == 11
    assert model.model_label_to_source_id == mapping
    image = torch.rand((3, 320, 320))
    target = {
        "boxes": torch.tensor([[20.0, 20.0, 80.0, 80.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    model.train()
    losses = model.compute_loss((image,), (target,))
    validate_loss_dictionary(losses)
    model.eval()
    prediction = model.predict((image,))
    assert prediction[0]["boxes"].ndim == 2
    assert prediction[0]["labels"].dtype == torch.int64


def test_pretrained_model_replaces_coco_predictor(monkeypatch):
    import torchvision.models.detection as detection
    from types import SimpleNamespace

    original_predictor = nn.Linear(8, 91)
    detector = nn.Module()
    detector.backbone = nn.Identity()
    detector.roi_heads = nn.Module()
    detector.roi_heads.box_predictor = nn.Module()
    detector.roi_heads.box_predictor.cls_score = original_predictor
    captured = {}
    weights = object()

    class FakeWeights:
        COCO_V1 = weights

    def fake_constructor(**kwargs):
        captured.update(kwargs)
        return detector

    monkeypatch.setattr(
        detection, "fasterrcnn_mobilenet_v3_large_320_fpn", fake_constructor
    )
    monkeypatch.setattr(
        detection, "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights", FakeWeights
    )
    model = create_detection_model(
        ModelConfig(pretrained_weights="COCO_V1"),
        {label: label for label in range(1, 11)},
    )
    assert captured["weights"] is weights
    assert captured["num_classes"] is None
    assert model.detector.roi_heads.box_predictor.cls_score.out_features == 11
    assert model.detector.roi_heads.box_predictor.cls_score is not original_predictor


def test_frozen_backbone_batch_norm_stays_in_eval_and_unfreezes():
    model = create_detection_model(
        ModelConfig(pretrained_weights=None), {1: 1, 2: 2}
    )
    model.train()
    model.set_backbone_trainable(False)
    batch_norms = [
        module
        for module in model.detector.backbone.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    assert batch_norms
    assert all(not module.training for module in batch_norms)
    assert all(not parameter.requires_grad for parameter in model.detector.backbone.parameters())
    model.train()
    assert all(not module.training for module in batch_norms)
    model.set_backbone_trainable(True)
    assert all(module.training for module in batch_norms)
    assert all(parameter.requires_grad for parameter in model.detector.backbone.parameters())


def test_faster_rcnn_accepts_an_empty_target():
    model = create_detection_model(
        ModelConfig(pretrained_weights=None), {1: 1, 2: 2}
    )
    image = torch.rand((3, 32, 32))
    target = {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.int64),
    }
    model.train()
    losses = model.compute_loss((image,), (target,))
    assert losses
    assert all(torch.isfinite(loss) for loss in losses.values())
    sum(losses.values()).backward()


def test_loss_dictionary_rejects_invalid_values():
    validate_loss_dictionary({"loss": torch.tensor(1.0)})
    with pytest.raises(ValueError, match="non-empty"):
        validate_loss_dictionary({})
    with pytest.raises(ValueError, match="scalar"):
        validate_loss_dictionary({"loss": torch.ones(2)})
    with pytest.raises(ValueError, match="not finite"):
        validate_loss_dictionary({"loss": torch.tensor(float("nan"))})


def test_metrics_report_counts_and_undefined_class_ap_as_null():
    evaluator = DetectionEvaluator(model_label_to_name={1: "car", 2: "truck"})
    metrics = evaluator.evaluate(
        _TinyDetector(),
        [_batch()],
        device=torch.device("cpu"),
        split_name="validation",
    )
    assert metrics["split"] == "validation"
    assert metrics["map_50"] == pytest.approx(1.0)
    assert metrics["map_75"] == pytest.approx(1.0)
    assert metrics["per_class"]["car"]["average_precision"] == pytest.approx(1.0)
    assert metrics["per_class"]["truck"]["average_precision"] is None
    assert metrics["per_class"]["car"]["ground_truth_objects"] == 1
    assert metrics["per_class"]["car"]["predictions"] == 1
    assert metrics["precision_iou50"] == pytest.approx(1.0)
    assert metrics["recall_iou50"] == pytest.approx(1.0)


def test_metric_empty_predictions_and_targets_preserve_undefined_values():
    class NoPredictionDetector(nn.Module):
        def predict(self, images):
            return [
                {
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                    "scores": torch.empty((0,), dtype=torch.float32),
                }
                for _ in images
            ]

    empty_target = {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.int64),
        "image_id": torch.tensor(1, dtype=torch.int64),
        "area": torch.empty((0,), dtype=torch.float32),
        "iscrowd": torch.empty((0,), dtype=torch.int64),
    }
    evaluator = DetectionEvaluator(model_label_to_name={1: "car"})
    no_objects = evaluator.evaluate(
        NoPredictionDetector(),
        [((torch.zeros((3, 32, 32)),), (empty_target,))],
        device=torch.device("cpu"),
        split_name="validation",
    )
    assert no_objects["map_50_95"] is None
    assert no_objects["precision_iou50"] is None
    assert no_objects["recall_iou50"] is None

    missed_object = evaluator.evaluate(
        NoPredictionDetector(),
        [_batch()],
        device=torch.device("cpu"),
        split_name="validation",
    )
    assert missed_object["map_50_95"] == pytest.approx(0.0)
    assert missed_object["precision_iou50"] is None
    assert missed_object["recall_iou50"] == pytest.approx(0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_amp_training_step_on_cuda(tmp_path):
    trainer = DetectionTrainer(
        _TinyDetector(),
        _config(Path("."), mixed_precision=True),
        device=torch.device("cuda"),
        run_dir=tmp_path / "amp-training-test",
        metadata={"experiment_id": "amp-test"},
        model_label_to_name={1: "car", 2: "truck"},
    )
    losses = trainer.train_one_batch(*_batch())
    assert losses["total_loss"] > 0
    assert trainer.global_step == 1


def test_smoke_step_uses_disposable_model_and_restores_rng(tmp_path, monkeypatch):
    from scenarioforge.training import experiment as experiment_module
    from scenarioforge.training.seed import capture_rng_state

    initialize_seed(456, mode="strict")
    experiment = DetectionExperiment.__new__(DetectionExperiment)
    experiment.config = _config(
        tmp_path, model=ModelConfig(pretrained_weights=None)
    )
    experiment.device = torch.device("cpu")
    experiment.run_dir = tmp_path / "smoke"
    experiment.metadata = {"experiment_id": "smoke-test"}
    experiment.model_label_to_source_id = {1: 1, 2: 2}
    experiment.model_label_to_name = {1: "car", 2: "truck"}
    experiment.train_loader = _OneBatchLoader(_batch())
    experiment.model = None
    experiment.trainer = None
    monkeypatch.setattr(
        experiment_module,
        "create_detection_model",
        lambda _config, _mapping: _TinyDetector(),
    )
    rng_before = capture_rng_state()
    loader_state_before = experiment.train_loader.generator.get_state()

    result = experiment.run_one_batch_smoke_test()

    rng_after = capture_rng_state()
    assert result["checkpoint_reload_verified"]
    assert experiment.model is None and experiment.trainer is None
    assert rng_after["python"] == rng_before["python"]
    assert rng_after["numpy"] == rng_before["numpy"]
    assert torch.equal(rng_after["torch_cpu"], rng_before["torch_cpu"])
    assert torch.equal(experiment.train_loader.generator.get_state(), loader_state_before)


def test_training_step_checkpoint_and_resume(tmp_path):
    model = _TinyDetector()
    config = _config(tmp_path, epochs=2)
    metadata = {
        "experiment_id": "test-run",
        "split_manifest_digest": "digest",
        "model_label_to_source_id": {1: 1, 2: 2},
    }
    trainer = DetectionTrainer(
        model, config, device=torch.device("cpu"), run_dir=tmp_path,
        metadata=metadata, model_label_to_name={1: "car", 2: "truck"},
    )
    losses = trainer.train_one_batch(*_batch())
    assert losses["total_loss"] > 0
    interrupting_loader = _InterruptAfterFirstEpoch(
        _OneBatchLoader(_batch())
    )
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        trainer.fit(interrupting_loader, [_batch()])
    history = trainer.history
    assert history[0]["validation_metrics"]["split"] == "validation"
    assert history[0]["epoch_duration_seconds"] > 0
    assert history[0]["cuda_peak_allocated_bytes"] is None
    checkpoint_path = tmp_path / "checkpoint_last.pth"
    resumed = DetectionTrainer(
        _TinyDetector(), _config(tmp_path, epochs=2),
        device=torch.device("cpu"), run_dir=tmp_path,
        metadata=metadata, model_label_to_name={1: "car", 2: "truck"},
    )
    resumed.resume(checkpoint_path)
    assert resumed.start_epoch == 1
    assert len(resumed.history) == 1
    resumed_history = resumed.fit(_OneBatchLoader(_batch()), [_batch()])
    assert len(resumed_history) == 2
    assert resumed_history[-1]["epoch"] == 2
    assert (tmp_path / "checkpoint_best.pth").is_file()
    assert json.loads((tmp_path / "experiment_summary.json").read_text())[
        "selection_and_checkpoint_frozen"
    ]
    summary = json.loads((tmp_path / "experiment_summary.json").read_text())
    assert summary["training_duration_seconds"] > 0
    assert summary["peak_cuda_memory_allocated_bytes"] is None
    weight_before_mismatch = resumed.model.weight.detach().clone()
    with pytest.raises(ValueError, match="split_manifest_digest"):
        load_checkpoint(
            checkpoint_path,
            model=resumed.model,
            expected_metadata={"split_manifest_digest": "wrong-digest"},
        )
    assert torch.equal(resumed.model.weight, weight_before_mismatch)
    loaded = load_checkpoint(checkpoint_path, model=resumed.model)
    assert loaded["metadata"] == metadata


def test_uninterrupted_and_interrupted_resume_are_deterministically_equivalent(tmp_path):
    from scenarioforge.training.seed import capture_rng_state

    config = _config(
        tmp_path / "config",
        epochs=4,
        scheduler_step_size=2,
        scheduler_gamma=0.5,
    )
    metadata = {
        "experiment_id": "resume-equivalence",
        "split_manifest_digest": "split",
        "model_label_to_source_id": {1: 1, 2: 2},
        "configuration_digest": "config",
        "provenance_digest": "provenance",
    }

    initialize_seed(987, mode="strict")
    uninterrupted_model = _StochasticTinyDetector()
    uninterrupted_loader = make_detection_dataloader(
        _IndexedDetectionDataset(), batch_size=1, shuffle=True, seed=987
    )
    uninterrupted = DetectionTrainer(
        uninterrupted_model,
        config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "uninterrupted",
        metadata=metadata,
        model_label_to_name={1: "car", 2: "truck"},
    )
    uninterrupted_history = uninterrupted.fit(uninterrupted_loader, [_batch()])
    uninterrupted_rng = capture_rng_state()

    initialize_seed(987, mode="strict")
    interrupted_model = _StochasticTinyDetector()
    interrupted_loader = make_detection_dataloader(
        _IndexedDetectionDataset(), batch_size=1, shuffle=True, seed=987
    )
    interrupted = DetectionTrainer(
        interrupted_model,
        config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "resumed",
        metadata=metadata,
        model_label_to_name={1: "car", 2: "truck"},
    )
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        interrupted.fit(
            _InterruptAfterFirstEpoch(interrupted_loader), [_batch()]
        )
    assert interrupted.global_step == 4
    checkpoint_path = tmp_path / "resumed" / "checkpoint_last.pth"
    saved_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert saved_checkpoint["epoch"] == 1
    assert saved_checkpoint["global_step"] == 4
    assert saved_checkpoint["best_validation_epoch"] == 1

    mismatched_model = _StochasticTinyDetector()
    mismatched_model.weight.data.fill_(5.0)
    mismatched = DetectionTrainer(
        mismatched_model,
        config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "resumed",
        metadata={**metadata, "configuration_digest": "wrong-config"},
        model_label_to_name={1: "car", 2: "truck"},
    )
    weight_before_mismatch = mismatched.model.weight.detach().clone()
    with pytest.raises(ValueError, match="configuration_digest"):
        mismatched.resume(checkpoint_path)
    assert torch.equal(mismatched.model.weight, weight_before_mismatch)

    initialize_seed(987, mode="strict")
    resumed_model = _StochasticTinyDetector()
    resumed_loader = make_detection_dataloader(
        _IndexedDetectionDataset(), batch_size=1, shuffle=True, seed=987
    )
    resumed = DetectionTrainer(
        resumed_model,
        config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "resumed",
        metadata=metadata,
        model_label_to_name={1: "car", 2: "truck"},
    )
    resumed.resume(checkpoint_path)
    resumed_history = resumed.fit(resumed_loader, [_batch()])

    assert resumed.global_step == uninterrupted.global_step == 16
    assert resumed.start_epoch == 1
    assert resumed.best_validation_epoch == uninterrupted.best_validation_epoch
    assert resumed.scheduler.state_dict() == uninterrupted.scheduler.state_dict()
    for resumed_record, uninterrupted_record in zip(
        resumed_history, uninterrupted_history
    ):
        assert {
            key: value
            for key, value in resumed_record.items()
            if key not in {"epoch_duration_seconds", "cuda_peak_allocated_bytes"}
        } == {
            key: value
            for key, value in uninterrupted_record.items()
            if key not in {"epoch_duration_seconds", "cuda_peak_allocated_bytes"}
        }
    assert torch.equal(resumed.model.weight, uninterrupted.model.weight)
    for parameter_id, full_state in uninterrupted.optimizer.state_dict()["state"].items():
        resumed_state = resumed.optimizer.state_dict()["state"][parameter_id]
        for key, full_value in full_state.items():
            resumed_value = resumed_state[key]
            if isinstance(full_value, torch.Tensor):
                assert torch.equal(full_value, resumed_value)
            else:
                assert full_value == resumed_value
    resumed_rng = capture_rng_state()
    assert resumed_rng["python"] == uninterrupted_rng["python"]
    assert resumed_rng["numpy"] == uninterrupted_rng["numpy"]
    assert torch.equal(resumed_rng["torch_cpu"], uninterrupted_rng["torch_cpu"])
    assert torch.equal(resumed_loader.generator.get_state(), uninterrupted_loader.generator.get_state())

    with pytest.raises(ValueError, match="refusing to overwrite historical results"):
        DetectionTrainer(
            _StochasticTinyDetector(),
            config,
            device=torch.device("cpu"),
            run_dir=tmp_path / "uninterrupted",
            metadata=metadata,
            model_label_to_name={1: "car", 2: "truck"},
        ).fit(uninterrupted_loader, [_batch()])


def test_fit_accepts_train_and_validation_only():
    parameter_names = inspect.signature(DetectionTrainer.fit).parameters
    assert "train_loader" in parameter_names
    assert "validation_loader" in parameter_names
    assert "test_loader" not in parameter_names


def test_training_selection_is_train_only_and_digest_is_recorded():
    image_rows = [
        {
            "image_id": f"sample-{scene}",
            "sample_token": f"sample-{scene}",
            "scene_token": f"scene-{scene}",
            "image_path": f"samples/{scene}.jpg",
            "width": "10",
            "height": "10",
            "image_status": "ok",
        }
        for scene in range(10)
    ]
    manifest = build_split_manifest(image_rows, dataset_version="mini", seed=5)
    selection = validate_training_selection(
        manifest,
        manifest["sample_tokens"]["train"][:1],
        image_rows=image_rows,
        dataset_version="mini",
    )
    assert selection["valid"]
    assert selection["heldout_samples_unchanged"]
    assert len(selection["selection_digest"]) == 64
    with pytest.raises(ValueError, match="non-training"):
        validate_training_selection(
            manifest,
            manifest["sample_tokens"]["validation"][:1],
            image_rows=image_rows,
            dataset_version="mini",
        )


def test_experiment_records_split_inventory_and_selection_digests(tmp_path):
    vision_dir = tmp_path / "vision"
    dataset_root = tmp_path / "dataset"
    vision_dir.mkdir()
    dataset_root.mkdir()
    image_fields = (
        "image_id", "sample_token", "scene_token", "image_path",
        "width", "height", "image_status",
    )
    annotation_fields = (
        "source_annotation_token", "sample_token", "image_id", "scene_token",
        "image_path", "image_width", "image_height", "conversion_status",
        "class_id", "detection_category", "x_min", "y_min", "x_max", "y_max",
    )
    image_rows = []
    annotation_rows = []
    for index in range(3):
        sample = f"sample-{index}"
        image_path = f"{sample}.jpg"
        Image.new("RGB", (32, 24), color=(index, index, index)).save(
            dataset_root / image_path
        )
        image_rows.append({
            "image_id": sample,
            "sample_token": sample,
            "scene_token": f"scene-{index}",
            "image_path": image_path,
            "width": "32",
            "height": "24",
            "image_status": "ok",
        })
        annotation_rows.append({
            "source_annotation_token": f"annotation-{index}",
            "sample_token": sample,
            "image_id": sample,
            "scene_token": f"scene-{index}",
            "image_path": image_path,
            "image_width": "32",
            "image_height": "24",
            "conversion_status": "accepted",
            "class_id": "1",
            "detection_category": "car",
            "x_min": "1",
            "y_min": "1",
            "x_max": "10",
            "y_max": "10",
        })
    for filename, fields, rows in (
        ("image_manifest.csv", image_fields, image_rows),
        ("annotations.csv", annotation_fields, annotation_rows),
    ):
        with (vision_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (vision_dir / "dataset_manifest.json").write_text(
        json.dumps({"dataset_version": "mini"}), encoding="utf-8"
    )
    (vision_dir / "category_mapping.json").write_text(json.dumps({
        "classes": [
            {"class_id": 1, "name": "car"},
            {"class_id": 2, "name": "truck"},
        ]
    }), encoding="utf-8")
    manifest = build_split_manifest(image_rows, dataset_version="mini", seed=5)
    split_path = vision_dir / "scene_split.json"
    split_path.write_text(json.dumps(manifest), encoding="utf-8")

    experiment = DetectionExperiment(ExperimentConfig(
        dataset_root=dataset_root,
        vision_dir=vision_dir,
        split_manifest=split_path,
        output_dir=tmp_path / "experiments",
        training_sample_limit=1,
        validation_sample_limit=1,
        model=ModelConfig(pretrained_weights=None),
        device="cpu",
        reproducibility_mode="best_effort",
    ))
    saved_environment = json.loads(
        (experiment.run_dir / "environment.json").read_text(encoding="utf-8")
    )
    assert saved_environment["split_manifest_digest"] == manifest["manifest_digest"]
    assert saved_environment["image_inventory_digest"] == manifest["inventory_digest"]
    assert saved_environment["training_selection"]["heldout_samples_unchanged"]
    assert saved_environment["training_selection"]["selection_digest"] == (
        experiment.selection_audit["selection_digest"]
    )
    deterministic_settings = saved_environment["hardware"]["deterministic_settings"]
    assert deterministic_settings["algorithms_enabled"]
    assert deterministic_settings["warn_only"] is True
    assert "cudnn_version" in saved_environment["hardware"]
    assert saved_environment["reproducibility_mode"] == "best_effort"
    assert saved_environment["runtime_environment"]["packages"]["torch"] == torch.__version__


def test_training_does_not_instantiate_or_read_test_dataset(tmp_path, monkeypatch):
    vision_dir = tmp_path / "vision"
    dataset_root = tmp_path / "dataset"
    vision_dir.mkdir()
    dataset_root.mkdir()
    image_fields = (
        "image_id", "sample_token", "scene_token", "image_path",
        "width", "height", "image_status",
    )
    annotation_fields = (
        "source_annotation_token", "sample_token", "image_id", "scene_token",
        "image_path", "image_width", "image_height", "conversion_status",
        "class_id", "detection_category", "x_min", "y_min", "x_max", "y_max",
    )
    image_rows = []
    annotation_rows = []
    for index in range(3):
        sample = f"sample-{index}"
        image_path = f"{sample}.jpg"
        image_rows.append({
            "image_id": sample,
            "sample_token": sample,
            "scene_token": f"scene-{index}",
            "image_path": image_path,
            "width": "32",
            "height": "24",
            "image_status": "ok",
        })
        if index < 2:
            Image.new("RGB", (32, 24), color=(index, index, index)).save(
                dataset_root / image_path
            )
            annotation_rows.append({
                "source_annotation_token": f"annotation-{index}",
                "sample_token": sample,
                "image_id": sample,
                "scene_token": f"scene-{index}",
                "image_path": image_path,
                "image_width": "32",
                "image_height": "24",
                "conversion_status": "accepted",
                "class_id": "1",
                "detection_category": "car",
                "x_min": "1",
                "y_min": "1",
                "x_max": "10",
                "y_max": "10",
            })
    annotation_rows.append({
        "source_annotation_token": "heldout-annotation",
        "sample_token": "sample-2",
        "image_id": "sample-2",
        "scene_token": "scene-2",
        "image_path": "sample-2.jpg",
        "image_width": "32",
        "image_height": "24",
        "conversion_status": "accepted",
        "class_id": "not-a-training-label",
        "detection_category": "heldout-only",
        "x_min": "invalid",
        "y_min": "invalid",
        "x_max": "invalid",
        "y_max": "invalid",
    })
    for filename, fields, rows in (
        ("image_manifest.csv", image_fields, image_rows),
        ("annotations.csv", annotation_fields, annotation_rows),
    ):
        with (vision_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (vision_dir / "dataset_manifest.json").write_text(
        json.dumps({"dataset_version": "isolated"}), encoding="utf-8"
    )
    (vision_dir / "category_mapping.json").write_text(json.dumps({
        "classes": [
            {"class_id": 1, "name": "car"},
            {"class_id": 2, "name": "truck"},
        ]
    }), encoding="utf-8")
    manifest = build_split_manifest(
        image_rows,
        dataset_version="isolated",
        scene_assignments={
            "train": ["scene-0"],
            "validation": ["scene-1"],
            "test": ["scene-2"],
        },
    )
    split_path = vision_dir / "scene_split.json"
    split_path.write_text(json.dumps(manifest), encoding="utf-8")
    experiment = DetectionExperiment(ExperimentConfig(
        dataset_root=dataset_root,
        vision_dir=vision_dir,
        split_manifest=split_path,
        output_dir=tmp_path / "experiments",
        training_sample_limit=1,
        validation_sample_limit=1,
        epochs=1,
        model=ModelConfig(pretrained_weights=None),
        device="cpu",
        reproducibility_mode="best_effort",
    ))
    monkeypatch.setattr(experiment, "create_model", lambda: _TinyDetector())
    experiment.train()
    assert set(experiment.datasets) == {"train", "validation"}
    assert "sample-2" not in experiment.train_dataset.dataset.sample_tokens
    with pytest.raises(ValueError, match="authorization"):
        experiment.evaluate_checkpoint(
            experiment.run_dir / "checkpoint_best.pth",
            split_name="test",
        )
    assert "test" not in experiment.datasets
