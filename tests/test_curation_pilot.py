from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from scenarioforge.training.config import ExperimentConfig
from scenarioforge.training.curation_pilot import (
    APPROVED_SEEDS,
    AUDITED_CHILD_SHA256,
    AUDITED_RUN_SHA256,
    _canonical_manifest_digest,
    _create_unique_run_root,
    _validate_baseline_config,
    _validate_curation_inputs,
    _verify_pretrained_checkpoint_provenance,
    run_curation_pilot,
)
from scenarioforge.training.model_factory import pretrained_checkpoint_provenance


ROOT = Path(__file__).resolve().parents[1]
CURATION_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "curation_experiments"
    / "curation_20261103T205532Z_77d0c39b"
)


def _pilot_config() -> ExperimentConfig:
    config = ExperimentConfig.from_json(
        ROOT / "configs" / "object_detection_extended_baseline_part3d3.json"
    )
    return replace(
        config,
        dataset_root=ROOT / config.dataset_root,
        vision_dir=ROOT / config.vision_dir,
        split_manifest=ROOT / config.split_manifest,
    )


def _make_seed_bundle(tmp_path: Path, seed: int) -> Path:
    bundle = tmp_path / f"curation_seed_{seed}"
    shutil.copytree(CURATION_ROOT, bundle)
    parent_path = bundle / "experiment_manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent["seed"] = seed
    parent["selection_seed"] = seed
    parent["training_seed"] = seed
    child_names = list(parent["subset_manifests"])
    for strategy in AUDITED_CHILD_SHA256:
        old_stem = f"{strategy}_n60_seed42"
        new_stem = f"{strategy}_n60_seed{seed}"
        old_manifest_path = bundle / f"{old_stem}_manifest.json"
        old_selected_path = bundle / f"{old_stem}_selected.json"
        manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        manifest["seed"] = seed
        manifest["selection_seed"] = seed
        manifest["training_seed"] = seed
        manifest["manifest_sha256"] = _canonical_manifest_digest(manifest)
        manifest_path = bundle / f"{new_stem}_manifest.json"
        if old_manifest_path != manifest_path:
            old_manifest_path.rename(manifest_path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        selected_path = bundle / f"{new_stem}_selected.json"
        if old_selected_path != selected_path:
            old_selected_path.rename(selected_path)
        child_name = f"{new_stem}_manifest.json"
        if child_name not in child_names:
            child_names.append(child_name)
    parent["subset_manifests"] = child_names
    parent_path.write_text(json.dumps(parent, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def _write_child(bundle: Path, strategy: str, manifest: dict, seed: int = 123) -> None:
    path = bundle / f"{strategy}_n60_seed{seed}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_bundle(bundle: Path, seed: int, config: ExperimentConfig | None = None):
    return _validate_curation_inputs(
        curation_dir=bundle,
        config=config or _pilot_config(),
        repository_root=ROOT,
        selection_seed=seed,
    )


def test_audited_pilot_inputs_pass_independent_preflight():
    config = ExperimentConfig.from_json(
        ROOT / "configs" / "object_detection_extended_baseline_part3d3.json"
    )
    config = ExperimentConfig(
        **{
            **config.to_dict(),
            "dataset_root": ROOT / config.dataset_root,
            "vision_dir": ROOT / config.vision_dir,
            "split_manifest": ROOT / config.split_manifest,
        }
    )
    _validate_baseline_config(config)
    validated = _validate_curation_inputs(
        curation_dir=(
            ROOT
            / "data"
            / "processed"
            / "curation_experiments"
            / "curation_20261103T205532Z_77d0c39b"
        ),
        config=config,
        repository_root=ROOT,
    )

    assert validated["run_manifest_sha256"] == AUDITED_RUN_SHA256
    assert set(validated["conditions"]) == set(AUDITED_CHILD_SHA256)
    assert all(
        len(condition["selected_sample_tokens"]) == 60
        for condition in validated["conditions"].values()
    )
    assert all(
        set(condition["per_scene_sample_counts"].values()) == {10}
        for condition in validated["conditions"].values()
    )


def test_preflight_only_checks_datasets_without_model_or_output_creation(
    tmp_path, monkeypatch
):
    from scenarioforge.training import curation_pilot

    requested_splits = []
    original_make_split_datasets = curation_pilot.make_split_datasets
    original_create_detection_model = curation_pilot.create_detection_model
    model_constructions = []

    def observe_splits(**kwargs):
        requested_splits.append(tuple(kwargs["splits"]))
        return original_make_split_datasets(**kwargs)

    def observe_model_construction(*args, **kwargs):
        model_constructions.append(True)
        return original_create_detection_model(*args, **kwargs)

    monkeypatch.setattr(curation_pilot, "create_detection_model", observe_model_construction)
    monkeypatch.setattr(curation_pilot, "make_split_datasets", observe_splits)
    result = curation_pilot.preflight_curation_pilot(
        baseline_config_path=(
            ROOT / "configs" / "object_detection_extended_baseline_part3d3.json"
        ),
        curation_dir=CURATION_ROOT,
        selection_seed=42,
        training_seed=42,
        repository_root=ROOT,
    )
    assert result["model_constructed"] is True
    assert result["initial_model_state_sha256"]
    assert len(model_constructions) == 4
    assert result["test_dataset_instantiated"] is False
    assert len(result["conditions"]) == 4
    assert {row["validation_sample_count"] for row in result["conditions"]} == {80}
    assert len({row["validation_tokens_sha256"] for row in result["conditions"]}) == 1
    assert all(row["selected_image_count"] == 60 for row in result["conditions"])
    assert requested_splits == [("train", "validation")] * 4
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("seed", APPROVED_SEEDS)
def test_seed_specific_manifests_pass_preflight_for_all_planned_seeds(tmp_path, seed):
    bundle = _make_seed_bundle(tmp_path, seed)
    validated = _validate_bundle(bundle, seed)
    assert set(validated["conditions"]) == set(AUDITED_CHILD_SHA256)
    assert all(
        condition["selection_seed"] == seed
        for condition in validated["conditions"].values()
    )
    assert all(
        len(condition["selected_sample_tokens"]) == 60
        for condition in validated["conditions"].values()
    )


def test_ranked_strategies_may_select_the_same_samples_across_seeds(tmp_path):
    first = _validate_bundle(_make_seed_bundle(tmp_path, 123), 123)
    second = _validate_bundle(_make_seed_bundle(tmp_path, 2027), 2027)
    for strategy in ("frequency", "diversity", "weighted_interest"):
        assert first["conditions"][strategy]["selected_sample_tokens"] == (
            second["conditions"][strategy]["selected_sample_tokens"]
        )


def test_wrong_selection_seed_is_rejected(tmp_path):
    bundle = _make_seed_bundle(tmp_path, 123)
    with pytest.raises(ValueError, match="run manifest"):
        _validate_bundle(bundle, 2027)


def test_wrong_training_seed_is_rejected_before_output_creation(tmp_path):
    with pytest.raises(ValueError, match="training_seed to equal selection_seed"):
        run_curation_pilot(
            baseline_config_path=(
                ROOT / "configs" / "object_detection_extended_baseline_part3d3.json"
            ),
            curation_dir=CURATION_ROOT,
            output_root=tmp_path / "outputs",
            selection_seed=42,
            training_seed=123,
            repository_root=ROOT,
        )
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("budget", 120, "approved condition"),
        ("selection_strategy", "diversity", "approved condition"),
        ("dataset_source_id", "wrong-dataset-source", "approved condition"),
        ("scoring_configuration", {"wrong": "configuration"}, "approved condition"),
        ("split_manifest_digest", "wrong-digest", "approved condition"),
        ("training_scenario_digest_sha256", "wrong-parent", "parent-run scenario"),
        ("per_scene_sample_counts", {"invalid_scene": 60}, "scene quotas"),
        ("scene_quotas", {"invalid_scene": 60}, "scene quotas"),
        ("selected_scene_tokens", ["invalid_scene"], "scene quotas"),
    ],
)
def test_invalid_child_provenance_and_quotas_are_rejected(
    tmp_path, field, value, message
):
    bundle = _make_seed_bundle(tmp_path, 123)
    path = bundle / "random_n60_seed123_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest["manifest_sha256"] = _canonical_manifest_digest(manifest)
    _write_child(bundle, "random", manifest)
    with pytest.raises(ValueError, match=message):
        _validate_bundle(bundle, 123)


def test_modified_child_hash_is_rejected(tmp_path):
    bundle = _make_seed_bundle(tmp_path, 123)
    path = bundle / "random_n60_seed123_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["budget"] = 120
    _write_child(bundle, "random", manifest)
    with pytest.raises(ValueError, match="manifest hash"):
        _validate_bundle(bundle, 123)


@pytest.mark.parametrize("kind", ["validation", "test", "duplicate", "unknown"])
def test_leaked_or_invalid_tokens_are_rejected(tmp_path, kind):
    bundle = _make_seed_bundle(tmp_path, 123)
    path = bundle / "random_n60_seed123_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    selected = list(manifest["selected_sample_tokens"])
    split = json.loads(
        (ROOT / "data" / "processed" / "vision" / "splits" / "scene_split_seed42.json")
        .read_text(encoding="utf-8")
    )
    if kind == "validation":
        selected[0] = split["sample_tokens"]["validation"][0]
    elif kind == "test":
        selected[0] = split["sample_tokens"]["test"][0]
    elif kind == "duplicate":
        selected[0] = selected[1]
    else:
        selected[0] = "unknown-sample-token"
    manifest["selected_sample_tokens"] = selected
    manifest["manifest_sha256"] = _canonical_manifest_digest(manifest)
    _write_child(bundle, "random", manifest)
    (bundle / "random_n60_seed123_selected.json").write_text(
        json.dumps(selected), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        _validate_bundle(bundle, 123)


def test_missing_selected_image_is_rejected(tmp_path):
    bundle = _make_seed_bundle(tmp_path, 123)
    config = replace(_pilot_config(), dataset_root=tmp_path / "missing_dataset")
    with pytest.raises(FileNotFoundError, match="Selected image file is missing"):
        _validate_bundle(bundle, 123, config)


def test_parent_manifest_must_link_all_four_child_manifests(tmp_path):
    bundle = _make_seed_bundle(tmp_path, 123)
    path = bundle / "experiment_manifest.json"
    parent = json.loads(path.read_text(encoding="utf-8"))
    parent["subset_manifests"] = []
    path.write_text(json.dumps(parent), encoding="utf-8")
    with pytest.raises(ValueError, match="does not list all four"):
        _validate_bundle(bundle, 123)


def test_parent_source_hash_mismatch_is_rejected(tmp_path):
    bundle = _make_seed_bundle(tmp_path, 123)
    path = bundle / "experiment_manifest.json"
    parent = json.loads(path.read_text(encoding="utf-8"))
    parent["image_manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(parent), encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        _validate_bundle(bundle, 123)


def test_config_persists_selection_and_training_seeds():
    config = ExperimentConfig(seed=123, selection_seed=123, training_seed=123)
    serialized = ExperimentConfig.from_dict(config.to_dict())
    assert serialized.selection_seed == 123
    assert serialized.training_seed == 123
    assert serialized.seed == 123
    with pytest.raises(ValueError, match="seed and training_seed"):
        ExperimentConfig(seed=42, training_seed=123)


def test_mismatched_pretrained_checkpoint_hashes_are_rejected():
    with pytest.raises(ValueError, match="provenance differs"):
        _verify_pretrained_checkpoint_provenance(
            {"identifier": "COCO_V1", "sha256": "a" * 64},
            {"identifier": "COCO_V1", "sha256": "b" * 64},
        )


def test_canonical_child_manifest_digest_detects_payload_changes(tmp_path):
    manifest = {"selection_strategy": "random", "budget": 60}
    manifest["manifest_sha256"] = _canonical_manifest_digest(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert _canonical_manifest_digest(loaded) == loaded["manifest_sha256"]
    loaded["budget"] = 120
    assert _canonical_manifest_digest(loaded) != manifest["manifest_sha256"]


def test_baseline_controls_reject_drift():
    config = ExperimentConfig(
        epochs=5,
        device="cpu",
        output_dir=ROOT / "experiments",
    )
    with pytest.raises(ValueError, match="epochs"):
        _validate_baseline_config(config)


def test_output_roots_are_unique_and_never_reused(tmp_path):
    first = _create_unique_run_root(tmp_path)
    second = _create_unique_run_root(tmp_path)
    assert first != second
    assert first.is_dir()
    assert second.is_dir()
    with pytest.raises(FileExistsError):
        first.mkdir(exist_ok=False)


def test_pretrained_checkpoint_provenance_uses_the_audited_cached_file():
    provenance = pretrained_checkpoint_provenance(
        ExperimentConfig().model
    )
    assert provenance is not None
    assert provenance["sha256"] == (
        "907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5"
    )
