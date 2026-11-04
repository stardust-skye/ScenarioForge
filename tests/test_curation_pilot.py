from __future__ import annotations

import json
from pathlib import Path

import pytest

from scenarioforge.training.config import ExperimentConfig
from scenarioforge.training.curation_pilot import (
    AUDITED_CHILD_SHA256,
    AUDITED_RUN_SHA256,
    _canonical_manifest_digest,
    _create_unique_run_root,
    _validate_baseline_config,
    _validate_curation_inputs,
)
from scenarioforge.training.model_factory import pretrained_checkpoint_provenance


ROOT = Path(__file__).resolve().parents[1]


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
