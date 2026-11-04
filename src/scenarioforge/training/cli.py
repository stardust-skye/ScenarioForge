"""Command-line entry points for object-detection experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from scenarioforge.training.experiment import DetectionExperiment
from scenarioforge.training.seed import environment_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_detection",
        description="Train and evaluate ScenarioForge object detectors.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect-environment", help="Report Python, PyTorch, and device details")
    for name, description in (
        ("create-model", "Construct a configured detector"),
        ("smoke-test", "Run one real forward/backward step and checkpoint round-trip"),
        ("train", "Run training with validation-only model selection"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--config", type=Path, required=True)
    resume = commands.add_parser("resume", help="Resume an existing experiment")
    resume.add_argument("--experiment-dir", type=Path, required=True)
    resume.add_argument("--checkpoint", type=Path)
    evaluate = commands.add_parser("evaluate", help="Evaluate a saved checkpoint")
    evaluate.add_argument("--experiment-dir", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument(
        "--test-authorization",
        type=Path,
        help="Auditable JSON authorization required when --split=test",
    )
    pilot = commands.add_parser(
        "curation-pilot",
        help="Preflight and sequentially run the audited Part 3F.4 curation pilot",
    )
    pilot.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/object_detection_extended_baseline_part3d3.json"),
    )
    pilot.add_argument(
        "--curation-dir",
        type=Path,
        help="Directory containing the verified parent and seed-specific child manifests",
    )
    pilot.add_argument(
        "--selection-seed",
        type=int,
        choices=(42, 123, 2027, 31415, 271828),
        default=42,
        help="Approved seed recorded by the curation manifests",
    )
    pilot.add_argument(
        "--training-seed",
        type=int,
        choices=(42, 123, 2027, 31415, 271828),
        help="Model-training seed; defaults to --selection-seed for the approved design",
    )
    pilot.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/part3f4_curation_pilots"),
    )
    pilot.add_argument(
        "--report-path",
        type=Path,
        help="Optional report path; by default the report is written inside the unique run directory",
    )
    pilot.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate manifests, datasets, and fresh model initialization without training",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect-environment":
        print(json.dumps(environment_metadata(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ), indent=2, sort_keys=True))
        return 0
    if args.command == "curation-pilot":
        from scenarioforge.training.curation_pilot import (
            preflight_curation_pilot,
            run_curation_pilot,
        )

        if args.preflight_only:
            if args.report_path is not None:
                raise ValueError("--report-path cannot be used with --preflight-only")
            result = preflight_curation_pilot(
                baseline_config_path=args.baseline_config,
                curation_dir=args.curation_dir,
                selection_seed=args.selection_seed,
                training_seed=args.training_seed,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        run_root = run_curation_pilot(
            baseline_config_path=args.baseline_config,
            curation_dir=args.curation_dir,
            output_root=args.output_root,
            report_path=args.report_path,
            selection_seed=args.selection_seed,
            training_seed=args.training_seed,
        )
        print(f"Completed Part 3F.4 curation pilot in {run_root}")
        return 0
    if args.command in ("create-model", "smoke-test", "train"):
        experiment = DetectionExperiment.from_config_file(args.config)
        if args.command == "create-model":
            model = experiment.create_model()
            parameters = list(model.parameters())
            print(json.dumps({
                "architecture": model.architecture,
                "pretrained_weights": model.pretrained_weights,
                "foreground_class_count": len(model.model_label_to_source_id),
                "output_class_count_including_background": (
                    model.detector.roi_heads.box_predictor.cls_score.out_features
                ),
                "parameters": sum(parameter.numel() for parameter in parameters),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in parameters if parameter.requires_grad
                ),
                "device": str(experiment.device),
            }, indent=2))
        elif args.command == "smoke-test":
            print(json.dumps(experiment.run_one_batch_smoke_test(), indent=2, sort_keys=True))
        else:
            history = experiment.train()
            print(f"Completed {len(history)} epochs in {experiment.run_dir}")
        return 0
    if args.command == "resume":
        experiment = DetectionExperiment.from_experiment_dir(args.experiment_dir)
        checkpoint = args.checkpoint or experiment.run_dir / "checkpoint_last.pth"
        history = experiment.train(resume_from=checkpoint)
        print(f"Resumed through epoch {history[-1]['epoch']} in {experiment.run_dir}")
        return 0
    experiment = DetectionExperiment.from_experiment_dir(args.experiment_dir)
    metrics = experiment.evaluate_checkpoint(
        args.checkpoint,
        split_name=args.split,
        test_authorization=args.test_authorization,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
