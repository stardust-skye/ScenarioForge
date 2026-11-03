"""CLI for scene splits and split-level quality reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from scenarioforge.vision_data.splits import (
    DEFAULT_SPLIT_RATIOS,
    build_split_manifest,
    load_split_manifest,
    write_split_manifest,
)
from scenarioforge.vision_data.statistics import (
    compute_split_statistics,
    load_csv_rows,
    write_split_statistics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_vision_data",
        description="Create scene-disjoint nuScenes detection splits and audit statistics.",
    )
    parser.add_argument("--vision-dir", type=Path, default=Path("data/processed/vision"))
    parser.add_argument("--dataset-root", type=Path, default=_default_dataset_root())
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split", help="Create and persist scene-grouped splits")
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["train"])
    split.add_argument("--validation-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["validation"])
    split.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["test"])
    split.add_argument("--assignments", type=Path, help="JSON mapping train/validation/test to scene tokens")
    split.add_argument("--output", type=Path, help="Split manifest path")

    stats = subparsers.add_parser("stats", help="Write split statistics and leakage reports")
    stats.add_argument("--split-manifest", type=Path, required=True)
    stats.add_argument("--output-dir", type=Path)
    stats.add_argument("--training-selection", type=Path, help="JSON array of selected train sample tokens")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    vision_dir = args.vision_dir.resolve()
    image_rows = load_csv_rows(vision_dir / "image_manifest.csv")
    annotation_rows = load_csv_rows(vision_dir / "annotations.csv")
    dataset_manifest = json.loads((vision_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    if args.command == "split":
        assignments = (
            json.loads(args.assignments.read_text(encoding="utf-8"))
            if args.assignments
            else None
        )
        ratios = {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        }
        manifest = build_split_manifest(
            image_rows,
            dataset_version=dataset_manifest["dataset_version"],
            seed=args.seed,
            ratios=ratios if assignments is None else None,
            scene_assignments=assignments,
        )
        default_name = "scene_split_explicit.json" if assignments is not None else f"scene_split_seed{args.seed}.json"
        output = args.output or vision_dir / "splits" / default_name
        write_split_manifest(
            manifest,
            output,
            image_rows=image_rows,
            dataset_version=dataset_manifest["dataset_version"],
        )
        print(
            f"Wrote {output}: "
            + ", ".join(
                f"{split}={len(manifest['scene_tokens'][split])} scenes/"
                f"{len(manifest['sample_tokens'][split])} images"
                for split in ("train", "validation", "test")
            )
        )
        print(f"Manifest digest: {manifest['manifest_digest']}")
        return 0

    manifest = load_split_manifest(
        args.split_manifest,
        image_rows=image_rows,
        dataset_version=dataset_manifest["dataset_version"],
    )
    selection = (
        json.loads(args.training_selection.read_text(encoding="utf-8"))
        if args.training_selection
        else None
    )
    report = compute_split_statistics(
        image_rows,
        annotation_rows,
        manifest,
        args.dataset_root,
        dataset_version=dataset_manifest["dataset_version"],
        selected_training_samples=selection,
    )
    output_dir = args.output_dir or args.split_manifest.parent
    paths = write_split_statistics(report, output_dir)
    for split, values in report["splits"].items():
        print(
            f"{split}: {values['scene_count']} scenes, {values['image_count']} images, "
            f"{values['annotation_count']} boxes, {values['empty_image_count']} empty images"
        )
    print(f"Leakage validation: {report['leakage_validation']}")
    print(f"Wrote reports: {paths}")
    return 0


def _default_dataset_root() -> Path:
    configured = os.environ.get("NUSCENES_ROOT")
    if configured:
        return Path(configured)
    local = Path("data/downloaded-dataset")
    return local if local.is_dir() else Path("data/raw/nuscenes")


if __name__ == "__main__":
    raise SystemExit(main())
