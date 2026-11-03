"""Command-line annotation conversion and visual inspection for nuScenes cameras."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from scenarioforge.cli import load_settings
from scenarioforge.database.connection import create_database_engine
from scenarioforge.database.models import Dataset
from scenarioforge.datasets.nuscenes_adapter import NuScenesAdapter
from scenarioforge.vision.category_mapping import CategoryMapping
from scenarioforge.vision.config import ProjectionConfig
from scenarioforge.vision.converter import convert_camera_samples
from scenarioforge.vision.dataset_builder import write_conversion
from scenarioforge.vision.inspection import INSPECTION_MODES, render_inspection

logger = logging.getLogger("scenarioforge.vision")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_vision_dataset",
        description="Project nuScenes 3D annotations into model-independent camera 2D boxes.",
    )
    parser.add_argument("--dataset-root", type=Path, help="nuScenes dataset root (otherwise project settings/database)")
    parser.add_argument("--version", help="nuScenes metadata version (default from settings)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: data/processed/vision)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert and export camera annotations")
    convert.add_argument("--config", type=Path, help="JSON ProjectionConfig overrides")
    convert.add_argument("--mapping", type=Path, help="JSON detection category mapping")

    inspect = subparsers.add_parser("inspect", help="Render saved annotations over source images")
    inspect.add_argument(
        "--manifest",
        type=Path,
        help="annotations.json path (default: <output-dir>/annotations.json)",
    )
    inspect.add_argument("--mode", choices=INSPECTION_MODES, default="random")
    inspect.add_argument("--sample-token", help="Inspect one sample token")
    inspect.add_argument("--count", type=int, default=4)
    inspect.add_argument("--seed", type=int, default=42)
    inspect.add_argument("--inspection-dir", type=Path, help="Where overlay PNGs are written")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
    version = args.version or settings.dataset_version
    output_dir = (args.output_dir or settings.project_root / "data" / "processed" / "vision").resolve()
    dataset_root = _resolve_dataset_root(args.dataset_root, settings, version)

    if args.command == "convert":
        config = _load_config(args.config)
        mapping = _load_mapping(args.mapping)
        adapter = NuScenesAdapter(dataset_root, version)
        problems = adapter.validate_accessibility()
        if problems:
            raise ValueError("Dataset is not accessible:\n" + "\n".join(problems))
        visibility = adapter.visibility_levels
        source_samples = list(adapter.iter_camera_samples(config.camera_channel))
        result = convert_camera_samples(
            source_samples,
            dataset_root=dataset_root,
            dataset_version=version,
            visibility_levels=visibility,
            config=config,
            category_mapping=mapping,
        )
        files = write_conversion(result, output_dir)
        print(
            f"Converted {result.report['total_samples_processed']} samples: "
            f"{len(result.accepted_annotations)} accepted, "
            f"{len(result.rejected_annotations)} rejected. Outputs: {output_dir}"
        )
        logger.info("Wrote %d vision dataset files", len(files))
        return 0

    manifest_path = args.manifest or output_dir / "annotations.json"
    inspection_dir = args.inspection_dir or output_dir / "inspection"
    saved = render_inspection(
        manifest_path,
        dataset_root,
        inspection_dir,
        mode=args.mode,
        sample_token=args.sample_token,
        count=args.count,
        seed=args.seed,
    )
    for path in saved:
        print(path)
    return 0


def _resolve_dataset_root(explicit: Path | None, settings, version: str) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        return root
    configured = settings.dataset_root
    if configured.is_dir():
        return configured
    engine = create_database_engine(settings)
    try:
        if inspect(engine).has_table(Dataset.__tablename__):
            with Session(engine) as session:
                stored_root = session.scalar(
                    select(Dataset.root_path).where(Dataset.source_id == f"nuscenes:{version}")
                )
        else:
            stored_root = None
    finally:
        engine.dispose()
    if stored_root:
        root = Path(stored_root).expanduser().resolve()
        if root.is_dir():
            logger.warning(
                "Configured dataset root %s is missing; using the valid root recorded in the "
                "ingested dataset metadata: %s",
                configured,
                root,
            )
            return root
    raise FileNotFoundError(
        f"nuScenes root {configured} is unavailable and no valid ingested dataset root was found; "
        "set NUSCENES_ROOT or pass --dataset-root."
    )


def _load_config(path: Path | None) -> ProjectionConfig:
    if path is None:
        return ProjectionConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(ProjectionConfig.__dataclass_fields__)
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown projection configuration keys: {sorted(unknown)}")
    return ProjectionConfig(**payload)


def _load_mapping(path: Path | None) -> CategoryMapping:
    if path is None:
        return CategoryMapping()
    return CategoryMapping.from_dict(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
