"""Command-line workflow for scenario intelligence.

Subcommands share one pipeline: ``extract`` reads the Part 1 database once and writes
``scenarios.json``; every other command consumes that file, so analyses are reproducible without
database access. Exit codes: 0 success, 2 usage/input errors, 1 unexpected failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from scenarioforge.scenario_intelligence import reporting as rep
from scenarioforge.scenario_intelligence.config import ScenarioIntelligenceConfig, default_output_dir
from scenarioforge.scenario_intelligence.curation_engine import (
    STRATEGIES,
    input_digest,
)
from scenarioforge.scenario_intelligence.diversity_analyzer import analyze_diversity
from scenarioforge.scenario_intelligence.duplicate_detector import compute_similarity, detect_duplicates
from scenarioforge.scenario_intelligence.rarity_analyzer import analyze_rarity
from scenarioforge.scenario_intelligence.scenario_features import CATEGORY_GROUPS, FEATURE_SPECS
from scenarioforge.scenario_intelligence.scenario_ranker import RankingConfig, rank_scenarios
from scenarioforge.scenario_intelligence.schemas import ScenarioRecord
from scenarioforge.scenario_intelligence.subset_comparison import compare_subset

logger = logging.getLogger("scenarioforge.scenario_intelligence")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scenario_intelligence",
        description="Metadata-based scenario extraction, rarity, diversity, duplicates, ranking, "
        "and curation for ingested autonomous-driving datasets (keyframe-level only).",
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: data/processed/scenario_intelligence)")
    parser.add_argument("--config", type=Path, help="JSON file overriding analysis configuration")
    parser.add_argument("--no-plots", action="store_true", help="Skip chart generation")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("extract", help="Extract scenarios from the database into scenarios.json")
    sub.add_parser("stats", help="Summarise extracted scenario features and missing values")
    sub.add_parser("rarity", help="Frequency-based rarity analysis")
    sub.add_parser("diversity", help="Entropy / Simpson / coverage analysis")
    duplicates = sub.add_parser("duplicates", help="Exact and near-duplicate metadata detection")
    duplicates.add_argument("--scene-scope", choices=("all", "same", "cross"),
                            help="Include pairs from all, same, or different scenes")
    duplicates.add_argument("--min-temporal-separation-s", type=float,
                            help="Exclude within-scene pairs closer than this many seconds")
    duplicates.add_argument("--temporal-adjacency-s", type=float,
                            help="Maximum time gap classified as temporally adjacent (default 1s)")
    duplicates.add_argument(
        "--exclude-temporally-adjacent",
        action="store_true",
        help="Exclude consecutive and temporally adjacent same-scene pairs from the reported candidates",
    )

    rank = sub.add_parser("rank", help="Rank scenarios by scenario-interest score")
    rank.add_argument("--weight", action="append", default=[], metavar="COMPONENT=VALUE",
                      help="Override component weight (rarity, composition_diversity, complexity, preference)")
    rank.add_argument("--prefer", action="append", default=[], metavar="FEATURE=WEIGHT",
                      help="Feature to favour (requires --weight preference=VALUE)")
    rank.add_argument("--top", type=int, default=20, help="Rows to plot (default 20)")

    cur = sub.add_parser(
        "curate",
        help="Disabled for training subsets; use curate-experiment for split-safe generation",
    )
    cur.add_argument("--strategy", required=True, choices=STRATEGIES)
    cur.add_argument("--n", type=int, required=True, help="Requested number of samples")
    cur.add_argument("--seed", type=int, default=0)
    cur.add_argument("--require-group", action="append", default=[], choices=CATEGORY_GROUPS)
    cur.add_argument("--min", action="append", default=[], dest="min_features", metavar="FEATURE=VALUE")
    cur.add_argument("--max", action="append", default=[], dest="max_features", metavar="FEATURE=VALUE")
    cur.add_argument("--max-similarity", type=float, help="Skip candidates at/above this similarity to a selected sample")
    cur.add_argument("--weight", action="append", default=[], metavar="COMPONENT=VALUE")
    cur.add_argument("--prefer", action="append", default=[], metavar="FEATURE=WEIGHT")

    experiment_curate = sub.add_parser(
        "curate-experiment",
        help="Create train-only, scene-balanced experiment subsets with provenance",
    )
    experiment_curate.add_argument("--scenario-file", type=Path)
    experiment_curate.add_argument("--vision-dir", type=Path)
    experiment_curate.add_argument("--split-manifest", type=Path)
    experiment_curate.add_argument("--dataset-root", type=Path)
    experiment_curate.add_argument(
        "--output-root",
        type=Path,
        help="Parent for a new immutable run directory (default: data/processed/curation_experiments)",
    )
    experiment_curate.add_argument("--seed", type=int, default=42)
    experiment_curate.add_argument(
        "--training-seed",
        type=int,
        help="Training RNG seed recorded separately; defaults to --seed",
    )
    experiment_curate.add_argument(
        "--budget",
        type=int,
        action="append",
        dest="budgets",
        help="Reduced subset budget to generate; repeat as needed (default: 60, 120, 180)",
    )

    cmp_ = sub.add_parser("compare", help="Compare original vs curated distributions from a manifest")
    cmp_.add_argument("--manifest", type=Path, required=True)
    return parser


def _pairs(items: Sequence[str], option: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise ValueError(f"{option} expects NAME=VALUE, got '{item}'")
        try:
            parsed[name] = float(value)
        except ValueError as exc:
            raise ValueError(f"{option} value for '{name}' is not a number: '{value}'") from exc
    return parsed


def _ranking_config(base: RankingConfig, weights: Sequence[str], prefs: Sequence[str]) -> RankingConfig:
    merged = {**base.weights, **_pairs(weights, "--weight")}
    preferences = {**base.preferences, **_pairs(prefs, "--prefer")}
    return RankingConfig(weights=merged, preferences=preferences)


def _load(args: argparse.Namespace, out: Path) -> tuple[str, list[ScenarioRecord]]:
    return rep.load_scenarios(out / rep.SCENARIO_FILE)


def _numeric_summary(records: Sequence[ScenarioRecord]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, spec in FEATURE_SPECS.items():
        values = [r.features.get(name) for r in records]
        present = [v for v in values if v is not None]
        entry: dict[str, Any] = {"kind": spec.kind, "source": spec.source, "description": spec.description,
                                 "missing": len(values) - len(present)}
        if spec.kind in {"count", "continuous"} and present:
            entry.update(min=min(present), max=max(present), mean=statistics.fmean(present), median=statistics.median(present))
        summary[name] = entry
    return summary


def run(args: argparse.Namespace) -> int:
    from scenarioforge.cli import load_settings  # deferred: loads .env and settings only when run

    settings = load_settings()
    out = args.output_dir or default_output_dir(settings.project_root)
    config = ScenarioIntelligenceConfig.from_file(args.config) if args.config else ScenarioIntelligenceConfig()
    plots = out / "plots"
    cmd = args.command

    if cmd == "curate-experiment":
        from scenarioforge.scenario_intelligence.experiment_curation import (
            create_experiment_subsets,
        )

        vision_dir = args.vision_dir or settings.project_root / "data" / "processed" / "vision"
        scenario_file = args.scenario_file or out / rep.SCENARIO_FILE
        split_manifest = args.split_manifest or vision_dir / "splits" / "scene_split_seed42.json"
        dataset_root = args.dataset_root or settings.project_root / "data" / "downloaded-dataset"
        output_root = (
            args.output_root
            or settings.project_root / "data" / "processed" / "curation_experiments"
        )
        run_dir = create_experiment_subsets(
            scenario_path=scenario_file,
            vision_dir=vision_dir,
            split_manifest_path=split_manifest,
            dataset_root=dataset_root,
            output_root=output_root,
            seed=args.seed,
            training_seed=args.training_seed,
            reduced_budgets=tuple(args.budgets) if args.budgets else (60, 120, 180),
        )
        logger.info("Created immutable train-only curation subsets in %s", run_dir)
        return 0

    if cmd == "curate":
        raise ValueError(
            "The unrestricted 'curate' command is disabled for training data because it "
            "scores all scenario records. Use 'curate-experiment' with the authoritative "
            "seed-42 split to generate leakage-safe training subsets."
        )

    if cmd == "extract":
        from scenarioforge.database.connection import create_database_engine, create_session_factory
        from scenarioforge.datasets.nuscenes_lookups import NuScenesLookups
        from scenarioforge.scenario_intelligence.data_access import SqlScenarioDataSource
        from scenarioforge.scenario_intelligence.scenario_extractor import extract_scenarios

        dataset_id = f"nuscenes:{settings.dataset_version}"
        lookups = NuScenesLookups.load(settings.dataset_root, settings.dataset_version)
        engine = create_database_engine(settings)
        try:
            with create_session_factory(engine)() as session:
                source = SqlScenarioDataSource(session, dataset_id, lookups)
                records = extract_scenarios(source.iter_samples(), config.near_ego_radius_m)
        finally:
            engine.dispose()
        path = rep.save_scenarios(records, out / rep.SCENARIO_FILE, dataset_id)
        logger.info("Wrote %d scenarios to %s", len(records), path)
        return 0

    dataset_id, records = _load(args, out)
    if not records:
        raise ValueError("scenarios.json contains no scenarios; run 'extract' against an ingested database")

    if cmd == "stats":
        groups: dict[str, int] = {}
        for r in records:
            for g, c in (r.features.get("group_counts") or {}).items():
                groups[g] = groups.get(g, 0) + c
        report = {"dataset_source_id": dataset_id, "n_scenarios": len(records),
                  "n_scenes": len({r.scene_id for r in records}),
                  "scope": "keyframe-level metadata; sweeps not included",
                  "category_group_object_counts": groups, "features": _numeric_summary(records)}
        rep.write_json(report, out / "scenario_statistics.json")
        rep.write_flat_csv(report, out / "scenario_statistics.csv")
        if not args.no_plots:
            from scenarioforge.scenario_intelligence.diversity_analyzer import object_category_counts
            rep.plot_bar(object_category_counts(records), "Annotated objects per category", plots / "category_distribution.png")
        return 0

    if cmd == "rarity":
        report = analyze_rarity(records, config.rarity)
        payload = report.to_dict()
        rep.write_json(payload, out / "rarity_report.json")
        rep.write_flat_csv({k: v for k, v in payload.items() if k != "scenario_scores"}, out / "rarity_report.csv")
        _write_rows(out / "rarity_scores.csv", ["sample_id", "category", "combination", "feature", "combined", "is_rare_combination"],
                    [[s.sample_id, s.category, s.combination, s.feature, s.combined, s.is_rare_combination]
                     for s in report.scenario_scores.values()])
        if not args.no_plots:
            rep.plot_bar(report.combination_frequency, "Category combination frequency (top 15)", plots / "combination_frequency.png", top=15, xlabel="scenarios")
            scores = [s.combined for s in report.scenario_scores.values() if s.combined is not None]
            rep.plot_histogram(scores, "Combined rarity score distribution", "combined rarity (0-1)", plots / "rarity_distribution.png")
        return 0

    if cmd == "diversity":
        report = analyze_diversity(records, config.diversity)
        rep.write_json(report, out / "diversity_report.json")
        rep.write_flat_csv(report, out / "diversity_report.csv")
        if not args.no_plots:
            metrics = {f"{k}": v["normalized_entropy"] for k, v in report["feature_diversity"].items()}
            metrics["object_category"] = report["object_category_distribution"]["normalized_entropy"]
            rep.plot_bar(metrics, "Normalized entropy by dimension (1 = even)", plots / "diversity_metrics.png", xlabel="normalized entropy")
            rep.plot_bar({k: v["non_missing_fraction"] for k, v in report["feature_coverage"].items()},
                         "Feature coverage (non-missing fraction)", plots / "feature_coverage.png", xlabel="fraction")
        return 0

    if cmd == "duplicates":
        duplicate_config = replace(
            config.duplicates,
            scene_scope=args.scene_scope if args.scene_scope is not None else config.duplicates.scene_scope,
            min_temporal_separation_s=(
                args.min_temporal_separation_s
                if args.min_temporal_separation_s is not None
                else config.duplicates.min_temporal_separation_s
            ),
            temporal_adjacency_s=(
                args.temporal_adjacency_s
                if args.temporal_adjacency_s is not None
                else config.duplicates.temporal_adjacency_s
            ),
            exclude_temporally_adjacent=(
                args.exclude_temporally_adjacent or config.duplicates.exclude_temporally_adjacent
            ),
        )
        report = detect_duplicates(records, duplicate_config)
        rep.write_json(report, out / "duplicate_report.json")
        rep.write_flat_csv({k: v for k, v in report.items() if k != "near_duplicate_pairs"}, out / "duplicate_report.csv")
        _write_rows(
            out / "near_duplicate_pairs.csv",
            ["sample_a", "sample_b", "similarity", "same_scene", "temporal_relationship",
             "scene_sample_order_gap", "time_separation_s", "similarity_components_json"],
            [[p["sample_a"], p["sample_b"], p["similarity"], p["same_scene"],
              p["temporal_relationship"], p["scene_sample_order_gap"], p["time_separation_s"],
              json.dumps(p["similarity_components"], sort_keys=True)]
             for p in report["near_duplicate_pairs"]],
        )
        if not args.no_plots:
            h = report["similarity_histogram"]
            rep.plot_histogram_counts(h["counts"], h["edges"], "Pairwise metadata similarity", "similarity", plots / "duplicate_similarity.png")
        return 0

    if cmd == "rank":
        rarity = analyze_rarity(records, config.rarity)
        ranking_config = _ranking_config(config.ranking, args.weight, args.prefer)
        ranked = rank_scenarios(records, rarity, ranking_config)
        rows = [r.to_dict() for r in ranked]
        rep.write_json({"dataset_source_id": dataset_id, "score_name": "scenario-interest score",
                        "note": "Heuristic for prioritising data exploration; not a safety or risk score.",
                        "weights": dict(ranking_config.weights), "preferences": dict(ranking_config.preferences),
                        "ranking": rows}, out / "scenario_ranking.json")
        _write_rows(out / "scenario_ranking.csv", ["rank", "sample_id", "scene_id", "score", "rarity", "composition_diversity", "complexity", "preference"],
                    [[r["rank"], r["sample_id"], r["scene_id"], r["scenario_interest_score"]] + [r["components"][c] for c in ("rarity", "composition_diversity", "complexity", "preference")] for r in rows])
        if not args.no_plots:
            rep.plot_stacked_ranking(rows, "Top scenarios by scenario-interest score", plots / "scenario_ranking.png", args.top)
        return 0

    if cmd == "compare":
        manifest = rep.read_json(args.manifest)
        if manifest.get("input_digest_sha256") != input_digest(records):
            raise ValueError("Manifest was created from different scenarios than scenarios.json; re-run extraction/curation")
        chosen = {row["sample_id"] for row in manifest["selected"]}
        subset = [r for r in records if r.sample_id in chosen]
        similarity = compute_similarity(records, config.duplicates)
        report = compare_subset(records, subset, similarity=similarity,
                                similarity_threshold=config.duplicates.similarity_threshold,
                                diversity_config=config.diversity)
        report["manifest_sha256"] = manifest.get("manifest_sha256")
        name = args.manifest.stem.replace("_manifest", "")
        rep.write_json(report, out / f"comparison_{name}.json")
        rep.write_flat_csv(report, out / f"comparison_{name}.csv")
        if not args.no_plots:
            d = report["category_distribution"]
            rep.plot_grouped_distribution(d["original"], d["curated"], f"Original vs curated ({name})", plots / f"comparison_{name}.png")
        return 0

    raise ValueError(f"Unknown command: {cmd}")


def _write_rows(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from scenarioforge.cli import load_settings
    from scenarioforge.config.logging_config import configure_logging

    configure_logging(load_settings().log_level)
    try:
        return run(args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    except Exception:  # noqa: BLE001 - reported with traceback, never silent
        logger.exception("Unexpected failure in scenario_intelligence %s", args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())
