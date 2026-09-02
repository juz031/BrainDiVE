#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mei_analysis.aggregate import aggregate_all
from mei_analysis.features.common import valid_completed
from mei_analysis.features.runner import run_feature_task
from mei_analysis.figures import make_all_figures
from mei_analysis.manifest import build_non_nsd_manifest, plan, write_final_manifest
from mei_analysis.nsd import select_nsd
from mei_analysis.prfs import build_prf_manifest
from mei_analysis.report import generate_reports
from mei_analysis.statistics import run_statistics
from mei_analysis.utils import atomic_write_json, ensure_output_tree, load_config, output_root, provenance


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    subparsers.add_parser("prepare")
    extract = subparsers.add_parser("extract")
    extract.add_argument("--feature", choices=("basic", "gabor", "curvature"), required=True)
    extract.add_argument("--task-id", type=int, required=True)
    extract.add_argument("--device", choices=("cpu", "cuda"))
    extract.add_argument("--batch-size", type=int)
    extract.add_argument("--force", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--allow-partial", action="store_true")
    subparsers.add_parser("statistics")
    subparsers.add_parser("figures")
    subparsers.add_parser("report")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--require-analysis", action="store_true")
    subparsers.add_parser("status")
    subparsers.add_parser("pilot-task-ids")
    query = subparsers.add_parser("config-value")
    query.add_argument("key")
    return result


def prepare(config: dict) -> dict:
    contexts, units = plan(config)
    root = output_root(config)
    non_nsd = build_non_nsd_manifest(config, contexts)
    nsd, predictions, overlap = select_nsd(config, contexts)
    manifest = write_final_manifest(config, non_nsd, nsd)
    prfs = build_prf_manifest(config, contexts)
    shutil.copy2(config["_config_path"], root / "config" / Path(config["_config_path"]).name)
    result = {
        "contexts": len(contexts), "work_units": len(units), "manifest_rows": len(manifest),
        "nsd_validation_rows": len(predictions), "nsd_overlap_rows": len(overlap),
        "prfs": len(prfs), **provenance(config),
    }
    atomic_write_json(result, root / "manifests" / "prepare_summary.json")
    return result


def validate(config: dict, *, require_analysis: bool = False) -> dict:
    root = output_root(config)
    manifest = pd.read_csv(root / "manifests" / "image_manifest.csv")
    contexts = pd.read_csv(root / "manifests" / "contexts.csv")
    units = pd.read_csv(root / "manifests" / "work_units.csv")
    prfs = pd.read_csv(root / "manifests" / "voxel_prf_manifest.csv")
    group_counts = manifest.groupby(["subject", "backbone", "roi", "voxel_id", "condition"]).size()
    expected_images = int(config["images_per_voxel"])
    conditions = set(manifest["condition"])
    excluded = {"gradient_pixel_raw", "gradient_maco_raw", "pixel_raw", "maco_raw"}
    result = {
        "manifest_rows": len(manifest), "contexts": len(contexts), "work_units": len(units),
        "prfs": len(prfs), "groups": len(group_counts),
        "all_groups_have_expected_images": bool((group_counts == expected_images).all()),
        "all_images_exist": bool(manifest["image_path"].map(lambda value: Path(value).is_file()).all()),
        "excluded_conditions_absent": not bool(conditions & excluded),
        "conditions": sorted(conditions),
    }
    result["valid"] = all(
        result[key] for key in (
            "all_groups_have_expected_images", "all_images_exist", "excluded_conditions_absent"
        )
    )
    if require_analysis:
        expected_rows = len(manifest)
        expected_summary_rows = (
            len(contexts) * len(config["conditions"]) * 2 *
            len(config["statistics"]["sensitivity_rank_cutoffs"]) * 2
        )
        task_ids = units["task_id"].astype(int).tolist()
        chunk_checks = {
            feature: sum(valid_completed(config, feature, task_id) for task_id in task_ids)
            for feature in ("basic", "gabor", "curvature")
        }
        table_rows = {}
        summary_rows = {}
        for feature in ("basic", "gabor", "curvature"):
            for directory, scope in (("full", "full"), ("prf_weighted", "prf")):
                table = root / "features" / directory / f"{feature}_features.csv"
                table_rows[f"{feature}_{scope}"] = len(pd.read_csv(table)) if table.is_file() else -1
            summary = root / "summaries" / f"voxel_summary_{feature}.csv"
            summary_rows[feature] = len(pd.read_csv(summary)) if summary.is_file() else -1
        figure_index_path = root / "figures" / "figure_index.csv"
        if figure_index_path.is_file():
            figure_index = pd.read_csv(figure_index_path)
            figure_paths_valid = bool(len(figure_index)) and bool(
                figure_index["path"].map(lambda value: Path(value).is_file()).all()
            )
        else:
            figure_index = pd.DataFrame()
            figure_paths_valid = False
        required_files = [
            root / "statistics" / "planned_paired_contrasts.csv",
            root / "statistics" / "repeated_measures_omnibus.csv",
            root / "report" / "index.html",
            root / "report" / "MEI_image_statistics_report.pdf",
        ]
        analysis_checks = {
            "completed_chunks": chunk_checks,
            "all_feature_tables_have_manifest_rows": all(value == expected_rows for value in table_rows.values()),
            "feature_table_rows": table_rows,
            "all_summaries_have_expected_rows": all(value == expected_summary_rows for value in summary_rows.values()),
            "summary_rows": summary_rows,
            "expected_summary_rows": expected_summary_rows,
            "figure_count": len(figure_index),
            "all_figure_paths_exist": figure_paths_valid,
            "required_final_files_exist": all(path.is_file() and path.stat().st_size > 0 for path in required_files),
        }
        analysis_checks["valid"] = (
            all(value == len(units) for value in chunk_checks.values()) and
            analysis_checks["all_feature_tables_have_manifest_rows"] and
            analysis_checks["all_summaries_have_expected_rows"] and
            analysis_checks["all_figure_paths_exist"] and
            analysis_checks["required_final_files_exist"]
        )
        result["analysis"] = analysis_checks
        result["valid"] = result["valid"] and analysis_checks["valid"]
    atomic_write_json(result, root / "manifests" / "validation_summary.json")
    if not result["valid"]:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def status(config: dict) -> pd.DataFrame:
    root = output_root(config)
    units_path = root / "manifests" / "work_units.csv"
    total = len(pd.read_csv(units_path)) if units_path.is_file() else 0
    rows = []
    for feature in ("basic", "gabor", "curvature"):
        markers = list((root / "features" / "chunks" / feature).glob("*.complete.json"))
        rows.append({"feature": feature, "completed": len(markers), "total": total})
    return pd.DataFrame(rows)


def pilot_task_ids(config: dict) -> list[int]:
    root = output_root(config)
    units = pd.read_csv(root / "manifests" / "work_units.csv")
    selected = []
    for _, group in units.groupby(["subject", "backbone", "roi"], sort=False):
        minimum_rank = group["voxel_rank"].min()
        selected.extend(group.loc[group["voxel_rank"] == minimum_rank, "task_id"].astype(int).tolist())
    return sorted(selected)


def config_value(config: dict, key: str):
    value = config
    for part in key.split("."):
        value = value[part]
    if isinstance(value, (dict, list)):
        print(json.dumps(value))
    else:
        print(value)


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config)
    ensure_output_tree(config)
    if args.command == "plan":
        contexts, units = plan(config)
        print(json.dumps({"contexts": len(contexts), "work_units": len(units)}, indent=2))
    elif args.command == "prepare":
        print(json.dumps(prepare(config), indent=2))
    elif args.command == "extract":
        print(run_feature_task(
            config, args.feature, args.task_id, force=args.force,
            device_name=args.device, batch_size=args.batch_size,
        ))
    elif args.command == "aggregate":
        print(aggregate_all(config, allow_partial=args.allow_partial))
    elif args.command == "statistics":
        print(json.dumps(run_statistics(config), indent=2))
    elif args.command == "figures":
        print(make_all_figures(config).to_string(index=False))
    elif args.command == "report":
        print(json.dumps(generate_reports(config), indent=2))
    elif args.command == "validate":
        print(json.dumps(validate(config, require_analysis=args.require_analysis), indent=2))
    elif args.command == "status":
        print(status(config).to_string(index=False))
    elif args.command == "pilot-task-ids":
        print(" ".join(map(str, pilot_task_ids(config))))
    elif args.command == "config-value":
        config_value(config, args.key)


if __name__ == "__main__":
    main()
