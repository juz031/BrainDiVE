from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .utils import atomic_write_dataframe, condition_map, ensure_output_tree, output_root, subject_path


VOXEL_PATTERN = re.compile(r"voxel-rank-(?P<rank>\d+)__id-(?P<id>\d+)")


def discover_contexts(config: dict) -> pd.DataFrame:
    """Discover target contexts from current MEI ranking files."""
    rows = []
    maximum_rank = config["voxel_selection"].get("maximum_rank")
    for subject in config["subjects"]:
        root = subject_path(config["paths"]["current_root"], subject)
        for backbone, definition in config["backbones"].items():
            model_root = root / definition["current_dir"]
            ranking_paths = sorted(model_root.glob("**/ranking.json"))
            for path in ranking_paths:
                match = VOXEL_PATTERN.fullmatch(path.parent.name)
                if not match:
                    continue
                roi_name = path.parent.parent.name
                if not roi_name.startswith("roi-"):
                    continue
                roi = roi_name.removeprefix("roi-")
                rank = int(match.group("rank"))
                if roi not in config["rois"] or (maximum_rank is not None and rank > maximum_rank):
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                voxel_id = int(payload.get("voxel_id", match.group("id")))
                if voxel_id != int(match.group("id")):
                    raise ValueError(f"Voxel folder/JSON mismatch at {path}")
                rows.append({
                    "subject": subject, "backbone": backbone, "roi": roi,
                    "voxel_id": voxel_id, "voxel_rank": rank,
                    "ranking_json": str(path), "voxel_r2": payload.get("voxel_r2"),
                })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No current MEI ranking files found under {root}")
    duplicate = frame.duplicated(["subject", "backbone", "roi", "voxel_id"], keep=False)
    if duplicate.any():
        raise ValueError(f"Duplicate contexts:\n{frame.loc[duplicate].to_string(index=False)}")
    return frame.sort_values(["subject", "backbone", "roi", "voxel_rank"]).reset_index(drop=True)


def _current_rows(config: dict, context: pd.Series) -> list[dict]:
    path = Path(context["ranking_json"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    limit = int(config["images_per_voxel"])
    results = sorted(payload["results"], key=lambda item: int(item["rank"]))[:limit]
    rows = []
    for item in results:
        image_path = Path(item.get("image_path", ""))
        if not image_path.is_file():
            image_path = path.parent / item["image_file"]
        rows.append({
            "condition": "current_mei", "condition_family": "current",
            "image_rank": int(item["rank"]), "image_path": str(image_path),
            "seed": item.get("seed"), "generation_score": item.get("gen_score"),
            "ranking_score": item.get("rank_score"),
            "measured_response": None, "predicted_response": None, "nsd_stimulus_id": None,
            "selection_prf_role": "generation_model_prf_used_by_existing_ranker",
        })
    return rows


def _gradient_rows(config: dict, context: pd.Series, condition: dict) -> list[dict]:
    definition = config["backbones"][context["backbone"]]
    method = condition["method"]
    voxel_dir = (
        subject_path(config["paths"]["gradient_root"], str(context["subject"])) / definition["gradient_dir"] /
        context["roi"] / f"voxel_{int(context['voxel_id']):06d}" / method
    )
    method_json = voxel_dir / "method.json"
    if not method_json.is_file():
        raise FileNotFoundError(method_json)
    payload = json.loads(method_json.read_text(encoding="utf-8"))
    limit = int(config["images_per_voxel"])
    records = sorted(payload["records"], key=lambda item: int(item["rank"]))[:limit]
    rows = []
    for item in records:
        image_path = Path(item.get("image_path", ""))
        if not image_path.is_file():
            matches = list((voxel_dir / "images").glob(f"rank_{int(item['rank']):02d}__*__step_100.png"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Cannot uniquely resolve gradient rank {item['rank']} in {voxel_dir}")
            image_path = matches[0]
        rows.append({
            "condition": condition["id"], "condition_family": "gradient",
            "image_rank": int(item["rank"]), "image_path": str(image_path),
            "seed": item.get("seed"), "generation_score": item.get("generation_score"),
            "ranking_score": item.get("ranking_score"),
            "measured_response": None, "predicted_response": None, "nsd_stimulus_id": None,
            "selection_prf_role": "existing_OPEN_CLIP_RN50_rank_at_generation_prf",
        })
    return rows


def build_non_nsd_manifest(config: dict, contexts: pd.DataFrame) -> pd.DataFrame:
    conditions = condition_map(config)
    rows = []
    for context in contexts.to_dict("records"):
        context_series = pd.Series(context)
        for record in _current_rows(config, context_series):
            rows.append({**context, **record})
        for condition in config["conditions"]:
            if condition["family"] != "gradient":
                continue
            for record in _gradient_rows(config, context_series, condition):
                rows.append({**context, **record})
    frame = pd.DataFrame(rows)
    expected = len(contexts) * (1 + sum(c["family"] == "gradient" for c in conditions.values())) * int(config["images_per_voxel"])
    if len(frame) != expected:
        raise ValueError(f"Non-NSD manifest has {len(frame)} rows, expected {expected}")
    missing = ~frame["image_path"].map(lambda value: Path(value).is_file())
    if missing.any():
        raise FileNotFoundError(f"Missing {missing.sum()} images; first: {frame.loc[missing, 'image_path'].iloc[0]}")
    return frame


def build_work_units(config: dict, contexts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    task_id = 0
    for context in contexts.to_dict("records"):
        for condition in config["conditions"]:
            rows.append({
                "task_id": task_id, "subject": context["subject"],
                "backbone": context["backbone"], "roi": context["roi"],
                "voxel_id": int(context["voxel_id"]), "voxel_rank": int(context["voxel_rank"]),
                "condition": condition["id"],
            })
            task_id += 1
    return pd.DataFrame(rows)


def plan(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_output_tree(config)
    root = output_root(config)
    contexts = discover_contexts(config)
    units = build_work_units(config, contexts)
    atomic_write_dataframe(contexts, root / "manifests" / "contexts.csv")
    atomic_write_dataframe(units, root / "manifests" / "work_units.csv")
    return contexts, units


def validate_final_manifest(config: dict, frame: pd.DataFrame) -> None:
    keys = ["subject", "backbone", "roi", "voxel_id", "condition"]
    counts = frame.groupby(keys).size()
    expected = int(config["images_per_voxel"])
    invalid = counts[counts != expected]
    if len(invalid):
        raise ValueError(f"Manifest groups must contain {expected} images:\n{invalid.to_string()}")
    expected_groups = len(discover_contexts(config)) * len(config["conditions"])
    if len(counts) != expected_groups:
        raise ValueError(f"Manifest has {len(counts)} groups; expected {expected_groups}")
    if frame["condition"].isin(config.get("excluded_gradient_methods", [])).any():
        raise ValueError("Excluded gradient methods entered the manifest")


def write_final_manifest(config: dict, non_nsd: pd.DataFrame, nsd: pd.DataFrame) -> pd.DataFrame:
    root = output_root(config)
    combined = pd.concat([non_nsd, nsd], ignore_index=True, sort=False)
    combined["image_exists"] = combined["image_path"].map(lambda value: Path(value).is_file())
    validate_final_manifest(config, combined)
    combined = combined.sort_values(
        ["subject", "backbone", "roi", "voxel_rank", "condition", "image_rank"]
    ).reset_index(drop=True)
    combined.insert(0, "observation_id", range(len(combined)))
    atomic_write_dataframe(combined, root / "manifests" / "image_manifest.csv")
    return combined
