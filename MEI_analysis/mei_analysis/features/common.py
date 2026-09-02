from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from ..utils import atomic_write_dataframe, atomic_write_json, output_root, provenance


IDENTITY_COLUMNS = [
    "observation_id", "subject", "backbone", "roi", "voxel_id", "voxel_rank",
    "condition", "condition_family", "image_rank", "image_path", "nsd_stimulus_id",
]


def task_rows(config: dict, task_id: int) -> tuple[pd.Series, pd.DataFrame]:
    root = output_root(config)
    units = pd.read_csv(root / "manifests" / "work_units.csv")
    selected = units[units["task_id"] == int(task_id)]
    if len(selected) != 1:
        raise IndexError(f"Task {task_id} resolved to {len(selected)} work units")
    unit = selected.iloc[0]
    manifest = pd.read_csv(root / "manifests" / "image_manifest.csv")
    mask = (
        (manifest["subject"] == unit["subject"]) &
        (manifest["backbone"] == unit["backbone"]) &
        (manifest["roi"] == unit["roi"]) &
        (manifest["voxel_id"].astype(int) == int(unit["voxel_id"])) &
        (manifest["condition"] == unit["condition"])
    )
    rows = manifest.loc[mask].sort_values("image_rank").reset_index(drop=True)
    expected = int(config["images_per_voxel"])
    if len(rows) != expected:
        raise ValueError(f"Task {task_id} has {len(rows)} images; expected {expected}")
    return unit, rows


def chunk_path(config: dict, feature: str, task_id: int) -> Path:
    return output_root(config) / "features" / "chunks" / feature / f"task-{int(task_id):06d}.csv"


def completion_path(config: dict, feature: str, task_id: int) -> Path:
    return output_root(config) / "features" / "chunks" / feature / f"task-{int(task_id):06d}.complete.json"


def valid_completed(config: dict, feature: str, task_id: int) -> bool:
    data_path = chunk_path(config, feature, task_id)
    marker = completion_path(config, feature, task_id)
    if not data_path.is_file() or not marker.is_file():
        return False
    try:
        frame = pd.read_csv(data_path)
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return len(frame) == int(config["images_per_voxel"]) and payload.get("rows") == len(frame)
    except Exception:
        return False


def write_chunk(config: dict, feature: str, task_id: int, frame: pd.DataFrame) -> Path:
    path = chunk_path(config, feature, task_id)
    atomic_write_dataframe(frame, path)
    marker = {
        "feature": feature, "task_id": int(task_id), "rows": len(frame),
        "columns": list(frame.columns), **provenance(config),
    }
    atomic_write_json(marker, completion_path(config, feature, task_id))
    return path


def identity_frame(rows: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in IDENTITY_COLUMNS if column in rows]
    return rows[available].copy()
