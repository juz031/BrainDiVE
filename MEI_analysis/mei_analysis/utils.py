from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from PIL import Image


ROI_ORDER = ("V1", "V2", "V3", "hV4")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    config["_config_hash"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return config


def output_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["output_root"])


def subject_path(template: str | Path, subject: str) -> Path:
    """Resolve optional subject placeholders used by reusable input roots."""
    number = int(subject.removeprefix("S").removeprefix("sub-"))
    return Path(str(template).format(
        subject=subject,
        subject_lower=subject.lower(),
        subject_number=number,
        subject_number_02=f"{number:02d}",
        bids_subject=f"sub-{number:02d}",
    ))


def ensure_output_tree(config: dict[str, Any]) -> None:
    root = output_root(config)
    for relative in (
        "config", "manifests", "masks", "cache/nsd_images", "cache/work_units",
        "features/chunks", "features/full", "features/prf_weighted", "summaries",
        "statistics", "figures/prf_maps", "figures/qc", "figures/galleries",
        "figures/comparisons", "logs", "report/assets",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def atomic_write_dataframe(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def load_rgb(path: str | Path, size: int = 256) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def load_gray(path: str | Path, size: int = 256) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32)


def condition_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {condition["id"]: condition for condition in config["conditions"]}


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def git_revision(project_root: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_path": config["_config_path"],
        "config_sha256": config["_config_hash"],
        "git_revision": git_revision(config["paths"]["project_root"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
        "hostname": platform.node(),
    }


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result
    order = finite[np.argsort(values[finite])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return result
