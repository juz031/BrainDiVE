from __future__ import annotations

import contextlib
import io
import time
import warnings

import numpy as np
import pandas as pd
import torch

from ..utils import atomic_write_dataframe, output_root
from .basic import extract_basic
from .common import task_rows, valid_completed, write_chunk
from .curvature import extract_curvature
from .gabor import extract_gabor, feature_info


def run_feature_task(
    config: dict, feature: str, task_id: int, *, force: bool = False,
    device_name: str | None = None, batch_size: int | None = None,
) -> str:
    if not force and valid_completed(config, feature, task_id):
        return "skipped"
    unit, rows = task_rows(config, task_id)
    prf_manifest = pd.read_csv(output_root(config) / "manifests" / "voxel_prf_manifest.csv")
    selected = prf_manifest[
        (prf_manifest["subject"] == unit["subject"]) &
        (prf_manifest["backbone"] == unit["backbone"]) &
        (prf_manifest["roi"] == unit["roi"]) &
        (prf_manifest["voxel_id"].astype(int) == int(unit["voxel_id"]))
    ]
    if len(selected) != 1:
        raise ValueError(f"Could not resolve one pRF for task {task_id}")
    prf = np.load(selected.iloc[0]["mask_path"]).astype(np.float64)
    image_size = int(config["features"]["image_size"])
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    started = time.time()
    # Legacy filter constructors are intentionally reused but are very verbose.
    # Keep array logs focused on task status and real errors.
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Frequency may be too high for a curved filter")
        if feature == "basic":
            result = extract_basic(rows, prf, image_size)
        elif feature == "gabor":
            result = extract_gabor(config, rows, prf, device, batch_size or 4)
            info_path = output_root(config) / "features" / "gabor_feature_info.csv"
            if not info_path.is_file():
                atomic_write_dataframe(feature_info(config), info_path)
        elif feature == "curvature":
            result = extract_curvature(config, rows, prf, device, batch_size or 4)
        else:
            raise ValueError(f"Unknown feature family: {feature}")
    result["task_id"] = int(task_id)
    result["runtime_seconds"] = time.time() - started
    write_chunk(config, feature, task_id, result)
    return "completed"
