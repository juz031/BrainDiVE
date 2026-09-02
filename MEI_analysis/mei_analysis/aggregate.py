from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features.common import IDENTITY_COLUMNS, chunk_path, valid_completed
from .utils import atomic_write_dataframe, output_root


GROUP_COLUMNS = [
    "subject", "backbone", "roi", "voxel_id", "voxel_rank", "condition", "condition_family",
]


def _derived_gabor(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    settings = config["features"]["gabor"]
    n_sf = len(settings["spatial_frequencies_cycles_per_image"])
    n_ori = len(settings["orientations_degrees"])
    matrix_columns = [f"gabor_sf{sf:02d}_ori{ori:02d}" for sf in range(n_sf) for ori in range(n_ori)]
    values = frame[matrix_columns].to_numpy(dtype=float).reshape(len(frame), n_sf, n_ori)
    frame = frame.copy()
    frame["gabor_total"] = np.nanmean(values, axis=(1, 2))
    for sf in range(n_sf):
        frame[f"gabor_sf_profile_{sf:02d}"] = np.nanmean(values[:, sf, :], axis=1)
    for ori in range(n_ori):
        frame[f"gabor_orientation_profile_{ori:02d}"] = np.nanmean(values[:, :, ori], axis=1)
    return frame


def combine_chunks(config: dict, feature: str, *, allow_partial: bool = False) -> pd.DataFrame:
    root = output_root(config)
    units = pd.read_csv(root / "manifests" / "work_units.csv")
    frames = []
    missing = []
    for task_id in units["task_id"].astype(int):
        if not valid_completed(config, feature, task_id):
            missing.append(task_id)
            continue
        frames.append(pd.read_csv(chunk_path(config, feature, task_id)))
    if missing and not allow_partial:
        raise RuntimeError(f"{feature}: {len(missing)} incomplete tasks; first IDs {missing[:10]}")
    if not frames:
        raise RuntimeError(f"No completed {feature} chunks")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined["observation_id"].duplicated().any():
        raise ValueError(f"Duplicate observation IDs in {feature} chunks")
    return combined


def split_scope(combined: pd.DataFrame, scope: str, feature: str, config: dict) -> pd.DataFrame:
    identity = [column for column in IDENTITY_COLUMNS if column in combined]
    prefix = f"{scope}__"
    feature_columns = [column for column in combined if column.startswith(prefix)]
    result = combined[identity + feature_columns].copy()
    result = result.rename(columns={column: column.removeprefix(prefix) for column in feature_columns})
    result["scope"] = scope
    if feature == "gabor":
        result = _derived_gabor(result, config)
    return result


def summarize_scope(frame: pd.DataFrame, config: dict, feature: str) -> pd.DataFrame:
    cutoffs = list(map(int, config["statistics"]["sensitivity_rank_cutoffs"]))
    identity = set(IDENTITY_COLUMNS + ["scope"])
    value_columns = [column for column in frame if column not in identity]
    outputs = []
    for cutoff in cutoffs:
        subset = frame[frame["image_rank"].astype(int) <= cutoff]
        grouped = subset.groupby(GROUP_COLUMNS + ["scope"], sort=False)[value_columns]
        for statistic, values in (("median", grouped.median()), ("mean", grouped.mean())):
            values = values.reset_index()
            values["rank_cutoff"] = cutoff
            values["image_summary"] = statistic
            values["n_images"] = grouped.size().to_numpy()
            values["feature_family"] = feature
            outputs.append(values)
    return pd.concat(outputs, ignore_index=True, sort=False)


def aggregate_feature(config: dict, feature: str, *, allow_partial: bool = False) -> dict[str, Path]:
    root = output_root(config)
    combined = combine_chunks(config, feature, allow_partial=allow_partial)
    outputs: dict[str, Path] = {}
    summaries = []
    for scope in ("full", "prf"):
        scoped = split_scope(combined, scope, feature, config)
        directory = "full" if scope == "full" else "prf_weighted"
        feature_path = root / "features" / directory / f"{feature}_features.csv"
        atomic_write_dataframe(scoped, feature_path)
        outputs[scope] = feature_path
        summaries.append(summarize_scope(scoped, config, feature))
    summary = pd.concat(summaries, ignore_index=True, sort=False)
    summary_path = root / "summaries" / f"voxel_summary_{feature}.csv"
    atomic_write_dataframe(summary, summary_path)
    outputs["summary"] = summary_path
    return outputs


def aggregate_all(config: dict, *, allow_partial: bool = False) -> dict[str, dict[str, Path]]:
    return {
        feature: aggregate_feature(config, feature, allow_partial=allow_partial)
        for feature in ("basic", "gabor", "curvature")
    }
