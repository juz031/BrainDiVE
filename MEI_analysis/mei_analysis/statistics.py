from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

from .utils import atomic_write_dataframe, bh_fdr, output_root, stable_seed


META_COLUMNS = {
    "subject", "backbone", "roi", "voxel_id", "voxel_rank", "condition",
    "condition_family", "scope", "rank_cutoff", "image_summary", "n_images", "feature_family",
}


def _value_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in META_COLUMNS and not frame[column].isna().all()]


def _paired_matrix(left: pd.DataFrame, right: pd.DataFrame, join: list[str], values: list[str]) -> tuple[np.ndarray, list[str]]:
    merged = left[join + values].merge(
        right[join + values], on=join, suffixes=("__a", "__b"), validate="one_to_one"
    )
    columns = [column for column in values if f"{column}__a" in merged and f"{column}__b" in merged]
    difference = np.column_stack([
        merged[f"{column}__a"].to_numpy(float) - merged[f"{column}__b"].to_numpy(float)
        for column in columns
    ]) if columns else np.empty((len(merged), 0))
    if difference.size:
        keep = np.any(np.isfinite(difference), axis=0)
        difference = difference[:, keep]
        columns = [column for column, retained in zip(columns, keep) if retained]
    return difference, columns


def _matrix_statistics(difference: np.ndarray, iterations: int, bootstraps: int, seed: int) -> dict[str, np.ndarray]:
    n, p = difference.shape
    means = np.nanmean(difference, axis=0)
    medians = np.nanmedian(difference, axis=0)
    std = np.nanstd(difference, axis=0, ddof=1)
    effect = np.divide(means, std, out=np.full_like(means, np.nan), where=std > 0)
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(iterations, n))
    permuted = np.abs(np.nanmean(signs[:, :, None] * difference[None, :, :], axis=1))
    observed = np.abs(means)
    p_values = (1 + np.sum(permuted >= observed[None, :], axis=0)) / (iterations + 1)
    indices = rng.integers(0, n, size=(bootstraps, n))
    bootstrap_values = np.nanmean(difference[indices, :], axis=1)
    ci_low, ci_high = np.nanpercentile(bootstrap_values, [2.5, 97.5], axis=0)
    return {
        "mean_difference": means, "median_difference": medians,
        "paired_standardized_effect": effect, "p_value": p_values,
        "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
        "n_pairs": np.full(p, n),
    }


def _comparison_rows(
    difference: np.ndarray, columns: list[str], metadata: dict,
    iterations: int, bootstraps: int, seed: int,
) -> list[dict]:
    if not len(columns) or difference.shape[0] == 0:
        return []
    stats = _matrix_statistics(difference, iterations, bootstraps, seed)
    rows = []
    for index, feature in enumerate(columns):
        rows.append({**metadata, "feature": feature, **{key: value[index] for key, value in stats.items()}})
    return rows


def planned_contrasts(config: dict, summary: pd.DataFrame, feature_family: str) -> pd.DataFrame:
    primary = summary[
        (summary["rank_cutoff"].astype(int) == int(config["images_per_voxel"])) &
        (summary["image_summary"] == config["statistics"]["primary_image_summary"])
    ].copy()
    values = _value_columns(primary)
    iterations = int(config["statistics"]["permutation_iterations"])
    bootstraps = int(config["statistics"]["bootstrap_iterations"])
    rows = []
    gradient = [condition["id"] for condition in config["conditions"] if condition["family"] == "gradient"]
    within_pairs = [("current_mei", condition) for condition in gradient]
    within_pairs += [("current_mei", "nsd_measured_top"), ("current_mei", "nsd_predicted_top")]
    within_pairs += list(combinations(gradient, 2))
    within_pairs += [("nsd_measured_top", "nsd_predicted_top")]
    join = ["subject", "backbone", "roi", "voxel_id", "scope"]
    for (subject, backbone, roi, scope), group in primary.groupby(["subject", "backbone", "roi", "scope"]):
        for condition_a, condition_b in within_pairs:
            left = group[group["condition"] == condition_a]
            right = group[group["condition"] == condition_b]
            difference, columns = _paired_matrix(left, right, join, values)
            rows.extend(_comparison_rows(
                difference, columns,
                {"comparison_type": "condition", "subject": subject, "backbone": backbone,
                 "roi": roi, "scope": scope, "condition_a": condition_a, "condition_b": condition_b,
                 "feature_family": feature_family},
                iterations, bootstraps,
                stable_seed(config["statistics"]["random_seed"], feature_family, subject, backbone, roi, scope, condition_a, condition_b),
            ))
    # Full versus pRF within each condition.
    join_scope = ["subject", "backbone", "roi", "voxel_id", "condition"]
    for (subject, backbone, roi, condition), group in primary.groupby(["subject", "backbone", "roi", "condition"]):
        difference, columns = _paired_matrix(
            group[group["scope"] == "prf"], group[group["scope"] == "full"], join_scope, values
        )
        rows.extend(_comparison_rows(
            difference, columns,
            {"comparison_type": "scope", "subject": subject, "backbone": backbone,
             "roi": roi, "scope": "prf_minus_full", "condition_a": condition,
             "condition_b": condition, "feature_family": feature_family},
            iterations, bootstraps,
            stable_seed(config["statistics"]["random_seed"], feature_family, "scope", subject, backbone, roi, condition),
        ))
    # Backbone contrasts on shared voxel IDs.
    backbones = list(config["backbones"])
    if len(backbones) >= 2:
        for backbone_a, backbone_b in combinations(backbones, 2):
            join_backbone = ["subject", "roi", "voxel_id", "condition", "scope"]
            for (subject, roi, condition, scope), group in primary.groupby(["subject", "roi", "condition", "scope"]):
                difference, columns = _paired_matrix(
                    group[group["backbone"] == backbone_a], group[group["backbone"] == backbone_b],
                    join_backbone, values,
                )
                rows.extend(_comparison_rows(
                    difference, columns,
                    {"comparison_type": "backbone", "subject": subject,
                     "backbone": f"{backbone_a}_minus_{backbone_b}", "roi": roi, "scope": scope,
                     "condition_a": condition, "condition_b": condition, "feature_family": feature_family},
                    iterations, bootstraps,
                    stable_seed(config["statistics"]["random_seed"], feature_family, "backbone", subject, roi, condition, scope),
                ))
    result = pd.DataFrame(rows)
    if len(result):
        family_columns = ["comparison_type", "subject", "backbone", "roi", "scope", "condition_a", "condition_b"]
        result["q_value"] = result.groupby(family_columns, dropna=False)["p_value"].transform(
            lambda values_: bh_fdr(values_)
        )
    return result


def omnibus_tests(config: dict, summary: pd.DataFrame, feature_family: str) -> pd.DataFrame:
    primary = summary[
        (summary["rank_cutoff"].astype(int) == int(config["images_per_voxel"])) &
        (summary["image_summary"] == config["statistics"]["primary_image_summary"])
    ]
    conditions = [condition["id"] for condition in config["conditions"]]
    rows = []
    for (subject, backbone, roi, scope), group in primary.groupby(["subject", "backbone", "roi", "scope"]):
        for feature in _value_columns(group):
            pivot = group.pivot(index="voxel_id", columns="condition", values=feature).dropna()
            if not all(condition in pivot for condition in conditions) or len(pivot) < 3:
                continue
            statistic, p_value = friedmanchisquare(*(pivot[condition].to_numpy() for condition in conditions))
            rows.append({
                "subject": subject, "backbone": backbone, "roi": roi, "scope": scope,
                "feature_family": feature_family, "feature": feature, "n_voxels": len(pivot),
                "friedman_statistic": statistic, "p_value": p_value,
            })
    result = pd.DataFrame(rows)
    if len(result):
        result["q_value"] = result.groupby(["subject", "backbone", "roi", "scope"])["p_value"].transform(
            lambda values_: bh_fdr(values_)
        )
    return result


def run_statistics(config: dict) -> dict[str, str]:
    root = output_root(config)
    contrast_frames = []
    omnibus_frames = []
    for feature in ("basic", "gabor", "curvature"):
        summary = pd.read_csv(root / "summaries" / f"voxel_summary_{feature}.csv")
        contrast_frames.append(planned_contrasts(config, summary, feature))
        omnibus_frames.append(omnibus_tests(config, summary, feature))
    contrasts = pd.concat(contrast_frames, ignore_index=True, sort=False)
    omnibus = pd.concat(omnibus_frames, ignore_index=True, sort=False)
    contrast_path = root / "statistics" / "planned_paired_contrasts.csv"
    omnibus_path = root / "statistics" / "repeated_measures_omnibus.csv"
    atomic_write_dataframe(contrasts, contrast_path)
    atomic_write_dataframe(omnibus, omnibus_path)
    return {"contrasts": str(contrast_path), "omnibus": str(omnibus_path)}
