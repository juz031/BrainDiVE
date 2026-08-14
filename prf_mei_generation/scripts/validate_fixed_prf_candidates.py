#!/usr/bin/env python3
"""Validate retained MEIs against every retained MEI and held-out NSD images.

For each voxel, the validator backbone is fitted online at the generator's
fixed pRF.  All retained MEIs (including other voxels/ROIs) and the validator
split's held-out natural images then compete in one response ranking.  A
second, parallel ranking contrast-matches every natural image to the MEI
luminance statistics and re-extracts its validator features from pixels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from PIL import Image
import tables
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from max_activation_utils import (  # noqa: E402
    PrfVoxelObjective,
    RMSContrastConstraint,
    apply_rms_contrast_constraint,
    batch_luminance_mean,
    batch_rms_contrast,
    build_feature_extractor,
)
from online_prf_fitting import (  # noqa: E402
    FixedPrfReadout,
    FixedPrfReadoutFitter,
)


SOURCE_REPO = Path("/home/hanfeig/MEI")
DEFAULT_RANKING_ROOT = Path(
    "/user_data/hanfeig/prf_max_activation/ranking_candidates/S1/"
    "CLIP_RN50_set1_gen__ADV_RN50_set1_fixed_gen_prf_rank"
)
NATURAL_STIMULUS_PATH = Path(
    "/lab_data/hendersonlab/datasets/nsd_preproc/stimuli/"
    "S1_stimuli_224.h5py"
)
DINO_MODEL_NAME = "DINO_RN50"
ADV_BACKBONE_CHECKPOINT = Path(
    "/user_data/junruz/prf_features/imagenet_l2_3_0.pt"
)

CANDIDATE_SCORE_FIELDS = (
    "validator_roi",
    "validator_layer",
    "validator_voxel_idx",
    "validator_prf_idx",
    "candidate_roi",
    "candidate_layer",
    "candidate_voxel_idx",
    "candidate_condition",
    "candidate_rank",
    "candidate_seed",
    "validation_score",
    "is_target_voxel",
    "image_path",
)
NATURAL_SCORE_FIELDS = (
    "validator_roi",
    "validator_layer",
    "validator_voxel_idx",
    "validator_prf_idx",
    "natural_image_id",
    "predicted_validation_score",
    "observed_beta",
)
MATCHED_NATURAL_SCORE_FIELDS = (
    "validator_roi",
    "validator_layer",
    "validator_voxel_idx",
    "validator_prf_idx",
    "natural_image_id",
    "predicted_validation_score",
    "luminance_mean_byte",
    "luminance_rms_byte",
)
VOXEL_COMPARISON_FIELDS = (
    "roi",
    "layer",
    "voxel_idx",
    "prf_idx",
    "heldout_val_correlation",
    "raw_best_target_mei_pool_rank",
    "matched_best_target_mei_pool_rank",
    "raw_target_meis_above_all_natural",
    "matched_target_meis_above_all_natural",
    "best_target_mei_score",
    "raw_best_natural_score",
    "matched_best_natural_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ranking-root",
        type=Path,
        default=DEFAULT_RANKING_ROOT,
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validator-model", default=DINO_MODEL_NAME)
    parser.add_argument("--validator-split", type=int, default=2)
    parser.add_argument(
        "--natural-stimulus-path",
        type=Path,
        default=NATURAL_STIMULUS_PATH,
    )
    parser.add_argument(
        "--matched-mean-byte",
        type=float,
        default=111.0,
        help="Target Rec.709 luminance mean on the 0-255 scale.",
    )
    parser.add_argument(
        "--matched-rms-byte",
        type=float,
        default=16.0,
        help="Target Rec.709 luminance RMS on the 0-255 scale.",
    )
    parser.add_argument(
        "--skip-contrast-matched",
        action="store_true",
        help="Only run the legacy comparison against unmodified natural images.",
    )
    return parser.parse_args()


def atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def write_csv(
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_candidates(ranking_root: Path) -> list[dict[str, Any]]:
    aggregate = ranking_root / "all_ranking_top10.csv"
    paths = [aggregate] if aggregate.is_file() else sorted(
        ranking_root.glob("*/layer*/voxel_*/**/ranking_top10.csv")
    )
    if not paths:
        raise FileNotFoundError(
            f"No ranking_top10.csv files found under {ranking_root}."
        )

    candidates: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for path in paths:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                image_path = row["image_path"]
                if image_path in seen_paths:
                    continue
                seen_paths.add(image_path)
                if not Path(image_path).is_file():
                    raise FileNotFoundError(image_path)
                candidates.append(row)
    if not candidates:
        raise RuntimeError("The ranking candidate table is empty.")
    return candidates


def validator_specs(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in candidates:
        key = (
            row["roi"],
            row["layer"],
            int(row["voxel_idx"]),
        )
        prf_idx = int(row["generator_prf_idx"])
        if key in unique and unique[key]["prf_idx"] != prf_idx:
            raise ValueError(f"Inconsistent generator pRF for {key}.")
        unique[key] = {
            "roi": key[0],
            "layer": key[1],
            "voxel_idx": key[2],
            "prf_idx": prf_idx,
        }
    return sorted(
        unique.values(),
        key=lambda item: (
            item["layer"],
            item["roi"],
            item["voxel_idx"],
        ),
    )


def load_image_batch(paths: Sequence[str], device: torch.device) -> torch.Tensor:
    images = []
    for path in paths:
        pixels = np.asarray(
            Image.open(path).convert("RGB"),
            dtype=np.float32,
        )
        images.append(torch.from_numpy(pixels).permute(2, 0, 1) / 255.0)
    return torch.stack(images).to(device)


def build_validator_objective(
    specs: Sequence[dict[str, Any]],
    fits: dict[int, FixedPrfReadout],
    model_name: str,
    layer: str,
    device: torch.device,
) -> tuple[PrfVoxelObjective, torch.nn.Module]:
    extractor = build_feature_extractor(
        model_name,
        layer,
        device,
        adv_checkpoint_path=str(ADV_BACKBONE_CHECKPOINT),
    )
    targets = [fits[spec["voxel_idx"]].target for spec in specs]
    objective = PrfVoxelObjective(
        feature_extractor=extractor,
        model_name=model_name,
        layer_name=layer,
        targets=targets,
        device=device,
        source_repo=str(SOURCE_REPO),
    ).eval()
    return objective, extractor


def score_candidates(
    objective: PrfVoxelObjective,
    candidates: Sequence[dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract each retained MEI once and score all same-layer voxels."""
    chunks = []
    with torch.no_grad():
        for offset in range(0, len(candidates), batch_size):
            batch = candidates[offset : offset + batch_size]
            images = load_image_batch(
                [row["image_path"] for row in batch],
                device,
            )
            chunks.append(objective(images).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def natural_pixels_to_bchw(pixels: np.ndarray) -> torch.Tensor:
    """Convert one NSD uint8 batch from BCHW/BHWC storage to float BCHW."""
    pixels = np.asarray(pixels)
    if pixels.ndim != 4:
        raise ValueError(
            f"Natural image batch must have four dimensions, got {pixels.shape}."
        )
    if pixels.shape[1] in {1, 3}:
        image = torch.from_numpy(np.ascontiguousarray(pixels))
    elif pixels.shape[-1] in {1, 3}:
        image = torch.from_numpy(
            np.ascontiguousarray(pixels.transpose(0, 3, 1, 2))
        )
    else:
        raise ValueError(
            "Natural images must have one or three channels, "
            f"got shape {pixels.shape}."
        )
    return image.to(dtype=torch.float32).div_(255.0)


def contrast_match_natural_batch(
    pixels: np.ndarray,
    constraint: RMSContrastConstraint,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Match luminance statistics, then quantize exactly like saved MEI PNGs."""
    images = natural_pixels_to_bchw(pixels).to(device)
    matched = apply_rms_contrast_constraint(images, constraint)
    matched = torch.round(matched * 255.0).div_(255.0)
    means = (
        batch_luminance_mean(matched).mul(255.0).cpu().numpy()
    )
    rms = batch_rms_contrast(matched).mul(255.0).cpu().numpy()
    return matched, means, rms


def score_contrast_matched_natural_images(
    objective: PrfVoxelObjective,
    natural_path: Path,
    natural_ids: np.ndarray,
    constraint: RMSContrastConstraint,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read, project, and re-extract held-out natural images from pixels."""
    score_chunks = []
    mean_chunks = []
    rms_chunks = []
    with tables.open_file(natural_path, mode="r") as handle, torch.no_grad():
        stimuli = handle.root.stimuli
        if np.any(natural_ids < 0) or np.any(natural_ids >= stimuli.shape[0]):
            raise IndexError(
                f"Natural image IDs are outside dataset shape {stimuli.shape}."
            )
        for offset in range(0, len(natural_ids), batch_size):
            batch_ids = natural_ids[offset : offset + batch_size]
            # PyTables does not reliably support unsorted fancy indexing here.
            pixels = np.stack(
                [np.asarray(stimuli[int(image_id)]) for image_id in batch_ids]
            )
            matched, means, rms = contrast_match_natural_batch(
                pixels,
                constraint,
                device,
            )
            score_chunks.append(objective(matched).cpu().numpy())
            mean_chunks.append(means)
            rms_chunks.append(rms)
    return (
        np.concatenate(score_chunks, axis=0),
        np.concatenate(mean_chunks),
        np.concatenate(rms_chunks),
    )


def descending_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def finite_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def compare_with_natural_pool(
    candidates: Sequence[dict[str, Any]],
    candidate_scores: np.ndarray,
    target_mask: np.ndarray,
    natural_ids: np.ndarray,
    natural_scores: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rank retained MEIs against one natural-image control population."""
    pool_scores = np.concatenate([candidate_scores, natural_scores])
    pool_ranks = descending_ranks(pool_scores)
    candidate_ranks = pool_ranks[: len(candidates)]
    target_indices = np.flatnonzero(target_mask)
    best_target_index = target_indices[
        np.argmax(candidate_scores[target_indices])
    ]
    best_target_rank = int(candidate_ranks[best_target_index])
    pool_size = len(pool_scores)
    best_row = candidates[best_target_index]

    other_mask = ~target_mask
    natural_max = float(np.max(natural_scores))
    other_mei_max = (
        float(np.max(candidate_scores[other_mask]))
        if np.any(other_mask)
        else None
    )
    target_scores = candidate_scores[target_mask]
    pool = {
        "retained_mei_count": len(candidates),
        "heldout_natural_count": len(natural_scores),
        "total_count": pool_size,
        "target_voxel_mei_count": int(np.sum(target_mask)),
    }
    result = {
        "target_mei_is_global_argmax": best_target_rank == 1,
        "best_target_mei_pool_rank": best_target_rank,
        "best_target_mei_percentile": float(
            100.0 * (pool_size - best_target_rank) / max(pool_size - 1, 1)
        ),
        "target_mei_pool_ranks": sorted(
            int(rank) for rank in candidate_ranks[target_mask]
        ),
        "target_meis_above_all_natural": int(
            np.sum(target_scores > natural_max)
        ),
        "best_target_mei_score": float(candidate_scores[best_target_index]),
        "best_natural_score": natural_max,
        "best_other_voxel_mei_score": other_mei_max,
        "best_natural_image_id": int(
            natural_ids[np.argmax(natural_scores)]
        ),
        "best_target_candidate": {
            "condition": best_row["condition"],
            "seed": int(best_row["seed"]),
            "image_path": best_row["image_path"],
        },
    }
    return pool, result


def summarize_voxel(
    spec: dict[str, Any],
    fit: FixedPrfReadout,
    candidates: Sequence[dict[str, Any]],
    candidate_scores: np.ndarray,
    natural_ids: np.ndarray,
    natural_scores: np.ndarray,
    matched_natural_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    candidate_voxels = np.asarray(
        [int(row["voxel_idx"]) for row in candidates]
    )
    target_mask = candidate_voxels == spec["voxel_idx"]
    if not np.any(target_mask):
        raise RuntimeError(
            f"No retained candidates for voxel {spec['voxel_idx']}."
        )

    raw_pool, raw_result = compare_with_natural_pool(
        candidates,
        candidate_scores,
        target_mask,
        natural_ids,
        natural_scores,
    )
    summary = {
        **spec,
        "fit": {
            "selected_lambda": fit.metadata["selected_lambda"],
            "heldout_val_r2": finite_float(fit.metadata["val_r2"]),
            "heldout_val_correlation": finite_float(
                fit.metadata["val_correlation"]
            ),
            "cache_path": str(fit.cache_path),
        },
        # Keep these legacy keys as the raw-natural comparison so old analysis
        # scripts continue to work.
        "pool": raw_pool,
        "result": raw_result,
        "raw_natural_pool": raw_pool,
        "raw_natural_result": raw_result,
    }
    if matched_natural_scores is not None:
        matched_pool, matched_result = compare_with_natural_pool(
            candidates,
            candidate_scores,
            target_mask,
            natural_ids,
            matched_natural_scores,
        )
        summary["contrast_matched_natural_pool"] = matched_pool
        summary["contrast_matched_natural_result"] = matched_result
    return summary


def comparison_table_rows(
    summaries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        raw = summary["raw_natural_result"]
        matched = summary.get("contrast_matched_natural_result", {})
        rows.append(
            {
                "roi": summary["roi"],
                "layer": summary["layer"],
                "voxel_idx": summary["voxel_idx"],
                "prf_idx": summary["prf_idx"],
                "heldout_val_correlation": summary["fit"][
                    "heldout_val_correlation"
                ],
                "raw_best_target_mei_pool_rank": raw[
                    "best_target_mei_pool_rank"
                ],
                "matched_best_target_mei_pool_rank": matched.get(
                    "best_target_mei_pool_rank"
                ),
                "raw_target_meis_above_all_natural": raw[
                    "target_meis_above_all_natural"
                ],
                "matched_target_meis_above_all_natural": matched.get(
                    "target_meis_above_all_natural"
                ),
                "best_target_mei_score": raw["best_target_mei_score"],
                "raw_best_natural_score": raw["best_natural_score"],
                "matched_best_natural_score": matched.get(
                    "best_natural_score"
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 0.0 <= args.matched_mean_byte <= 255.0:
        raise ValueError("--matched-mean-byte must be in [0, 255].")
    if not 0.0 < args.matched_rms_byte <= 127.5:
        raise ValueError("--matched-rms-byte must be in (0, 127.5].")
    output_root = args.output_root or (
        args.ranking_root
        / "validation"
        / f"{args.validator_model}_set{args.validator_split}_fixed_gen_prf"
    )
    if not args.natural_stimulus_path.is_file():
        raise FileNotFoundError(args.natural_stimulus_path)
    matched_constraint = RMSContrastConstraint(
        target=args.matched_rms_byte / 255.0,
        mode="match",
        target_mean=args.matched_mean_byte / 255.0,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    candidates = load_candidates(args.ranking_root)
    specs = validator_specs(candidates)
    print(
        f"device={device} candidates={len(candidates)} "
        f"validator_voxels={len(specs)}",
        flush=True,
    )

    all_candidate_rows: list[dict[str, Any]] = []
    all_natural_rows: list[dict[str, Any]] = []
    all_matched_natural_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for layer in sorted({spec["layer"] for spec in specs}):
        layer_specs = [spec for spec in specs if spec["layer"] == layer]
        fitter = FixedPrfReadoutFitter(
            model_name=args.validator_model,
            layer_name=layer,
            split_id=args.validator_split,
            device=device,
            cache_dir=output_root / "_online_readouts",
        )
        fits = fitter.fit_targets(
            {
                spec["voxel_idx"]: spec["prf_idx"]
                for spec in layer_specs
            }
        )
        objective, extractor = build_validator_objective(
            layer_specs,
            fits,
            args.validator_model,
            layer,
            device,
        )
        score_matrix = score_candidates(
            objective,
            candidates,
            args.batch_size,
            device,
        )
        natural_ids = fitter.splits["val"]
        matched_score_matrix = None
        matched_means = None
        matched_rms = None
        if not args.skip_contrast_matched:
            (
                matched_score_matrix,
                matched_means,
                matched_rms,
            ) = score_contrast_matched_natural_images(
                objective,
                args.natural_stimulus_path,
                natural_ids,
                matched_constraint,
                args.batch_size,
                device,
            )
        observed = fitter.load_responses(
            [spec["voxel_idx"] for spec in layer_specs],
            natural_ids,
        )

        for column, spec in enumerate(layer_specs):
            fit = fits[spec["voxel_idx"]]
            candidate_scores = score_matrix[:, column]
            natural_scores = fitter.predict_natural_images(
                fit,
                natural_ids,
            )
            matched_natural_scores = (
                matched_score_matrix[:, column]
                if matched_score_matrix is not None
                else None
            )
            for row, score in zip(candidates, candidate_scores):
                all_candidate_rows.append(
                    {
                        "validator_roi": spec["roi"],
                        "validator_layer": layer,
                        "validator_voxel_idx": spec["voxel_idx"],
                        "validator_prf_idx": spec["prf_idx"],
                        "candidate_roi": row["roi"],
                        "candidate_layer": row["layer"],
                        "candidate_voxel_idx": int(row["voxel_idx"]),
                        "candidate_condition": row["condition"],
                        "candidate_rank": int(row["rank"]),
                        "candidate_seed": int(row["seed"]),
                        "validation_score": float(score),
                        "is_target_voxel": (
                            int(row["voxel_idx"]) == spec["voxel_idx"]
                        ),
                        "image_path": row["image_path"],
                    }
                )
            for image_id, predicted, actual in zip(
                natural_ids,
                natural_scores,
                observed[:, column],
            ):
                all_natural_rows.append(
                    {
                        "validator_roi": spec["roi"],
                        "validator_layer": layer,
                        "validator_voxel_idx": spec["voxel_idx"],
                        "validator_prf_idx": spec["prf_idx"],
                        "natural_image_id": int(image_id),
                        "predicted_validation_score": float(predicted),
                        "observed_beta": float(actual),
                    }
                )
            if matched_natural_scores is not None:
                for image_id, predicted, mean, rms in zip(
                    natural_ids,
                    matched_natural_scores,
                    matched_means,
                    matched_rms,
                ):
                    all_matched_natural_rows.append(
                        {
                            "validator_roi": spec["roi"],
                            "validator_layer": layer,
                            "validator_voxel_idx": spec["voxel_idx"],
                            "validator_prf_idx": spec["prf_idx"],
                            "natural_image_id": int(image_id),
                            "predicted_validation_score": float(predicted),
                            "luminance_mean_byte": float(mean),
                            "luminance_rms_byte": float(rms),
                        }
                    )
            summaries.append(
                summarize_voxel(
                    spec,
                    fit,
                    candidates,
                    candidate_scores,
                    natural_ids,
                    natural_scores,
                    matched_natural_scores,
                )
            )
            matched_rank = (
                summaries[-1]["contrast_matched_natural_result"][
                    "best_target_mei_pool_rank"
                ]
                if matched_natural_scores is not None
                else "skipped"
            )
            print(
                f"[{spec['roi']} voxel={spec['voxel_idx']}] "
                f"best target rank: raw="
                f"{summaries[-1]['result']['best_target_mei_pool_rank']} "
                f"matched={matched_rank}",
                flush=True,
            )
        del objective, extractor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(
        all_candidate_rows,
        CANDIDATE_SCORE_FIELDS,
        output_root / "candidate_validation_scores.csv",
    )
    write_csv(
        all_natural_rows,
        NATURAL_SCORE_FIELDS,
        output_root / "natural_validation_scores.csv",
    )
    if not args.skip_contrast_matched:
        write_csv(
            all_matched_natural_rows,
            MATCHED_NATURAL_SCORE_FIELDS,
            output_root / "contrast_matched_natural_validation_scores.csv",
        )
    write_csv(
        comparison_table_rows(summaries),
        VOXEL_COMPARISON_FIELDS,
        output_root / "raw_vs_contrast_matched_summary.csv",
    )
    atomic_json_dump(
        {
            "experiment": "fixed-generator-pRF cross-backbone validation",
            "ranking_root": str(args.ranking_root),
            "validator_model": args.validator_model,
            "validator_split": args.validator_split,
            "prf_policy": (
                "validator readout refit online at each generator pRF"
            ),
            "natural_stimulus_path": str(args.natural_stimulus_path),
            "natural_partition": "validator split held-out val",
            "raw_natural_comparison_pool": (
                "all retained MEIs across all voxels/conditions plus all "
                "unmodified held-out natural images"
            ),
            "contrast_matched_natural_comparison_pool": (
                None
                if args.skip_contrast_matched
                else (
                    "all retained MEIs across all voxels/conditions plus "
                    "the same held-out natural images after pixel-space "
                    "luminance matching and fresh validator extraction"
                )
            ),
            "contrast_matching": {
                "enabled": not args.skip_contrast_matched,
                "luminance_definition": (
                    "Rec.709: 0.2126 R + 0.7152 G + 0.0722 B"
                ),
                "target_luminance_mean_byte": args.matched_mean_byte,
                "target_luminance_rms_byte": args.matched_rms_byte,
                "projection": (
                    "iterative affine mean/RMS match with clipping compensation"
                ),
                "quantization": (
                    "round to 8-bit after matching, then convert back to [0,1]"
                ),
                "feature_policy": (
                    "re-run matched pixels through the validator backbone; "
                    "never reuse original natural-image features"
                ),
            },
            "voxel_summaries": summaries,
        },
        output_root / "validation_summary.json",
    )
    print(f"validation outputs: {output_root}", flush=True)


if __name__ == "__main__":
    main()
