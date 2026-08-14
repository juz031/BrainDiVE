#!/usr/bin/env python3
"""Create ranked BrainDiVE image grids, one figure per gen/rank model pair."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

BRAINDIVE_CODE = Path("/home/junruz/BrainDiVE")
if str(BRAINDIVE_CODE) not in sys.path:
    sys.path.insert(0, str(BRAINDIVE_CODE))

import prf_utils


DEFAULT_INPUT = Path(
    "/user_data/junruz/BrainDiVE/contrast_con/"
    "pRF_zscore_model_ranked/sub-01"
)
DEFAULT_OUTPUT = Path("/user_data/junruz/BrainDiVE_ranked_grids/sub-01")
DEFAULT_MODEL_ROOT = Path("/user_data/junruz/prf_models/fixed/split_1_zscore")
DEFAULT_FEATURE_ROOT = Path("/user_data/junruz/prf_features")
DEFAULT_NSD_ROOT = Path("/lab_data/hendersonlab/datasets/nsd_preproc")


@dataclass
class NaturalImageSelection:
    indices: np.ndarray
    scores: np.ndarray
    score_label: str


class NaturalImageProvider:
    def __init__(
        self,
        nsd_root: Path,
        feature_root: Path,
        model_root: Path,
        fitting_device: str,
    ) -> None:
        self.nsd_root = nsd_root
        self.feature_root = feature_root
        self.model_root = model_root
        self.fitting_device = fitting_device
        self._split_cache: dict[tuple[int, int], dict] = {}
        self._response_cache: dict[tuple[int, int], np.ndarray] = {}
        self._prediction_cache: dict[tuple, np.ndarray] = {}

    def _stimulus_path(self, subject: int) -> Path:
        return self.nsd_root / "stimuli" / f"S{subject}_stimuli_224.h5py"

    def _response_path(self, subject: int) -> Path:
        return self.nsd_root / "data" / f"S{subject}_betas_avg_bigmask.hdf5"

    def responses(self, subject: int, voxel_id: int) -> np.ndarray:
        key = (subject, voxel_id)
        if key not in self._response_cache:
            import h5py

            with h5py.File(self._response_path(subject), "r") as handle:
                values = np.asarray(handle["betas"][:, voxel_id], dtype=np.float64)
            self._response_cache[key] = values
        return self._response_cache[key]

    def images(self, subject: int, indices: np.ndarray) -> list[Image.Image]:
        import h5py

        output = []
        with h5py.File(self._stimulus_path(subject), "r") as handle:
            stimuli = handle["stimuli"]
            for index in indices:
                array = np.moveaxis(np.asarray(stimuli[int(index)]), 0, 2)
                output.append(Image.fromarray(array.astype(np.uint8), mode="RGB"))
        return output

    def split(self, subject: int, split_id: int) -> dict:
        key = (subject, split_id)
        if key not in self._split_cache:
            split_path = (
                self.model_root
                / f"S{subject}"
                / f"data_splits_S{subject}.pkl"
            )
            with split_path.open("rb") as handle:
                split_payload = pickle.load(handle)
            try:
                selected = split_payload[split_id]
            except (IndexError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Cannot read split {split_id} from {split_path}"
                ) from error
            self._split_cache[key] = selected
        return self._split_cache[key]

    def top_ground_truth(
        self,
        subject: int,
        voxel_id: int,
        count: int,
    ) -> NaturalImageSelection:
        responses = self.responses(subject, voxel_id)
        valid = np.flatnonzero(np.isfinite(responses))
        order = valid[np.argsort(responses[valid])[::-1]][:count]
        return NaturalImageSelection(
            indices=order,
            scores=responses[order],
            score_label="response",
        )

    def ranking_predictions(
        self,
        *,
        subject: int,
        split_id: int,
        rank_model: str,
        rank_layer: str,
        voxel_id: int,
        prf_idx: int,
    ) -> np.ndarray:
        cache_key = (
            subject,
            split_id,
            rank_model,
            rank_layer,
            voxel_id,
            prf_idx,
        )
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        import torch

        import model_fitting_utils

        feature_path = (
            self.feature_root
            / f"S{subject}"
            / rank_model
            / rank_layer
            / f"features_prf_{prf_idx}.npy"
        )
        features = np.load(feature_path, mmap_mode="r")
        responses = self.responses(subject, voxel_id)
        split = self.split(subject, split_id)
        train_ids = np.asarray(split["train"], dtype=int)
        nest_ids = np.asarray(split["nest"], dtype=int)

        fit_ids = np.concatenate([train_ids, nest_ids])
        feature_mean = np.mean(features[fit_ids], axis=0, keepdims=True)
        feature_std = np.std(features[fit_ids], axis=0, keepdims=True) + 1e-12
        train_features = (
            np.asarray(features[train_ids], dtype=np.float32) - feature_mean
        ) / feature_std
        nest_features = (
            np.asarray(features[nest_ids], dtype=np.float32) - feature_mean
        ) / feature_std
        train_features = np.concatenate(
            [
                train_features,
                np.ones((len(train_features), 1), dtype=np.float32),
            ],
            axis=1,
        )
        nest_features = np.concatenate(
            [
                nest_features,
                np.ones((len(nest_features), 1), dtype=np.float32),
            ],
            axis=1,
        )

        device = torch.device(self.fitting_device)
        train_tensor = torch.from_numpy(train_features).to(device)
        nest_tensor = torch.from_numpy(nest_features).to(device)
        train_response = torch.from_numpy(
            responses[train_ids].astype(np.float32)
        ).unsqueeze(1).to(device)
        nest_response = torch.from_numpy(
            responses[nest_ids].astype(np.float32)
        ).unsqueeze(1).to(device)
        small_value = 0.0001
        lambdas = (
            np.logspace(
                np.log(small_value),
                np.log(10**10 + small_value),
                20,
                dtype=np.float32,
                base=np.e,
            )
            - small_value
        )
        weights, _, _ = model_fitting_utils.solve_ridge(
            train_tensor,
            train_response,
            nest_tensor,
            nest_response,
            lambdas,
            eps=1e-4,
            return_loss=True,
        )
        weights = weights[:, 0].detach().cpu().numpy()
        feature_weights = weights[:-1]
        intercept = float(weights[-1])

        predictions = np.empty(features.shape[0], dtype=np.float64)
        chunk_size = 1024
        for start in range(0, len(predictions), chunk_size):
            stop = min(start + chunk_size, len(predictions))
            standardized = (
                np.asarray(features[start:stop], dtype=np.float64)
                - feature_mean
            ) / feature_std
            predictions[start:stop] = standardized @ feature_weights + intercept

        self._prediction_cache[cache_key] = predictions
        return predictions

    def top_ranking_prediction(
        self,
        *,
        subject: int,
        split_id: int,
        rank_model: str,
        rank_layer: str,
        voxel_id: int,
        prf_idx: int,
        count: int,
    ) -> NaturalImageSelection:
        predictions = self.ranking_predictions(
            subject=subject,
            split_id=split_id,
            rank_model=rank_model,
            rank_layer=rank_layer,
            voxel_id=voxel_id,
            prf_idx=prf_idx,
        )
        order = np.argsort(predictions)[::-1][:count]
        return NaturalImageSelection(
            indices=order,
            scores=predictions[order],
            score_label="prediction",
        )


def numeric_field(name: str, field: str) -> int:
    match = re.search(rf"(?:^|__){re.escape(field)}-(\d+)(?:__|$)", name)
    return int(match.group(1)) if match else -1


def voxel_rank(path: Path) -> int:
    match = re.search(r"voxel-rank-(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse voxel rank from {path}")
    return int(match.group(1))


def load_ranking(voxel_dir: Path) -> dict:
    ranking_path = voxel_dir / "ranking.json"
    with ranking_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def eligible_voxels(run_dir: Path, n_images: int) -> list[Path]:
    voxels = []
    for ranking_path in run_dir.glob("roi-*/voxel-rank-*/ranking.json"):
        data = load_ranking(ranking_path.parent)
        if len(data.get("results", [])) >= n_images:
            voxels.append(ranking_path.parent)
    return sorted(voxels, key=voxel_rank)


def choose_run(pair_dir: Path, n_voxels: int | None, n_images: int) -> Path:
    candidates = []
    for config_path in pair_dir.glob("*/run_config.json"):
        run_dir = config_path.parent
        voxels = eligible_voxels(run_dir, n_images)
        if n_voxels is not None and len(voxels) < n_voxels:
            continue
        # Prefer the main run: more candidates, then larger scale/keep values.
        score = (
            numeric_field(run_dir.name, "seeds"),
            numeric_field(run_dir.name, "scale"),
            numeric_field(run_dir.name, "keep"),
            len(voxels),
        )
        candidates.append((score, run_dir))
    if not candidates:
        requested = "all available" if n_voxels is None else str(n_voxels)
        raise RuntimeError(
            f"No run in {pair_dir} has {requested} voxels "
            f"and {n_images} ranked images per voxel"
        )
    return max(candidates, key=lambda item: item[0])[1]


def parse_pair_name(pair_dir: Path) -> tuple[str, str]:
    match = re.fullmatch(r"gen-(.+)__rank-(.+)", pair_dir.name)
    if not match:
        return pair_dir.name, ""
    return match.group(1), match.group(2)


def load_run_config(run_dir: Path) -> dict:
    with (run_dir / "run_config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prf_parameters(
    run_dir: Path,
    model_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    config = load_run_config(run_dir)
    subject = int(config["subject"])
    split = int(config["split"])
    gen_model = config["generation_model"]["name"]
    gen_layer = config["generation_model"]["layer"]
    model_dir = (
        model_root
        / f"S{subject}"
        / f"{gen_model}_set{split}"
        / gen_layer
    )
    best_prf_path = model_dir / "best_prf_idx.npy"
    if not best_prf_path.is_file():
        raise FileNotFoundError(f"Missing fitted pRF indices: {best_prf_path}")
    best_prf_idx = np.load(best_prf_path)
    prf_grid, _ = prf_utils.get_prf_grid(grid_name="default-log-polar")
    return best_prf_idx, prf_grid


def resolve_image_path(result: dict, voxel_dir: Path) -> Path:
    stored_path = result.get("image_path")
    if stored_path and Path(stored_path).is_file():
        return Path(stored_path)
    relative_path = result.get("image_file")
    if relative_path and (voxel_dir / relative_path).is_file():
        return voxel_dir / relative_path
    raise FileNotFoundError(
        f"Image for rank {result.get('rank')} is missing in {voxel_dir}"
    )


def apply_prf(
    image: Image.Image,
    prf_parameters: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[float, float],
    float,
    tuple[float, float],
    tuple[float, float],
]:
    rgb = image.convert("RGB")
    image_array = np.asarray(rgb, dtype=np.float32)
    height, width = image_array.shape[:2]
    if height != width:
        raise ValueError(f"Expected a square image, got {width}x{height}")

    x, y, sigma = (float(value) for value in prf_parameters)
    prf_2d = prf_utils.gauss_2d(
        center=[x, y],
        sd=sigma,
        patch_size=width,
    )
    prf_2d = prf_2d / np.max(prf_2d)
    weighted = np.clip(
        image_array * prf_2d[:, :, np.newaxis],
        0,
        255,
    ).astype(np.uint8)

    center = (width / 2 + x * width, height / 2 - y * height)
    radius = 2 * sigma * width
    crop_radius = radius * 1.2
    x_limits = (center[0] - crop_radius, center[0] + crop_radius)
    # Reversed so imshow retains the image's top-to-bottom direction.
    y_limits = (center[1] + crop_radius, center[1] - crop_radius)
    return image_array.astype(np.uint8), weighted, center, radius, x_limits, y_limits


def plot_natural_images(
    *,
    pair_dir: Path,
    run_dir: Path,
    voxels: list[Path],
    output_dir: Path,
    best_prf_idx: np.ndarray,
    prf_grid: np.ndarray,
    provider: NaturalImageProvider,
    count: int,
    dpi: int,
    selection_method: str,
) -> list[Path]:
    config = load_run_config(run_dir)
    subject = int(config["subject"])
    split_id = int(config["split"])
    rank_model = config["ranking_model"]["name"]
    rank_layer = config["ranking_model"]["layer"]
    gen_model = config["generation_model"]["name"]
    gen_layer = config["generation_model"]["layer"]
    roi = load_ranking(voxels[0]).get("region", "unknown ROI")

    if selection_method == "ground-truth":
        selection_title = "Natural images ranked by measured NSD response"
    elif selection_method == "ranking-model":
        selection_title = (
            f"Natural images ranked by {rank_model}/{rank_layer} prediction"
        )
    else:
        raise ValueError(f"Unknown natural-image selection: {selection_method}")

    figure_height = 3.8 * len(voxels) + 1.5
    selections = []
    for voxel_dir in voxels:
        data = load_ranking(voxel_dir)
        voxel_id = int(data["voxel_id"])
        prf_idx = int(best_prf_idx[voxel_id])
        if selection_method == "ground-truth":
            selection = provider.top_ground_truth(subject, voxel_id, count)
        else:
            selection = provider.top_ranking_prediction(
                subject=subject,
                split_id=split_id,
                rank_model=rank_model,
                rank_layer=rank_layer,
                voxel_id=voxel_id,
                prf_idx=prf_idx,
                count=count,
            )
        selections.append((data, prf_idx, selection))

    display_modes = (
        ("prf-circle", "Full natural images with pRF circle", False, False),
        ("prf-masked", "Full natural images with pRF mask", True, False),
        (
            "prf-masked-zoomed",
            "pRF-masked natural images, zoomed",
            True,
            True,
        ),
    )
    output_paths = []

    for suffix, treatment_title, use_mask, zoom in display_modes:
        fig, axes = plt.subplots(
            len(voxels),
            count,
            figsize=(4.0 * count, figure_height),
            squeeze=False,
        )
        row_labels = []

        for row, (data, prf_idx, selection) in enumerate(selections):
            voxel_id = int(data["voxel_id"])
            prf_parameters = prf_grid[prf_idx]
            natural_images = provider.images(subject, selection.indices)

            for col, (image, stimulus_index, score) in enumerate(
                zip(natural_images, selection.indices, selection.scores)
            ):
                axis = axes[row, col]
                (
                    original,
                    weighted,
                    center,
                    radius,
                    x_limits,
                    y_limits,
                ) = apply_prf(image, prf_parameters)
                image.close()
                axis.imshow(weighted if use_mask else original)
                axis.add_patch(
                    Circle(
                        center,
                        radius,
                        color="white",
                        fill=False,
                        linewidth=1.5,
                    )
                )
                if zoom:
                    axis.set_xlim(x_limits)
                    axis.set_ylim(y_limits)
                axis.set_title(
                    f"Rank {col + 1}\n"
                    f"{selection.score_label}={score:.2f}\n"
                    f"NSD image {int(stimulus_index)}",
                    fontsize=18,
                    pad=5,
                )
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)

            row_labels.append(
                f"Voxel rank {data['voxel_rank_in_topk']}\n"
                f"ID {voxel_id}\n"
                f"$R^2$ = {data['voxel_r2']:.3f}\n"
                f"pRF {prf_idx}"
            )

        fig.suptitle(
            f"BrainDiVE natural-image controls — {roi}\n"
            f"Generation: {gen_model}/{gen_layer}   |   "
            f"Ranking: {rank_model}/{rank_layer}\n"
            f"{selection_title} · {treatment_title}",
            fontsize=18,
            y=0.985,
        )
        fig.subplots_adjust(
            left=0.23,
            right=0.995,
            bottom=0.015,
            top=1.0 - 1.35 / figure_height,
            wspace=0.012,
            hspace=0.16,
        )
        for row, row_label in enumerate(row_labels):
            position = axes[row, 0].get_position()
            fig.text(
                0.195,
                (position.y0 + position.y1) / 2,
                row_label,
                fontsize=18,
                ha="right",
                va="center",
                zorder=20,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.95,
                    "pad": 2,
                },
            )

        output_path = (
            output_dir
            / f"{pair_dir.name}__natural-{selection_method}"
            f"__top-{len(voxels)}x{count}__{suffix}.png"
        )
        fig.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def plot_pair(
    pair_dir: Path,
    output_dir: Path,
    model_root: Path,
    n_voxels: int | None,
    n_images: int,
    natural_provider: NaturalImageProvider | None,
    natural_images: int,
    include_generated: bool,
    dpi: int,
) -> list[Path]:
    run_dir = choose_run(pair_dir, n_voxels, n_images)
    voxels = eligible_voxels(run_dir, n_images)
    if n_voxels is not None:
        voxels = voxels[:n_voxels]
    if not voxels:
        raise RuntimeError(
            f"No voxel folders with at least {n_images} ranked images "
            f"were found in selected run {run_dir}"
        )
    plotted_voxels = len(voxels)
    gen_model, rank_model = parse_pair_name(pair_dir)
    best_prf_idx, prf_grid = load_prf_parameters(run_dir, model_root)

    roi = load_ranking(voxels[0]).get("region", "unknown ROI")
    output_dir.mkdir(parents=True, exist_ok=True)
    display_modes = (
        ("prf-circle", "Full images with pRF circle", False, False),
        ("prf-masked", "Full images with pRF mask", True, False),
        ("prf-masked-zoomed", "pRF-masked images, zoomed", True, True),
    )
    output_paths = []

    for suffix, treatment_title, use_mask, zoom in (
        display_modes if include_generated else ()
    ):
        figure_height = 3.8 * plotted_voxels + 1.5
        fig, axes = plt.subplots(
            plotted_voxels,
            n_images,
            figsize=(3.4 * n_images, figure_height),
            squeeze=False,
        )
        row_labels = []

        for row, voxel_dir in enumerate(voxels):
            data = load_ranking(voxel_dir)
            results = sorted(
                data["results"],
                key=lambda item: int(item["rank"]),
            )[:n_images]
            voxel_id = int(data["voxel_id"])
            prf_idx = int(best_prf_idx[voxel_id])
            prf_parameters = prf_grid[prf_idx]

            for col, result in enumerate(results):
                axis = axes[row, col]
                image_path = resolve_image_path(result, voxel_dir)
                with Image.open(image_path) as image:
                    (
                        original,
                        weighted,
                        center,
                        radius,
                        x_limits,
                        y_limits,
                    ) = apply_prf(image, prf_parameters)

                axis.imshow(weighted if use_mask else original)
                axis.add_patch(
                    Circle(
                        center,
                        radius,
                        color="white",
                        fill=False,
                        linewidth=1.5,
                    )
                )
                if zoom:
                    axis.set_xlim(x_limits)
                    axis.set_ylim(y_limits)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)

                if row == 0:
                    axis.set_title(
                        f"Rank {result['rank']}",
                        fontsize=18,
                        pad=5,
                        zorder=20,
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.9,
                            "pad": 1.5,
                        },
                    )

            row_labels.append(
                f"Voxel rank {data['voxel_rank_in_topk']}\n"
                f"ID {data['voxel_id']}\n"
                f"$R^2$ = {data['voxel_r2']:.3f}\n"
                f"pRF {prf_idx}"
            )

        fig.suptitle(
            f"BrainDiVE ranked images — {roi}\n"
            f"Generation: {gen_model}   |   Ranking: {rank_model}\n"
            f"{treatment_title}",
            fontsize=18,
            y=0.985,
        )
        fig.subplots_adjust(
            left=0.23,
            right=0.995,
            bottom=0.015,
            top=1.0 - 1.35 / figure_height,
            wspace=0.012,
            hspace=0.06,
        )
        for row, row_label in enumerate(row_labels):
            position = axes[row, 0].get_position()
            fig.text(
                0.195,
                (position.y0 + position.y1) / 2,
                row_label,
                fontsize=18,
                ha="right",
                va="center",
                zorder=20,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.95,
                    "pad": 2,
                },
            )

        output_path = (
            output_dir
            / f"{pair_dir.name}__top-{plotted_voxels}x{n_images}__{suffix}.png"
        )
        fig.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    if natural_provider is not None:
        for selection_method in ("ground-truth", "ranking-model"):
            output_paths.extend(
                plot_natural_images(
                    pair_dir=pair_dir,
                    run_dir=run_dir,
                    voxels=voxels,
                    output_dir=output_dir,
                    best_prf_idx=best_prf_idx,
                    prf_grid=prf_grid,
                    provider=natural_provider,
                    count=natural_images,
                    dpi=dpi,
                    selection_method=selection_method,
                )
            )

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--nsd-root", type=Path, default=DEFAULT_NSD_ROOT)
    parser.add_argument(
        "--generation-model",
        default=None,
        help=(
            "Only process pairs using this generation model, for example "
            "OPEN_CLIP_RN50. Omit to process every generation model."
        ),
    )
    parser.add_argument(
        "--voxels",
        type=int,
        default=0,
        help="Number of top voxels to plot; 0 plots every available voxel",
    )
    parser.add_argument("--images", type=int, default=5)
    parser.add_argument(
        "--natural-images",
        type=int,
        default=5,
        help="Number of natural images per voxel for each ranking criterion",
    )
    parser.add_argument(
        "--fitting-device",
        default="cuda",
        help="Torch device used to fit the natural-image ranking models",
    )
    parser.add_argument(
        "--skip-natural",
        action="store_true",
        help="Only create generated-image figures",
    )
    parser.add_argument(
        "--natural-only",
        action="store_true",
        help="Only create the two natural-image control figures",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    n_voxels = None if args.voxels == 0 else args.voxels
    if n_voxels is not None and n_voxels < 1:
        parser.error("--voxels must be 0 (all) or a positive integer")
    if args.natural_images < 1:
        parser.error("--natural-images must be a positive integer")
    if args.skip_natural and args.natural_only:
        parser.error("--skip-natural and --natural-only cannot be used together")

    pair_dirs = sorted(
        path
        for path in args.input.glob("gen-*__rank-*")
        if path.is_dir()
    )
    if args.generation_model is not None:
        selected_pairs = []
        for pair_dir in pair_dirs:
            run_configs = sorted(pair_dir.glob("*/run_config.json"))
            if any(
                load_run_config(config_path.parent)["generation_model"]["name"]
                == args.generation_model
                for config_path in run_configs
            ):
                selected_pairs.append(pair_dir)
        pair_dirs = selected_pairs
    if not pair_dirs:
        generation_filter = (
            ""
            if args.generation_model is None
            else f" for generation model {args.generation_model!r}"
        )
        raise RuntimeError(
            f"No gen/rank model pair directories found in {args.input}"
            f"{generation_filter}"
        )

    natural_provider = None
    if not args.skip_natural:
        natural_provider = NaturalImageProvider(
            nsd_root=args.nsd_root,
            feature_root=args.feature_root,
            model_root=args.model_root,
            fitting_device=args.fitting_device,
        )

    outputs = []
    for pair_dir in pair_dirs:
        pair_outputs = plot_pair(
            pair_dir,
            args.output,
            args.model_root,
            n_voxels,
            args.images,
            natural_provider,
            args.natural_images,
            not args.natural_only,
            args.dpi,
        )
        outputs.extend(pair_outputs)
        for output_path in pair_outputs:
            print(output_path)

    print(f"Created {len(outputs)} figures in {args.output}")


if __name__ == "__main__":
    main()
