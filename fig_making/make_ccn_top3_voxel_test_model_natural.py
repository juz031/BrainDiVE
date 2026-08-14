#!/usr/bin/env python3
"""Create a 3×9 voxel-block figure using test-model-ranked natural images."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from make_ccn_mei_threeway_grid import (
    DEFAULT_DIFFUSION_ROOT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_NSD_ROOT,
    DEFAULT_SHARE_ROOT,
    apply_fitted_prf,
    configure_font,
    fitted_prfs,
    parse_model_pair,
)
from make_ccn_single_voxel_nine_image_grid import (
    raw_diffusion_voxel_dir,
    top_diffusion_images,
    top_gradient_images,
)
from make_ccn_top3_voxel_horizontal_block import (
    display_model_name,
    top_three_voxels,
)


plt.rcParams["pdf.fonttype"] = 42


DEFAULT_FEATURE_ROOT = Path("/user_data/junruz/prf_features")
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "figure"
    / "ccn_poster_top3_voxel_test_model_natural"
)
DEFAULT_TEST_MODEL = "OPEN_CLIP_CONVNEXT_BASE"
TEST_MODEL_DISPLAY_NAME = "OpenCLIP ConvNeXt-Base"
TEST_LAYER_BY_SOURCE_LAYER = {
    "layer2": "stage2",
    "layer3": "stage3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-pair",
        default="ADV_set1_to_DINO_set1",
    )
    parser.add_argument("--roi", choices=("V1", "V4"), default="V1")
    parser.add_argument("--layer", default="layer2")
    parser.add_argument("--test-model", default=DEFAULT_TEST_MODEL)
    parser.add_argument("--test-set", type=int, default=2)
    parser.add_argument("--share-root", type=Path, default=DEFAULT_SHARE_ROOT)
    parser.add_argument(
        "--diffusion-root",
        type=Path,
        default=DEFAULT_DIFFUSION_ROOT,
    )
    parser.add_argument("--nsd-root", type=Path, default=DEFAULT_NSD_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=DEFAULT_FEATURE_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def test_model_predictions(
    *,
    model_root: Path,
    feature_root: Path,
    test_model: str,
    test_set: int,
    test_layer: str,
    voxel_id: int,
) -> np.ndarray:
    model_dir = (
        model_root
        / "S1"
        / f"{test_model}_set{test_set}"
        / test_layer
    )
    best_prf_idx = np.load(model_dir / "best_prf_idx.npy")
    feature_mean = np.load(
        model_dir / "best_features_m.npy",
        mmap_mode="r",
    )
    feature_std = np.load(
        model_dir / "best_features_s.npy",
        mmap_mode="r",
    )
    with (model_dir / "best_weights.pkl").open("rb") as handle:
        best_weights = pickle.load(handle)

    prf_index = int(best_prf_idx[voxel_id])
    feature_path = (
        feature_root
        / "S1"
        / test_model
        / test_layer
        / f"features_prf_{prf_index}.npy"
    )
    features = np.load(feature_path, mmap_mode="r")
    weights_with_intercept = np.asarray(
        best_weights[str(voxel_id)],
        dtype=np.float64,
    ).reshape(-1)
    weights = weights_with_intercept[:-1]
    intercept = float(weights_with_intercept[-1])
    mean = np.asarray(feature_mean[voxel_id], dtype=np.float64)
    std = np.asarray(feature_std[voxel_id], dtype=np.float64)

    if features.shape[1] != len(weights):
        raise ValueError(
            f"Feature/readout mismatch for voxel {voxel_id}: "
            f"{features.shape[1]} channels versus {len(weights)} weights"
        )
    predictions = np.empty(features.shape[0], dtype=np.float64)
    for start in range(0, len(predictions), 1024):
        stop = min(start + 1024, len(predictions))
        standardized = (
            np.asarray(features[start:stop], dtype=np.float64) - mean
        ) / std
        predictions[start:stop] = standardized @ weights + intercept
    return predictions


def top_test_model_natural_images(
    *,
    args: argparse.Namespace,
    voxel_id: int,
    test_layer: str,
    count: int = 3,
) -> tuple[list[np.ndarray], list[int], list[float]]:
    predictions = test_model_predictions(
        model_root=args.model_root,
        feature_root=args.feature_root,
        test_model=args.test_model,
        test_set=args.test_set,
        test_layer=test_layer,
        voxel_id=voxel_id,
    )
    finite = np.flatnonzero(np.isfinite(predictions))
    order = finite[np.argsort(predictions[finite])[-count:][::-1]]
    indices = [int(value) for value in order]
    scores = [float(predictions[value]) for value in order]
    stimulus_path = args.nsd_root / "stimuli" / "S1_stimuli_224.h5py"
    images = []
    with h5py.File(stimulus_path, "r") as handle:
        for index in indices:
            chw = np.asarray(handle["stimuli"][index])
            images.append(np.moveaxis(chw, 0, 2).astype(np.uint8))
    return images, indices, scores


def load_voxel_block(
    *,
    args: argparse.Namespace,
    test_layer: str,
    voxel_rank: int,
    voxel_id: int,
    gradient_dir: Path,
) -> dict:
    natural_images, natural_ids, natural_scores = (
        top_test_model_natural_images(
            args=args,
            voxel_id=voxel_id,
            test_layer=test_layer,
        )
    )
    diffusion_dir = raw_diffusion_voxel_dir(
        args.diffusion_root,
        args.model_pair,
        args.roi,
        args.layer,
        voxel_id,
    )
    diffusion_images, generation_r2 = top_diffusion_images(diffusion_dir)
    return {
        "voxel_rank": voxel_rank,
        "voxel_id": voxel_id,
        "generation_r2": generation_r2,
        "natural_images": natural_images,
        "natural_ids": natural_ids,
        "natural_scores": natural_scores,
        "diffusion_images": diffusion_images,
        "gradient_images": top_gradient_images(gradient_dir),
    }


def create_figure(args: argparse.Namespace) -> Path:
    configure_font()
    try:
        test_layer = TEST_LAYER_BY_SOURCE_LAYER[args.layer]
    except KeyError as error:
        raise ValueError(
            f"No test-model layer mapping for {args.layer}"
        ) from error

    pair_dir = args.share_root / args.model_pair
    blocks = [
        load_voxel_block(
            args=args,
            test_layer=test_layer,
            voxel_rank=voxel_rank,
            voxel_id=voxel_id,
            gradient_dir=gradient_dir,
        )
        for voxel_rank, voxel_id, gradient_dir in top_three_voxels(
            pair_dir,
            args.roi,
            args.layer,
        )
    ]
    generator, ranker = parse_model_pair(args.model_pair)
    generation_prfs = fitted_prfs(
        model_root=args.model_root,
        generation_model=generator,
        layer=args.layer,
        voxel_ids=[block["voxel_id"] for block in blocks],
    )

    fig, axes = plt.subplots(
        3,
        9,
        figsize=(22.0, 8.3),
        squeeze=False,
        facecolor="white",
    )
    row_labels = ("Natural (test-ranked)", "Diffusion", "Gradient")

    for block_index, block in enumerate(blocks):
        image_rows = (
            block["natural_images"],
            block["diffusion_images"],
            block["gradient_images"],
        )
        for row, images in enumerate(image_rows):
            for image_rank, full_pixels in enumerate(images):
                col = block_index * 3 + image_rank
                axis = axes[row, col]
                pixels, center, radius, x_limits, y_limits = apply_fitted_prf(
                    full_pixels,
                    generation_prfs[block["voxel_id"]],
                )
                axis.imshow(pixels, interpolation="nearest")
                axis.add_patch(
                    Circle(
                        center,
                        radius,
                        edgecolor="white",
                        facecolor="none",
                        linewidth=1.8,
                    )
                )
                axis.set_xlim(x_limits)
                axis.set_ylim(y_limits)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    axis.set_title(
                        (
                            f"NSD {block['natural_ids'][image_rank]}\n"
                            f"score {block['natural_scores'][image_rank]:.2f}"
                        ),
                        fontsize=12.5,
                        pad=4,
                        linespacing=0.95,
                    )

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(
            label,
            fontsize=15.5,
            fontweight="semibold",
            rotation=0,
            ha="right",
            va="center",
            labelpad=14,
        )

    fig.suptitle(
        f"{args.roi}  |  Generation: {display_model_name(generator)}  |  "
        f"Ranking: {display_model_name(ranker)}  |  "
        f"Test: {TEST_MODEL_DISPLAY_NAME}",
        fontsize=21,
        fontweight="semibold",
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.995,
        bottom=0.025,
        top=0.84,
        wspace=0.025,
        hspace=0.07,
    )

    for block_index, block in enumerate(blocks):
        left = axes[0, block_index * 3].get_position().x0
        right = axes[0, block_index * 3 + 2].get_position().x1
        fig.text(
            (left + right) / 2,
            0.895,
            (
                f"Voxel {block['voxel_id']}  |  "
                f"Gen. $R^2$ = {block['generation_r2']:.3f}"
            ),
            ha="center",
            va="center",
            fontsize=17,
            fontweight="semibold",
        )
        if block_index < 2:
            next_left = axes[0, block_index * 3 + 3].get_position().x0
            separator_x = (right + next_left) / 2
            fig.add_artist(
                Line2D(
                    [separator_x, separator_x],
                    [0.025, 0.91],
                    transform=fig.transFigure,
                    color="#C7CDD3",
                    linewidth=1.2,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output_dir
        / f"{args.model_pair}__{args.roi}_{args.layer}"
        "__top-3-voxels__3x9__test-model-ranked-natural.png"
    )
    fig.savefig(
        output_path,
        dpi=args.dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    if "_set1" not in args.model_pair or "_set2" in args.model_pair:
        raise ValueError("--model-pair must use set 1")
    print(f"Saved test-model-ranked natural figure to {create_figure(args)}")


if __name__ == "__main__":
    main()
