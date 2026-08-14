#!/usr/bin/env python3
"""Create a 4-row × 12-column, four-mode top-three-voxel figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

import prf_utils
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
    top_natural_images,
)
from make_ccn_top3_voxel_horizontal_block import (
    display_model_name,
    top_three_voxels,
)
from make_ccn_top3_voxel_test_model_natural import (
    DEFAULT_FEATURE_ROOT,
    DEFAULT_TEST_MODEL,
    TEST_LAYER_BY_SOURCE_LAYER,
    TEST_MODEL_DISPLAY_NAME,
    top_test_model_natural_images,
)


plt.rcParams["pdf.fonttype"] = 42


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "figure"
    / "ccn_poster_top3_voxel_four_mode_4x12"
)
IMAGES_PER_MODE = 4
SHARED_ADV_DINO_VOXELS = {
    "V1": (9656, 8066, 7752),
    "V4": (13177, 13162, 13641),
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


def fitted_prf_ids_and_parameters(
    *,
    model_root: Path,
    model_directory_name: str,
    layer: str,
    voxel_ids: list[int],
) -> tuple[dict[int, int], dict[int, np.ndarray]]:
    best_prf_path = (
        model_root
        / "S1"
        / model_directory_name
        / layer
        / "best_prf_idx.npy"
    )
    best_prf_idx = np.load(best_prf_path)
    prf_grid, _ = prf_utils.get_prf_grid(grid_name="default-log-polar")
    ids = {
        voxel_id: int(best_prf_idx[voxel_id])
        for voxel_id in voxel_ids
    }
    parameters = {
        voxel_id: np.asarray(prf_grid[prf_id], dtype=np.float64)
        for voxel_id, prf_id in ids.items()
    }
    return ids, parameters


def selected_voxels(
    *,
    pair_dir: Path,
    roi: str,
    layer: str,
    generator: str,
) -> list[tuple[int, int, Path]]:
    if generator not in {"ADV", "DINO"}:
        return top_three_voxels(pair_dir, roi, layer)

    gradient_root = pair_dir / "gradient_hanfei" / f"{roi}_{layer}"
    selected = []
    for shared_rank, voxel_id in enumerate(
        SHARED_ADV_DINO_VOXELS[roi],
        start=1,
    ):
        matches = list(
            gradient_root.glob(f"voxel-rank-*_id-{voxel_id:06d}")
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one gradient directory for shared voxel "
                f"{voxel_id} in {gradient_root}; found {len(matches)}"
            )
        selected.append((shared_rank, voxel_id, matches[0]))
    return selected


def load_voxel_block(
    *,
    args: argparse.Namespace,
    test_layer: str,
    voxel_rank: int,
    voxel_id: int,
    gradient_dir: Path,
) -> dict:
    gt_images, _ = top_natural_images(
        args.nsd_root,
        voxel_id,
        count=IMAGES_PER_MODE,
    )
    test_images, _, _ = top_test_model_natural_images(
        args=args,
        voxel_id=voxel_id,
        test_layer=test_layer,
        count=IMAGES_PER_MODE,
    )
    diffusion_dir = raw_diffusion_voxel_dir(
        args.diffusion_root,
        args.model_pair,
        args.roi,
        args.layer,
        voxel_id,
    )
    diffusion_images, generation_r2 = top_diffusion_images(
        diffusion_dir,
        count=IMAGES_PER_MODE,
    )
    gradient_images = top_gradient_images(
        gradient_dir,
        count=IMAGES_PER_MODE,
    )
    return {
        "voxel_rank": voxel_rank,
        "voxel_id": voxel_id,
        "generation_r2": generation_r2,
        "gt_images": gt_images,
        "test_images": test_images,
        "diffusion_images": diffusion_images,
        "gradient_images": gradient_images,
    }


def draw_panel(
    *,
    axis: plt.Axes,
    full_pixels: np.ndarray,
    prf_parameters: np.ndarray,
) -> None:
    pixels, center, radius, x_limits, y_limits = apply_fitted_prf(
        full_pixels,
        prf_parameters,
    )
    axis.imshow(pixels, interpolation="nearest")
    axis.add_patch(
        Circle(
            center,
            radius,
            edgecolor="white",
            facecolor="none",
            linewidth=1.65,
        )
    )
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def create_figure(args: argparse.Namespace) -> Path:
    configure_font()
    try:
        test_layer = TEST_LAYER_BY_SOURCE_LAYER[args.layer]
    except KeyError as error:
        raise ValueError(
            f"No test-model layer mapping for {args.layer}"
        ) from error

    pair_dir = args.share_root / args.model_pair
    generator, ranker = parse_model_pair(args.model_pair)
    blocks = [
        load_voxel_block(
            args=args,
            test_layer=test_layer,
            voxel_rank=voxel_rank,
            voxel_id=voxel_id,
            gradient_dir=gradient_dir,
        )
        for voxel_rank, voxel_id, gradient_dir in selected_voxels(
            pair_dir=pair_dir,
            roi=args.roi,
            layer=args.layer,
            generator=generator,
        )
    ]
    voxel_ids = [block["voxel_id"] for block in blocks]

    generation_parameters = fitted_prfs(
        model_root=args.model_root,
        generation_model=generator,
        layer=args.layer,
        voxel_ids=voxel_ids,
    )
    generation_prf_ids, _ = fitted_prf_ids_and_parameters(
        model_root=args.model_root,
        model_directory_name=f"{generator}_RN50_set1"
        if generator != "OPENCLIP"
        else "OPEN_CLIP_RN50_set1",
        layer=args.layer,
        voxel_ids=voxel_ids,
    )
    fig = plt.figure(
        figsize=(24.0, 10.2),
        facecolor="white",
    )
    grid = fig.add_gridspec(
        4,
        14,
        width_ratios=(
            1,
            1,
            1,
            1,
            0.34,
            1,
            1,
            1,
            1,
            0.34,
            1,
            1,
            1,
            1,
        ),
        wspace=0.02,
        hspace=0.08,
    )
    axes = np.empty((4, 12), dtype=object)
    for block_index in range(3):
        for image_rank in range(IMAGES_PER_MODE):
            col = block_index * IMAGES_PER_MODE + image_rank
            grid_col = block_index * (IMAGES_PER_MODE + 1) + image_rank
            for row in range(4):
                axes[row, col] = fig.add_subplot(grid[row, grid_col])

    row_labels = (
        "Nat. Top NSD GT",
        "Nat. Top test pred.",
        "Gradient",
        "Diffusion",
    )

    for block_index, block in enumerate(blocks):
        voxel_id = block["voxel_id"]
        image_rows = (
            block["gt_images"],
            block["test_images"],
            block["gradient_images"],
            block["diffusion_images"],
        )
        for row, images in enumerate(image_rows):
            for image_rank, full_pixels in enumerate(images):
                col = block_index * IMAGES_PER_MODE + image_rank
                axis = axes[row, col]
                draw_panel(
                    axis=axis,
                    full_pixels=full_pixels,
                    prf_parameters=generation_parameters[voxel_id],
                )

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(
            label,
            fontsize=20,
            fontweight=(
                "bold"
                if label in {"Gradient", "Diffusion"}
                else "normal"
            ),
            rotation=0,
            ha="right",
            va="center",
            labelpad=14,
            linespacing=1.05,
        )

    fig.suptitle(
        f"{args.roi}  |  Generation: {display_model_name(generator)}  |  "
        f"Ranking: {display_model_name(ranker)}  |  "
        f"Test: {TEST_MODEL_DISPLAY_NAME}",
        fontsize=29,
        fontweight="normal",
        y=0.988,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.996,
        bottom=0.018,
        top=0.84,
    )

    for block_index, block in enumerate(blocks):
        voxel_id = block["voxel_id"]
        left = axes[0, block_index * IMAGES_PER_MODE].get_position().x0
        right = axes[
            0,
            block_index * IMAGES_PER_MODE + IMAGES_PER_MODE - 1,
        ].get_position().x1
        fig.text(
            (left + right) / 2,
            0.895,
            (
                f"Voxel {voxel_id}  |  "
                f"$R^2$ = {block['generation_r2']:.3f}  |  "
                f"pRF {generation_prf_ids[voxel_id]}"
            ),
            ha="center",
            va="center",
            fontsize=22,
            fontweight="normal",
        )
        if block_index < 2:
            next_left = axes[
                0,
                block_index * IMAGES_PER_MODE + IMAGES_PER_MODE,
            ].get_position().x0
            separator_x = (right + next_left) / 2
            fig.add_artist(
                Line2D(
                    [separator_x, separator_x],
                    [0.018, 0.91],
                    transform=fig.transFigure,
                    color="#C7CDD3",
                    linewidth=1.15,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        args.output_dir
        / f"{args.model_pair}__{args.roi}_{args.layer}"
        "__top-3-voxels__4x12__gt-test-diffusion-gradient"
    )
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=args.dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        format="pdf",
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    return stem.with_suffix(".png")


def main() -> None:
    args = parse_args()
    if "_set1" not in args.model_pair or "_set2" in args.model_pair:
        raise ValueError("--model-pair must use set 1")
    print(f"Saved 4×12 four-mode figure to {create_figure(args)}")


if __name__ == "__main__":
    main()
