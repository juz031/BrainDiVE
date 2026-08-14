#!/usr/bin/env python3
"""Create a 3-row × 9-column figure containing three voxel blocks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
    parse_voxel_folder,
)
from make_ccn_single_voxel_nine_image_grid import (
    raw_diffusion_voxel_dir,
    top_diffusion_images,
    top_gradient_images,
    top_natural_images,
)


plt.rcParams["pdf.fonttype"] = 42


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "figure"
    / "ccn_poster_top3_voxel_horizontal_blocks"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-pair",
        default="ADV_set1_to_DINO_set1",
    )
    parser.add_argument("--roi", choices=("V1", "V4"), default="V1")
    parser.add_argument("--layer", default="layer2")
    parser.add_argument("--share-root", type=Path, default=DEFAULT_SHARE_ROOT)
    parser.add_argument(
        "--diffusion-root",
        type=Path,
        default=DEFAULT_DIFFUSION_ROOT,
    )
    parser.add_argument("--nsd-root", type=Path, default=DEFAULT_NSD_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def display_model_name(short_name: str) -> str:
    return {
        "ADV": "ADV RN50",
        "DINO": "DINO RN50",
        "OPENCLIP": "OpenCLIP RN50",
    }[short_name]


def top_three_voxels(
    pair_dir: Path,
    roi: str,
    layer: str,
) -> list[tuple[int, int, Path]]:
    gradient_root = pair_dir / "gradient_hanfei" / f"{roi}_{layer}"
    candidates = []
    for path in gradient_root.glob("voxel-rank-*_id-*"):
        voxel_rank, voxel_id = parse_voxel_folder(path)
        candidates.append((voxel_rank, voxel_id, path))
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < 3:
        raise RuntimeError(f"Found fewer than three voxels in {gradient_root}")
    return candidates[:3]


def load_voxel_block(
    *,
    args: argparse.Namespace,
    voxel_rank: int,
    voxel_id: int,
    gradient_dir: Path,
) -> dict:
    natural_images, natural_ids = top_natural_images(
        args.nsd_root,
        voxel_id,
    )
    diffusion_dir = raw_diffusion_voxel_dir(
        args.diffusion_root,
        args.model_pair,
        args.roi,
        args.layer,
        voxel_id,
    )
    diffusion_images, voxel_r2 = top_diffusion_images(diffusion_dir)
    gradient_images = top_gradient_images(gradient_dir)
    return {
        "voxel_rank": voxel_rank,
        "voxel_id": voxel_id,
        "voxel_r2": voxel_r2,
        "natural_images": natural_images,
        "natural_ids": natural_ids,
        "diffusion_images": diffusion_images,
        "gradient_images": gradient_images,
    }


def create_figure(args: argparse.Namespace) -> Path:
    configure_font()
    pair_dir = args.share_root / args.model_pair
    selected = top_three_voxels(pair_dir, args.roi, args.layer)
    blocks = [
        load_voxel_block(
            args=args,
            voxel_rank=voxel_rank,
            voxel_id=voxel_id,
            gradient_dir=gradient_dir,
        )
        for voxel_rank, voxel_id, gradient_dir in selected
    ]

    generator, ranker = parse_model_pair(args.model_pair)
    prfs = fitted_prfs(
        model_root=args.model_root,
        generation_model=generator,
        layer=args.layer,
        voxel_ids=[block["voxel_id"] for block in blocks],
    )

    fig, axes = plt.subplots(
        3,
        9,
        figsize=(22.0, 8.1),
        squeeze=False,
        facecolor="white",
    )
    row_labels = ("Natural", "Diffusion", "Gradient")

    for block_index, block in enumerate(blocks):
        voxel_id = block["voxel_id"]
        image_rows = (
            block["natural_images"],
            block["diffusion_images"],
            block["gradient_images"],
        )
        for row, images in enumerate(image_rows):
            for image_rank, full_pixels in enumerate(images):
                col = block_index * 3 + image_rank
                axis = axes[row, col]
                (
                    pixels,
                    center,
                    radius,
                    x_limits,
                    y_limits,
                ) = apply_fitted_prf(full_pixels, prfs[voxel_id])
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
                        f"NSD {block['natural_ids'][image_rank]}",
                        fontsize=14,
                        pad=5,
                    )

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(
            label,
            fontsize=17,
            fontweight="semibold",
            rotation=0,
            ha="right",
            va="center",
            labelpad=16,
        )

    fig.suptitle(
        f"{args.roi}  |  Generation: {display_model_name(generator)}  |  "
        f"Ranking: {display_model_name(ranker)}",
        fontsize=23,
        fontweight="semibold",
        y=0.988,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.025,
        top=0.85,
        wspace=0.025,
        hspace=0.065,
    )

    for block_index, block in enumerate(blocks):
        left = axes[0, block_index * 3].get_position().x0
        right = axes[0, block_index * 3 + 2].get_position().x1
        fig.text(
            (left + right) / 2,
            0.91,
            (
                f"Voxel {block['voxel_id']}  |  "
                f"$R^2$ = {block['voxel_r2']:.3f}"
            ),
            ha="center",
            va="center",
            fontsize=18,
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
    stem = (
        args.output_dir
        / f"{args.model_pair}__{args.roi}_{args.layer}"
        "__top-3-voxels__3x9-horizontal-block"
    )
    output_path = stem.with_suffix(".png")
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
    print(f"Saved horizontal top-three-voxel figure to {create_figure(args)}")


if __name__ == "__main__":
    main()
