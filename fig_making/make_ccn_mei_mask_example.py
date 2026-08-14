#!/usr/bin/env python3
"""Create a CCN-poster illustration of full, masked, and zoomed MEIs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import prf_utils


plt.rcParams["pdf.fonttype"] = 42


DEFAULT_IMAGE = Path(
    "/user_data/junruz/BrainDiVE/contrast_con_fixed/"
    "pRF_zscore_model_ranked/sub-01/"
    "gen-ADV_RN50-layer2__rank-OPEN_CLIP_RN50-layer2/"
    "split-01__steps-200__scale-1000__contrast-match-rms57.48"
    "__k-5__seeds-1000__keep-10/roi-V1/"
    "voxel-rank-02__id-008066/images/"
    "rank-02__seed-3185550500.png"
)
DEFAULT_MODEL_ROOT = Path("/user_data/junruz/prf_models/fixed/split_1_zscore")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figure" / "ccn_poster_mei"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--model", default="ADV_RN50")
    parser.add_argument("--layer", default="layer2")
    parser.add_argument("--voxel-id", type=int, default=8066)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_prf(
    *,
    model_root: Path,
    subject: int,
    split: int,
    model: str,
    layer: str,
    voxel_id: int,
) -> tuple[int, np.ndarray]:
    model_dir = (
        model_root
        / f"S{subject}"
        / f"{model}_set{split}"
        / layer
    )
    best_prf_idx = np.load(model_dir / "best_prf_idx.npy")
    prf_grid, _ = prf_utils.get_prf_grid(grid_name="default-log-polar")
    prf_idx = int(best_prf_idx[voxel_id])
    return prf_idx, np.asarray(prf_grid[prf_idx], dtype=np.float64)


def make_masked_mei(
    image: Image.Image,
    prf_parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], float, tuple[int, ...]]:
    original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = original.shape[:2]
    if height != width:
        raise ValueError(f"Expected a square MEI, got {width}x{height}")

    x, y, sigma = (float(value) for value in prf_parameters)
    gaussian = prf_utils.gauss_2d(
        center=[x, y],
        sd=sigma,
        patch_size=width,
    )
    gaussian /= gaussian.max()
    masked = np.clip(
        original.astype(np.float32) * gaussian[:, :, None],
        0,
        255,
    ).astype(np.uint8)

    center = (width / 2 + x * width, height / 2 - y * height)
    radius = 2 * sigma * width
    crop_radius = 1.2 * radius
    left = max(0, int(np.floor(center[0] - crop_radius)))
    top = max(0, int(np.floor(center[1] - crop_radius)))
    right = min(width, int(np.ceil(center[0] + crop_radius)))
    bottom = min(height, int(np.ceil(center[1] + crop_radius)))
    crop_box = (left, top, right, bottom)
    return original, masked, center, radius, crop_box


def style_image_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def add_prf_outline(
    axis: plt.Axes,
    center: tuple[float, float],
    radius: float,
    *,
    linewidth: float,
) -> None:
    axis.add_patch(
        Circle(
            center,
            radius,
            fill=False,
            edgecolor="white",
            linewidth=linewidth,
            zorder=5,
        )
    )


def save_individual_panels(
    *,
    output_dir: Path,
    original: np.ndarray,
    masked: np.ndarray,
    center: tuple[float, float],
    radius: float,
    crop_box: tuple[int, ...],
    dpi: int,
) -> None:
    left, top, right, bottom = crop_box
    zoomed = masked[top:bottom, left:right]
    panels = (
        ("01_full_image_mei.png", original, True, None),
        ("02_full_image_masked_mei.png", masked, True, None),
        ("03_masked_zoomed_mei.png", zoomed, True, (left, top)),
    )
    for filename, pixels, show_outline, offset in panels:
        fig, axis = plt.subplots(figsize=(4, 4))
        axis.imshow(pixels, interpolation="nearest")
        if show_outline:
            if offset is None:
                outline_center = center
            else:
                outline_center = (
                    center[0] - offset[0],
                    center[1] - offset[1],
                )
            add_prf_outline(axis, outline_center, radius, linewidth=2.0)
        style_image_axis(axis)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(
            output_dir / filename,
            dpi=dpi,
            facecolor="white",
            bbox_inches=None,
            pad_inches=0,
        )
        plt.close(fig)


def create_composite(
    *,
    output_dir: Path,
    original: np.ndarray,
    masked: np.ndarray,
    center: tuple[float, float],
    radius: float,
    crop_box: tuple[int, ...],
    voxel_id: int,
    dpi: int,
) -> None:
    left, top, right, bottom = crop_box
    zoomed = masked[top:bottom, left:right]

    for font_path in Path("/usr/share/fonts/urw-base35").glob(
        "NimbusSans-*.otf"
    ):
        font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "Nimbus Sans",
            "font.size": 14,
            "axes.titleweight": "semibold",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(16.0, 6.0), facecolor="white")
    grid = fig.add_gridspec(
        1,
        3,
        left=0.033,
        right=0.995,
        bottom=0.04,
        top=0.84,
        wspace=0.20,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]

    axes[0].imshow(original, interpolation="nearest")
    axes[1].imshow(masked, interpolation="nearest")
    axes[2].imshow(zoomed, interpolation="nearest")

    add_prf_outline(axes[0], center, radius, linewidth=2.0)
    add_prf_outline(axes[1], center, radius, linewidth=2.0)
    add_prf_outline(
        axes[2],
        (center[0] - left, center[1] - top),
        radius,
        linewidth=2.0,
    )

    titles = (
        "Full-image MEI",
        "pRF-masked MEI",
        "Masked MEI · zoomed",
    )
    for axis, title in zip(axes, titles):
        style_image_axis(axis)
        axis.set_box_aspect(1)
        axis.set_title(title, fontsize=17, pad=10)

    for left_axis, right_axis in zip(axes[:-1], axes[1:]):
        left_box = left_axis.get_position()
        right_box = right_axis.get_position()
        axes[0].annotate(
            "",
            xy=(right_box.x0 - 0.0015, (right_box.y0 + right_box.y1) / 2),
            xytext=(
                left_box.x1 + 0.0015,
                (left_box.y0 + left_box.y1) / 2,
            ),
            xycoords=fig.transFigure,
            textcoords=fig.transFigure,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#263238",
                "linewidth": 1.8,
                "mutation_scale": 16,
            },
            annotation_clip=False,
        )

    fig.text(
        0.5,
        0.975,
        f"V1 voxel {voxel_id}  ·  White circle = 2σ of Gaussian",
        ha="center",
        va="top",
        fontsize=25,
        fontweight="semibold",
        color="black",
    )

    stem = output_dir / "ccn_full_masked_zoomed_mei"
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=dpi,
        facecolor="white",
        bbox_inches=None,
        pad_inches=0,
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        format="pdf",
        facecolor="white",
        bbox_inches=None,
        pad_inches=0,
    )
    fig.savefig(
        stem.with_suffix(".svg"),
        facecolor="white",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prf_idx, prf_parameters = load_prf(
        model_root=DEFAULT_MODEL_ROOT,
        subject=args.subject,
        split=args.split,
        model=args.model,
        layer=args.layer,
        voxel_id=args.voxel_id,
    )
    with Image.open(args.image) as image:
        original, masked, center, radius, crop_box = make_masked_mei(
            image,
            prf_parameters,
        )

    save_individual_panels(
        output_dir=args.output_dir,
        original=original,
        masked=masked,
        center=center,
        radius=radius,
        crop_box=crop_box,
        dpi=args.dpi,
    )
    create_composite(
        output_dir=args.output_dir,
        original=original,
        masked=masked,
        center=center,
        radius=radius,
        crop_box=crop_box,
        voxel_id=args.voxel_id,
        dpi=args.dpi,
    )
    print(f"Saved CCN MEI illustration to {args.output_dir}")
    print(f"pRF index={prf_idx}; x, y, sigma={prf_parameters.tolist()}")


if __name__ == "__main__":
    main()
