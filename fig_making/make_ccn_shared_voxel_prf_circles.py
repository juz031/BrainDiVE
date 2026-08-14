#!/usr/bin/env python3
"""Illustrate shared-voxel ADV and DINO fitted pRFs on black images."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from make_ccn_mei_threeway_grid import DEFAULT_MODEL_ROOT, configure_font
from make_ccn_top3_voxel_four_mode_4x12 import (
    SHARED_ADV_DINO_VOXELS,
    fitted_prf_ids_and_parameters,
)


plt.rcParams["pdf.fonttype"] = 42


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "figure"
    / "ccn_poster_shared_voxel_prf_circles"
)
IMAGE_SIZE = 224
MODEL_SPECS = (
    ("ADV", "ADV RN50", "ADV_RN50_set1"),
    ("DINO", "DINO RN50", "DINO_RN50_set1"),
)
ROI_SPECS = (
    ("V1", "layer2"),
    ("V4", "layer3"),
)
ROI_VOXEL_COLORS = {
    "V1": ("#00D9FF", "#FF4FA3", "#FFD43B"),
    "V4": ("#65E572", "#FF8A3D", "#A970FF"),
}
VOXEL_LINESTYLES = ("solid", "dashed", "dotted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def circle_geometry(prf_parameters: np.ndarray) -> tuple[tuple[float, float], float]:
    x, y, sigma = (float(value) for value in prf_parameters)
    center = (
        IMAGE_SIZE / 2 + x * IMAGE_SIZE,
        IMAGE_SIZE / 2 - y * IMAGE_SIZE,
    )
    radius = 2 * sigma * IMAGE_SIZE
    return center, radius


def create_figure(args: argparse.Namespace) -> Path:
    configure_font()
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20.0, 6.0),
        squeeze=False,
        facecolor="white",
    )
    black_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    for row, (roi, layer) in enumerate(ROI_SPECS):
        voxel_ids = list(SHARED_ADV_DINO_VOXELS[roi])
        for col, (_model_key, model_label, model_directory) in enumerate(
            MODEL_SPECS
        ):
            panel = row * len(MODEL_SPECS) + col
            prf_ids, prf_parameters = fitted_prf_ids_and_parameters(
                model_root=args.model_root,
                model_directory_name=model_directory,
                layer=layer,
                voxel_ids=voxel_ids,
            )
            axis = axes[0, panel]
            axis.imshow(black_image, interpolation="nearest")

            legend_handles = []
            for voxel_id, color, linestyle in zip(
                voxel_ids,
                ROI_VOXEL_COLORS[roi],
                VOXEL_LINESTYLES,
            ):
                center, radius = circle_geometry(prf_parameters[voxel_id])
                axis.add_patch(
                    Circle(
                        center,
                        radius,
                        edgecolor=color,
                        facecolor="none",
                        linewidth=3.2,
                        linestyle=linestyle,
                    )
                )
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linewidth=3.2,
                        linestyle=linestyle,
                        label=f"Voxel {voxel_id} · pRF {prf_ids[voxel_id]}",
                    )
                )

            axis.set_xlim(-0.5, IMAGE_SIZE - 0.5)
            axis.set_ylim(IMAGE_SIZE - 0.5, -0.5)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.set_title(
                f"{roi} · Generation: {model_label}",
                fontsize=17,
                fontweight="semibold",
                pad=10,
            )
            axis.legend(
                handles=legend_handles,
                loc="lower left",
                bbox_to_anchor=(0.025, 0.025),
                frameon=False,
                fontsize=14,
                ncol=1,
                handlelength=2.2,
                labelspacing=0.35,
                labelcolor="white",
            )

    fig.suptitle(
        "Fitted pRFs on the full image · circle radius = 2σ",
        fontsize=25,
        fontweight="semibold",
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.035,
        right=0.995,
        bottom=0.04,
        top=0.84,
        wspace=0.08,
        hspace=0.0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "ADV_DINO__V1_V4__shared-voxel-prfs__1x4"
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=args.dpi,
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
    plt.close(fig)
    return stem.with_suffix(".png")


def main() -> None:
    args = parse_args()
    print(f"Saved fitted-pRF illustration to {create_figure(args)}")


if __name__ == "__main__":
    main()
