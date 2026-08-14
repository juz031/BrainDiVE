#!/usr/bin/env python3
"""Plot four model layer-wise R² curves for V1 and V4."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


plt.rcParams["pdf.fonttype"] = 42


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = Path(
    "/user_data/junruz/prf_models/fixed/split_1_zscore/S1/"
    "layerwise_mean_r2/layerwise_mean_r2.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "figure" / "ccn_poster_r2_single_roi"
)

MODELS = (
    "ADV_RN50",
    "DINO_RN50",
    "OPEN_CLIP_RN50",
    "OPEN_CLIP_CONVNEXT_BASE",
)
MODEL_LABELS = {
    "ADV_RN50": "ADV-RN50",
    "DINO_RN50": "DINO-RN50",
    "OPEN_CLIP_RN50": "OpenCLIP-RN50",
    "OPEN_CLIP_CONVNEXT_BASE": "OpenCLIP-ConvNeXt-B",
}
MODEL_COLORS = {
    "ADV_RN50": "#0072B2",
    "DINO_RN50": "#D55E00",
    "OPEN_CLIP_RN50": "#009E73",
    "OPEN_CLIP_CONVNEXT_BASE": "#CC79A7",
}
MODEL_MARKERS = {
    "ADV_RN50": "o",
    "DINO_RN50": "s",
    "OPEN_CLIP_RN50": "^",
    "OPEN_CLIP_CONVNEXT_BASE": "D",
}
MODEL_LAYERS = {
    "ADV_RN50": ("relu", "layer1", "layer2", "layer3", "layer4"),
    "DINO_RN50": ("relu", "layer1", "layer2", "layer3", "layer4"),
    "OPEN_CLIP_RN50": ("relu", "layer1", "layer2", "layer3", "layer4"),
    "OPEN_CLIP_CONVNEXT_BASE": (
        "stem",
        "stage1",
        "stage2",
        "stage3",
        "stage4",
    ),
}
LAYER_LABELS = ("Act", "Layer 1", "Layer 2", "Layer 3", "Layer 4")
ROI_NAMES = {
    "V1": "V1",
    "hV4": "V4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subset", type=int, default=1)
    parser.add_argument(
        "--rois",
        nargs="+",
        choices=tuple(ROI_NAMES),
        default=("V1", "hV4"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_font() -> None:
    for font_path in Path("/usr/share/fonts/urw-base35").glob(
        "NimbusSans-*.otf"
    ):
        font_manager.fontManager.addfont(font_path)
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        font_family = "Arial"
    except ValueError:
        # Nimbus Sans is the installed metric-compatible Arial substitute.
        font_family = "Nimbus Sans"
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 22,
            "axes.titlesize": 32,
            "axes.titleweight": "semibold",
            "axes.labelsize": 27,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 21,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_values(
    csv_path: Path,
    subset: int,
    rois: tuple[str, ...] | list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    values = {
        roi: {model: {} for model in MODELS}
        for roi in rois
    }
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            roi = row["roi"]
            model = row["backbone"]
            layer = row["layer"]
            if (
                roi not in values
                or model not in MODELS
                or layer not in MODEL_LAYERS[model]
                or int(row["subset"]) != subset
            ):
                continue
            values[roi][model][layer] = float(row["mean_r2"])

    missing = []
    for roi in rois:
        for model in MODELS:
            for layer in MODEL_LAYERS[model]:
                if layer not in values[roi][model]:
                    missing.append(f"{roi}/{model}/{layer}")
    if missing:
        raise ValueError("Missing CSV entries: " + ", ".join(missing))
    return values


def create_roi_plot(
    *,
    roi: str,
    values: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
    subset: int,
    dpi: int,
) -> Path:
    x = np.arange(len(LAYER_LABELS))
    fig, axis = plt.subplots(
        figsize=(9.6, 7.2),
        facecolor="white",
    )

    for model in MODELS:
        y = np.asarray(
            [
                values[roi][model][layer]
                for layer in MODEL_LAYERS[model]
            ],
            dtype=np.float64,
        )
        axis.plot(
            x,
            y,
            color=MODEL_COLORS[model],
            linewidth=3.0,
            marker=MODEL_MARKERS[model],
            markersize=13,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=MODEL_LABELS[model],
            zorder=3,
        )
        best_index = int(np.nanargmax(y))
        axis.scatter(
            x[best_index],
            y[best_index],
            s=420,
            marker="*",
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.9,
            zorder=5,
        )

    axis.set_xlabel("Model layer", labelpad=10)
    axis.set_ylabel("Mean voxel $R^2$", labelpad=10)
    axis.set_xticks(
        x,
        LAYER_LABELS,
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_ylim(0.0, 0.25)
    axis.set_yticks(np.arange(0.0, 0.251, 0.05))
    axis.grid(
        axis="y",
        color="#D9DEE3",
        linewidth=1.0,
        alpha=0.9,
        zorder=0,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.1)
    axis.spines["bottom"].set_linewidth(1.1)
    axis.tick_params(length=5, width=1.1)
    fig.suptitle(
        ROI_NAMES[roi],
        fontsize=27,
        fontweight="semibold",
        y=0.98,
    )
    fig.legend(
        *axis.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        fontsize=19,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.25,
        handletextpad=0.5,
    )
    fig.text(
        0.985,
        0.018,
        f"Subject 1 · data subset {subset} · star = best layer",
        ha="right",
        va="bottom",
        fontsize=13,
        color="#59636D",
    )
    fig.subplots_adjust(
        left=0.15,
        right=0.98,
        bottom=0.15,
        top=0.72,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{ROI_NAMES[roi]}__layerwise-r2__four-models"
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=dpi,
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
    fig.savefig(
        stem.with_suffix(".svg"),
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    return stem.with_suffix(".png")


def create_combined_plot(
    *,
    rois: tuple[str, str],
    values: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
    subset: int,
    dpi: int,
) -> Path:
    x = np.arange(len(LAYER_LABELS))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.8),
        sharex=True,
        sharey=True,
        facecolor="white",
    )

    for axis, roi in zip(axes, rois):
        for model in MODELS:
            y = np.asarray(
                [
                    values[roi][model][layer]
                    for layer in MODEL_LAYERS[model]
                ],
                dtype=np.float64,
            )
            axis.plot(
                x,
                y,
                color=MODEL_COLORS[model],
                linewidth=3.0,
                marker=MODEL_MARKERS[model],
                markersize=13,
                markeredgecolor="white",
                markeredgewidth=1.0,
                label=MODEL_LABELS[model],
                zorder=3,
            )
            best_index = int(np.nanargmax(y))
            axis.scatter(
                x[best_index],
                y[best_index],
                s=420,
                marker="*",
                color=MODEL_COLORS[model],
                edgecolor="black",
                linewidth=0.9,
                zorder=5,
            )

        axis.set_title(ROI_NAMES[roi], pad=13)
        axis.set_xticks(
            x,
            LAYER_LABELS,
            rotation=30,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_ylim(0.0, 0.25)
        axis.set_yticks(np.arange(0.0, 0.251, 0.05))
        axis.grid(
            axis="y",
            color="#D9DEE3",
            linewidth=1.0,
            alpha=0.9,
            zorder=0,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.1)
        axis.spines["bottom"].set_linewidth(1.1)
        axis.tick_params(length=5, width=1.1)

    fig.supxlabel("Model layer", y=0.025, fontsize=27)
    axes[0].set_ylabel("Mean voxel $R^2$", labelpad=14, fontsize=27)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        fontsize=18,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    fig.text(
        0.985,
        0.012,
        f"Subject 1 · data subset {subset} · star = best layer",
        ha="right",
        va="bottom",
        fontsize=16,
        color="#59636D",
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.23,
        top=0.77,
        wspace=0.12,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "V1_V4__layerwise-r2__four-models__1x2"
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=dpi,
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
    fig.savefig(
        stem.with_suffix(".svg"),
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    return stem.with_suffix(".png")


def main() -> None:
    args = parse_args()
    configure_font()
    values = load_values(args.csv, args.subset, args.rois)
    for roi in args.rois:
        output = create_roi_plot(
            roi=roi,
            values=values,
            output_dir=args.output_dir,
            subset=args.subset,
            dpi=args.dpi,
        )
        print(f"Saved single-ROI layer-wise R² plot to {output}")
    if set(args.rois) == {"V1", "hV4"}:
        combined = create_combined_plot(
            rois=("V1", "hV4"),
            values=values,
            output_dir=args.output_dir,
            subset=args.subset,
            dpi=args.dpi,
        )
        print(f"Saved combined 1×2 layer-wise R² plot to {combined}")


if __name__ == "__main__":
    main()
