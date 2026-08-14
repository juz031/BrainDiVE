#!/usr/bin/env python3
"""Plot stacked layer-wise R² summaries for three RN50 backbones."""

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figure" / "ccn_poster_r2"

MODELS = ("ADV_RN50", "DINO_RN50", "OPEN_CLIP_RN50")
MODEL_LABELS = {
    "ADV_RN50": "ADV-RN50",
    "DINO_RN50": "DINO-RN50",
    "OPEN_CLIP_RN50": "OpenCLIP-RN50",
}
LAYERS = ("relu", "layer1", "layer2", "layer3", "layer4")
LAYER_LABELS = ("ReLU", "Layer 1", "Layer 2", "Layer 3", "Layer 4")
ROIS = ("V1", "V2", "V3", "hV4", "FFA", "PPA")
ROI_COLORS = {
    "V1": "#0072B2",
    "V2": "#E69F00",
    "V3": "#009E73",
    "hV4": "#D55E00",
    "FFA": "#CC79A7",
    "PPA": "#6F4E3D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subset", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_font() -> None:
    for font_path in Path("/usr/share/fonts/urw-base35").glob(
        "NimbusSans-*.otf"
    ):
        font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "Nimbus Sans",
            "font.size": 16,
            "axes.titlesize": 22,
            "axes.titleweight": "semibold",
            "axes.labelsize": 19,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_values(
    csv_path: Path,
    subset: int,
) -> dict[str, dict[str, dict[str, float]]]:
    values = {
        model: {roi: {} for roi in ROIS}
        for model in MODELS
    }
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row["backbone"]
            roi = row["roi"]
            layer = row["layer"]
            if (
                model not in MODELS
                or roi not in ROIS
                or layer not in LAYERS
                or int(row["subset"]) != subset
            ):
                continue
            values[model][roi][layer] = float(row["mean_r2"])

    missing = []
    for model in MODELS:
        for roi in ROIS:
            for layer in LAYERS:
                if layer not in values[model][roi]:
                    missing.append(f"{model}/{roi}/{layer}")
    if missing:
        raise ValueError("Missing CSV entries: " + ", ".join(missing))
    return values


def create_plot(
    *,
    values: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
    subset: int,
    dpi: int,
) -> None:
    configure_font()
    x = np.arange(len(LAYERS))
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9.5, 12.0),
        sharex=True,
        sharey=True,
        facecolor="white",
    )

    legend_handles = []
    for panel_index, (axis, model) in enumerate(zip(axes, MODELS)):
        for roi in ROIS:
            y = np.asarray(
                [values[model][roi][layer] for layer in LAYERS],
                dtype=np.float64,
            )
            (line,) = axis.plot(
                x,
                y,
                color=ROI_COLORS[roi],
                linewidth=2.4,
                marker="o",
                markersize=6.5,
                markeredgewidth=0,
                label=roi,
                zorder=3,
            )
            best_index = int(np.nanargmax(y))
            axis.scatter(
                x[best_index],
                y[best_index],
                s=165,
                marker="*",
                color=ROI_COLORS[roi],
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )
            if panel_index == 0:
                legend_handles.append(line)

        axis.set_title(MODEL_LABELS[model], loc="left", pad=8)
        axis.set_ylim(0.0, 0.25)
        axis.set_yticks(np.arange(0.0, 0.251, 0.05))
        axis.grid(
            axis="y",
            color="#D9DEE3",
            linewidth=0.9,
            alpha=0.9,
            zorder=0,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.0)
        axis.spines["bottom"].set_linewidth(1.0)
        axis.tick_params(length=4, width=1)

    axes[-1].set_xticks(x, LAYER_LABELS)
    axes[-1].set_xlabel("Model layer", labelpad=8)
    fig.supylabel("Mean voxel $R^2$", x=0.02, fontsize=19)
    fig.legend(
        legend_handles,
        ROIS,
        title="ROI",
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        frameon=False,
        handlelength=2.3,
        columnspacing=1.25,
        handletextpad=0.45,
    )
    fig.text(
        0.99,
        0.012,
        f"Subject 1 · data subset {subset} · star = best layer",
        ha="right",
        va="bottom",
        fontsize=13,
        color="#59636D",
    )
    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.085,
        top=0.89,
        hspace=0.23,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "layerwise_r2_three_models_stacked"
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


def main() -> None:
    args = parse_args()
    values = load_values(args.csv, args.subset)
    create_plot(
        values=values,
        output_dir=args.output_dir,
        subset=args.subset,
        dpi=args.dpi,
    )
    print(f"Saved stacked R² summary to {args.output_dir}")


if __name__ == "__main__":
    main()
