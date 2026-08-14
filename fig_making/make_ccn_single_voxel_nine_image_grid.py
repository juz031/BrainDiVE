#!/usr/bin/env python3
"""Create a 3×3 natural/diffusion/gradient grid for one voxel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

from make_ccn_mei_threeway_grid import (
    DEFAULT_DIFFUSION_ROOT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_NSD_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SHARE_ROOT,
    apply_fitted_prf,
    configure_font,
    diffusion_model_name,
    fitted_prfs,
    parse_model_pair,
    parse_voxel_folder,
)


plt.rcParams["pdf.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-pair",
        default="ADV_set1_to_OPENCLIP_set1",
    )
    parser.add_argument("--roi", choices=("V1", "V4"), default="V1")
    parser.add_argument("--layer", default="layer2")
    parser.add_argument("--voxel-id", type=int)
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
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Write only the PNG output instead of PNG, PDF, and SVG.",
    )
    return parser.parse_args()


def select_voxel_dir(
    pair_dir: Path,
    roi: str,
    layer: str,
    requested_voxel_id: int | None,
) -> tuple[int, int, Path]:
    root = pair_dir / "gradient_hanfei" / f"{roi}_{layer}"
    candidates = []
    for path in root.glob("voxel-rank-*_id-*"):
        rank, voxel_id = parse_voxel_folder(path)
        if requested_voxel_id is None or voxel_id == requested_voxel_id:
            candidates.append((rank, voxel_id, path))
    if not candidates:
        requested = (
            f"voxel {requested_voxel_id}"
            if requested_voxel_id is not None
            else "a ranked voxel"
        )
        raise FileNotFoundError(f"Could not find {requested} in {root}")
    return min(candidates, key=lambda item: item[0])


def top_natural_images(
    nsd_root: Path,
    voxel_id: int,
    count: int = 3,
) -> tuple[list[np.ndarray], list[int]]:
    response_path = nsd_root / "data" / "S1_betas_avg_bigmask.hdf5"
    stimulus_path = nsd_root / "stimuli" / "S1_stimuli_224.h5py"
    with h5py.File(response_path, "r") as handle:
        values = np.asarray(handle["betas"][:, voxel_id], dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) < count:
        raise ValueError(f"Voxel {voxel_id} has fewer than {count} responses")
    order = finite[np.argsort(values[finite])[-count:][::-1]]
    indices = [int(value) for value in order]
    images = []
    with h5py.File(stimulus_path, "r") as handle:
        for index in indices:
            chw = np.asarray(handle["stimuli"][index])
            images.append(np.moveaxis(chw, 0, 2).astype(np.uint8))
    return images, indices


def raw_diffusion_voxel_dir(
    diffusion_root: Path,
    model_pair: str,
    roi: str,
    layer: str,
    voxel_id: int,
) -> Path:
    generator, ranker = parse_model_pair(model_pair)
    pair_dir = (
        diffusion_root
        / (
            f"gen-{diffusion_model_name(generator)}-{layer}"
            f"__rank-{diffusion_model_name(ranker)}-{layer}"
        )
    )
    run_dirs = sorted(pair_dir.glob("split-01__*"))
    if not run_dirs:
        raise FileNotFoundError(f"No set-1 diffusion run in {pair_dir}")
    diffusion_roi = "hV4" if roi == "V4" else roi
    matches = list(
        run_dirs[-1].glob(
            f"roi-{diffusion_roi}/voxel-rank-*__id-{voxel_id:06d}"
        )
    )
    if not matches:
        raise FileNotFoundError(
            f"No diffusion results for voxel {voxel_id} in {pair_dir}"
        )
    return matches[0]


def top_diffusion_images(
    voxel_dir: Path,
    count: int = 3,
) -> tuple[list[np.ndarray], float]:
    with (voxel_dir / "ranking.json").open("r", encoding="utf-8") as handle:
        ranking = json.load(handle)
    images = []
    for result in sorted(ranking["results"], key=lambda item: int(item["rank"])):
        if not (
            np.isfinite(float(result["gen_score"]))
            and np.isfinite(float(result["rank_score"]))
        ):
            continue
        path = voxel_dir / result["image_file"]
        if not path.is_file():
            continue
        with Image.open(path) as image:
            images.append(np.asarray(image.convert("RGB")))
        if len(images) == count:
            break
    if len(images) < count:
        raise RuntimeError(
            f"Found only {len(images)} finite diffusion images in {voxel_dir}"
        )
    return images, float(ranking["voxel_r2"])


def top_gradient_images(
    voxel_dir: Path,
    count: int = 3,
) -> list[np.ndarray]:
    images = []
    for rank in range(1, count + 1):
        matches = sorted(voxel_dir.glob(f"rank-{rank:02d}*__original.png"))
        if not matches:
            raise FileNotFoundError(
                f"Missing gradient rank {rank} full image in {voxel_dir}"
            )
        with Image.open(matches[0]) as image:
            images.append(np.asarray(image.convert("RGB")))
    return images


def create_figure(args: argparse.Namespace) -> Path:
    configure_font()
    pair_dir = args.share_root / args.model_pair
    voxel_rank, voxel_id, gradient_dir = select_voxel_dir(
        pair_dir,
        args.roi,
        args.layer,
        args.voxel_id,
    )
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

    generator, ranker = parse_model_pair(args.model_pair)
    prf = fitted_prfs(
        model_root=args.model_root,
        generation_model=generator,
        layer=args.layer,
        voxel_ids=[voxel_id],
    )[voxel_id]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(9.2, 9.0),
        squeeze=False,
        facecolor="white",
    )
    rows = (natural_images, diffusion_images, gradient_images)
    row_labels = ("Natural images", "Diffusion MEIs", "Gradient MEIs")

    for row, images in enumerate(rows):
        for col, full_pixels in enumerate(images):
            axis = axes[row, col]
            pixels, center, radius, x_limits, y_limits = apply_fitted_prf(
                full_pixels,
                prf,
            )
            axis.imshow(pixels, interpolation="nearest")
            axis.add_patch(
                Circle(
                    center,
                    radius,
                    edgecolor="white",
                    facecolor="none",
                    linewidth=2.2,
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
                    f"NSD image {natural_ids[col]}",
                    fontsize=17,
                    pad=8,
                )
        axes[row, 0].set_ylabel(
            row_labels[row],
            fontsize=17,
            fontweight="semibold",
            rotation=0,
            ha="right",
            va="center",
            labelpad=20,
        )

    fig.suptitle(
        f"{args.roi} {voxel_id} $R^2$={voxel_r2:.3f}, "
        f"Generation: {diffusion_model_name(generator)}, "
        f"Ranking: {diffusion_model_name(ranker)}",
        fontsize=21,
        fontweight="semibold",
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.19,
        right=0.99,
        bottom=0.02,
        top=0.91,
        wspace=0.035,
        hspace=0.075,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        args.output_dir
        / f"{args.model_pair}__{args.roi}_{args.layer}"
        f"__voxel-rank-{voxel_rank:02d}_id-{voxel_id:06d}"
        "__three-natural-three-diffusion-three-gradient"
        "__masked-zoomed-prf-circle"
    )
    suffixes = (".png",) if args.png_only else (".png", ".pdf", ".svg")
    for suffix in suffixes:
        kwargs = {
            "facecolor": "white",
            "bbox_inches": "tight",
            "pad_inches": 0.04,
        }
        if suffix == ".png":
            kwargs["dpi"] = args.dpi
        elif suffix == ".pdf":
            kwargs["format"] = "pdf"
        fig.savefig(stem.with_suffix(suffix), **kwargs)
    plt.close(fig)
    return stem.with_suffix(".png")


def main() -> None:
    args = parse_args()
    if "_set1" not in args.model_pair or "_set2" in args.model_pair:
        raise ValueError("--model-pair must use set 1")
    print(f"Saved single-voxel grid to {create_figure(args)}")


if __name__ == "__main__":
    main()
