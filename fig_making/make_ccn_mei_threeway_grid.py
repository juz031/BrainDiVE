#!/usr/bin/env python3
"""Create a 3×3 natural/diffusion/gradient pRF-masked MEI grid."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle
from PIL import Image

import prf_utils


plt.rcParams["pdf.fonttype"] = 42


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SHARE_ROOT = Path(
    "/user_data/junruz/BrainDiVE/model_pair_top5_individual_rf_share"
)
DEFAULT_DIFFUSION_ROOT = Path(
    "/user_data/junruz/BrainDiVE/contrast_con_fixed/"
    "pRF_zscore_model_ranked/sub-01"
)
DEFAULT_NSD_ROOT = Path(
    "/lab_data/hendersonlab/datasets/nsd_preproc"
)
DEFAULT_MODEL_ROOT = Path("/user_data/junruz/prf_models/fixed/split_1_zscore")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figure" / "ccn_poster_mei_grid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-pair",
        default="ADV_set1_to_OPENCLIP_set1",
        help="Set-1 model-pair directory name.",
    )
    parser.add_argument("--roi", choices=("V1", "V4"), default="V1")
    parser.add_argument("--layer", default="layer2")
    parser.add_argument("--n-voxels", type=int, default=3)
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


def configure_font() -> None:
    for font_path in Path("/usr/share/fonts/urw-base35").glob(
        "NimbusSans-*.otf"
    ):
        font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "Nimbus Sans",
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.titleweight": "semibold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def parse_voxel_folder(path: Path) -> tuple[int, int]:
    match = re.fullmatch(r"voxel-rank-(\d+)_id-(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse voxel folder: {path}")
    return int(match.group(1)), int(match.group(2))


def select_voxels(
    pair_dir: Path,
    roi: str,
    layer: str,
    count: int,
) -> list[tuple[int, int, Path, Path | None]]:
    roi_layer = f"{roi}_{layer}"
    gradient_root = pair_dir / "gradient_hanfei" / roi_layer
    diffusion_root = pair_dir / "diffusion_junru_fixed" / roi_layer
    if not gradient_root.is_dir():
        raise FileNotFoundError(
            f"Missing gradient results for {pair_dir.name}/{roi_layer}"
        )

    selected = []
    for gradient_voxel_dir in gradient_root.glob("voxel-rank-*_id-*"):
        rank, voxel_id = parse_voxel_folder(gradient_voxel_dir)
        diffusion_matches = (
            list(
                diffusion_root.glob(
                    f"voxel-rank-*_id-{voxel_id:06d}"
                )
            )
            if diffusion_root.is_dir()
            else []
        )
        selected.append(
            (
                rank,
                voxel_id,
                gradient_voxel_dir,
                diffusion_matches[0] if diffusion_matches else None,
            )
        )
    selected.sort(key=lambda item: item[0])
    if len(selected) < count:
        raise RuntimeError(
            f"Found only {len(selected)} matched voxels; requested {count}"
        )
    return selected[:count]


def rank_one_original(voxel_dir: Path) -> Path:
    matches = sorted(voxel_dir.glob("rank-01*__original.png"))
    if not matches:
        raise FileNotFoundError(
            f"No rank-01 full image in {voxel_dir}"
        )
    return matches[0]


def diffusion_model_name(short_name: str) -> str:
    mapping = {
        "ADV": "ADV_RN50",
        "DINO": "DINO_RN50",
        "OPENCLIP": "OPEN_CLIP_RN50",
    }
    try:
        return mapping[short_name]
    except KeyError as error:
        raise ValueError(f"Unknown model name: {short_name}") from error


def original_diffusion_image(
    *,
    diffusion_root: Path,
    model_pair: str,
    roi: str,
    layer: str,
    voxel_id: int,
) -> tuple[Path, float]:
    match = re.fullmatch(
        r"([A-Z]+)_set1_to_([A-Z]+)_set1",
        model_pair,
    )
    if not match:
        raise ValueError(f"Cannot parse model pair: {model_pair}")
    generator = diffusion_model_name(match.group(1))
    ranker = diffusion_model_name(match.group(2))
    pair_dir = (
        diffusion_root
        / f"gen-{generator}-{layer}__rank-{ranker}-{layer}"
    )
    run_dirs = sorted(pair_dir.glob("split-01__*"))
    if not run_dirs:
        raise FileNotFoundError(f"No set-1 diffusion run in {pair_dir}")
    diffusion_roi = "hV4" if roi == "V4" else roi
    voxel_matches = list(
        run_dirs[-1].glob(
            f"roi-{diffusion_roi}/voxel-rank-*__id-{voxel_id:06d}"
        )
    )
    if not voxel_matches:
        raise FileNotFoundError(
            f"No diffusion result for voxel {voxel_id} in {pair_dir}"
        )
    ranking_path = voxel_matches[0] / "ranking.json"
    with ranking_path.open("r", encoding="utf-8") as handle:
        ranking = json.load(handle)
    for result in sorted(
        ranking["results"],
        key=lambda item: int(item["rank"]),
    ):
        if not (
            np.isfinite(float(result["gen_score"]))
            and np.isfinite(float(result["rank_score"]))
        ):
            continue
        image_path = voxel_matches[0] / result["image_file"]
        if image_path.is_file():
            return image_path, float(ranking["voxel_r2"])
    raise FileNotFoundError(
        f"No finite diffusion image in {voxel_matches[0]}"
    )


def load_top_natural_images(
    nsd_root: Path,
    voxel_ids: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    response_path = nsd_root / "data" / "S1_betas_avg_bigmask.hdf5"
    stimulus_path = nsd_root / "stimuli" / "S1_stimuli_224.h5py"

    top_indices = {}
    with h5py.File(response_path, "r") as handle:
        responses = handle["betas"]
        for voxel_id in voxel_ids:
            values = np.asarray(responses[:, voxel_id], dtype=np.float64)
            finite = np.flatnonzero(np.isfinite(values))
            if not len(finite):
                raise ValueError(
                    f"Voxel {voxel_id} has no finite natural responses"
                )
            top_indices[voxel_id] = int(
                finite[np.argmax(values[finite])]
            )

    images = {}
    with h5py.File(stimulus_path, "r") as handle:
        stimuli = handle["stimuli"]
        for voxel_id, stimulus_index in top_indices.items():
            chw = np.asarray(stimuli[stimulus_index])
            images[voxel_id] = np.moveaxis(chw, 0, 2).astype(np.uint8)
    return images, top_indices


def parse_model_pair(model_pair: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"([A-Z]+)_set1_to_([A-Z]+)_set1",
        model_pair,
    )
    if not match:
        raise ValueError(f"Cannot parse model pair: {model_pair}")
    return match.group(1), match.group(2)


def fitted_prfs(
    *,
    model_root: Path,
    generation_model: str,
    layer: str,
    voxel_ids: list[int],
) -> dict[int, np.ndarray]:
    model_dir = (
        model_root
        / "S1"
        / f"{diffusion_model_name(generation_model)}_set1"
        / layer
    )
    best_prf_path = model_dir / "best_prf_idx.npy"
    if not best_prf_path.is_file():
        raise FileNotFoundError(
            f"Missing fitted generation-model pRFs: {best_prf_path}"
        )
    best_prf_idx = np.load(best_prf_path)
    prf_grid, _ = prf_utils.get_prf_grid(grid_name="default-log-polar")
    return {
        voxel_id: np.asarray(prf_grid[int(best_prf_idx[voxel_id])])
        for voxel_id in voxel_ids
    }


def apply_fitted_prf(
    image_array: np.ndarray,
    prf_parameters: np.ndarray,
) -> tuple[
    np.ndarray,
    tuple[float, float],
    float,
    tuple[float, float],
    tuple[float, float],
]:
    image = Image.fromarray(image_array).convert("RGB").resize(
        (224, 224),
        Image.Resampling.BILINEAR,
    )
    pixels = np.asarray(image, dtype=np.float32)
    x, y, sigma = (float(value) for value in prf_parameters)
    gaussian = prf_utils.gauss_2d(
        center=[x, y],
        sd=sigma,
        patch_size=224,
    )
    gaussian = gaussian / np.max(gaussian)
    masked = np.clip(
        pixels * gaussian[:, :, np.newaxis],
        0,
        255,
    ).astype(np.uint8)
    center = (112 + x * 224, 112 - y * 224)
    radius = 2 * sigma * 224
    crop_radius = 1.2 * radius
    x_limits = (
        center[0] - crop_radius,
        center[0] + crop_radius,
    )
    y_limits = (
        center[1] + crop_radius,
        center[1] - crop_radius,
    )
    return masked, center, radius, x_limits, y_limits


def create_grid(
    *,
    pair_dir: Path,
    roi: str,
    layer: str,
    voxels: list[tuple[int, int, Path, Path | None]],
    natural_images: dict[int, np.ndarray],
    natural_indices: dict[int, int],
    diffusion_root: Path,
    model_root: Path,
    output_dir: Path,
    dpi: int,
) -> Path:
    configure_font()
    n_rows = len(voxels)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(9.2, 9.2),
        squeeze=False,
        facecolor="white",
    )
    column_titles = (
        "Top natural image",
        "Diffusion MEI",
        "Gradient MEI",
    )
    generator, ranker = parse_model_pair(pair_dir.name)
    prf_parameters = fitted_prfs(
        model_root=model_root,
        generation_model=generator,
        layer=layer,
        voxel_ids=[item[1] for item in voxels],
    )

    for row, (voxel_rank, voxel_id, gradient_dir, diffusion_dir) in enumerate(
        voxels
    ):
        if diffusion_dir is not None:
            with Image.open(rank_one_original(diffusion_dir)) as image:
                diffusion_original = np.asarray(image.convert("RGB"))
            _, voxel_r2 = original_diffusion_image(
                diffusion_root=diffusion_root,
                model_pair=pair_dir.name,
                roi=roi,
                layer=layer,
                voxel_id=voxel_id,
            )
        else:
            diffusion_path, voxel_r2 = original_diffusion_image(
                diffusion_root=diffusion_root,
                model_pair=pair_dir.name,
                roi=roi,
                layer=layer,
                voxel_id=voxel_id,
            )
            with Image.open(diffusion_path) as image:
                diffusion_original = np.asarray(image.convert("RGB"))
        with Image.open(rank_one_original(gradient_dir)) as image:
            gradient_original = np.asarray(image.convert("RGB"))

        for col, full_pixels in enumerate(
            (
                natural_images[voxel_id],
                diffusion_original,
                gradient_original,
            )
        ):
            axis = axes[row, col]
            (
                pixels,
                center,
                radius,
                x_limits,
                y_limits,
            ) = apply_fitted_prf(
                full_pixels,
                prf_parameters[voxel_id],
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
                axis.set_title(column_titles[col], pad=12)

        axes[row, 0].set_ylabel(
            f"Voxel {voxel_id}\nRank {voxel_rank}\n$R^2$ = {voxel_r2:.3f}",
            fontsize=16,
            fontweight="semibold",
            rotation=0,
            ha="right",
            va="center",
            labelpad=18,
        )
        axes[row, 0].text(
            0.5,
            -0.055,
            f"NSD image {natural_indices[voxel_id]}",
            transform=axes[row, 0].transAxes,
            ha="center",
            va="top",
            fontsize=11,
            color="#59636D",
        )

    fig.suptitle(
        f"{roi}, Generation model: {diffusion_model_name(generator)}, "
        f"Ranking model: {diffusion_model_name(ranker)}",
        fontsize=22,
        fontweight="semibold",
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.20,
        right=0.985,
        bottom=0.025,
        top=0.91,
        wspace=0.035,
        hspace=0.12,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        output_dir
        / f"{pair_dir.name}__{roi}_{layer}__top-{n_rows}"
        "__natural-diffusion-gradient__masked-zoomed"
    )
    for suffix in (".png", ".pdf", ".svg"):
        kwargs = {
            "facecolor": "white",
            "bbox_inches": "tight",
            "pad_inches": 0.04,
        }
        if suffix == ".png":
            kwargs["dpi"] = dpi
        elif suffix == ".pdf":
            kwargs["format"] = "pdf"
        fig.savefig(stem.with_suffix(suffix), **kwargs)
    plt.close(fig)
    return stem.with_suffix(".png")


def main() -> None:
    args = parse_args()
    if "_set1" not in args.model_pair or "_set2" in args.model_pair:
        raise ValueError("--model-pair must use set 1")
    pair_dir = args.share_root / args.model_pair
    voxels = select_voxels(
        pair_dir,
        args.roi,
        args.layer,
        args.n_voxels,
    )
    voxel_ids = [item[1] for item in voxels]
    natural_images, natural_indices = load_top_natural_images(
        args.nsd_root,
        voxel_ids,
    )
    output_path = create_grid(
        pair_dir=pair_dir,
        roi=args.roi,
        layer=args.layer,
        voxels=voxels,
        natural_images=natural_images,
        natural_indices=natural_indices,
        diffusion_root=args.diffusion_root,
        model_root=args.model_root,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )
    print(f"Saved three-way MEI grid to {output_path}")


if __name__ == "__main__":
    main()
