import argparse
from dataclasses import asdict
import json
import os

import numpy as np
import torch

from max_activation_utils import (
    OptimizationConfig,
    PrfVoxelObjective,
    RMSContrastConstraint,
    batch_luminance_mean,
    batch_rms_contrast,
    build_feature_extractor,
    initialize_image_param,
    load_checkpoint,
    optimize_image,
    parse_voxel_indices,
    resolve_device,
    save_history,
    save_image_tensor,
    save_metadata,
    set_random_seed,
)


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE_REPO = os.path.abspath(
    os.path.join(PROJECT_DIR, os.pardir, "MEI")
)
DEFAULT_ADV_CHECKPOINT_PATH = (
    "/user_data/hanfeig/prf_model_weights/ADV_RN50/imagenet_l2_3_0.pt"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize an input image to maximize a fitted pRF encoding-model response."
    )
    parser.add_argument("--task_config", type=str, default=None)
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument(
        "--source_repo",
        type=str,
        default=DEFAULT_SOURCE_REPO,
        help="Read-only path to the source repository containing prf_utils.py.",
    )
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--layer_name", type=str, default=None)
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory containing best_weights.pkl, best_prf_idx.npy, and normalization files.",
    )
    parser.add_argument(
        "--best_prf_idx_path",
        type=str,
        default=None,
        help="Path to best_prf_idx.npy; the checkpoint directory is inferred from its parent.",
    )
    parser.add_argument(
        "--adv_checkpoint_path",
        type=str,
        default=DEFAULT_ADV_CHECKPOINT_PATH,
        help="Backbone checkpoint used by ADV_RN50.",
    )
    parser.add_argument("--voxel_indices", nargs="+", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--which_prf_grid", type=str, default="default-log-polar")

    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "sgd"],
        default="adam",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--init_mode", type=str, choices=["random", "gray"], default="random")
    parser.add_argument("--init_image_path", type=str, default=None)
    parser.add_argument("--direction", type=str, choices=["maximize", "minimize"], default="maximize")

    parser.add_argument("--tv_weight", type=float, default=1e-4)
    parser.add_argument("--smooth_weight", type=float, default=None)
    parser.add_argument("--l2_weight", type=float, default=1e-5)
    parser.add_argument("--jitter", type=int, default=8)
    parser.add_argument(
        "--jitter_mode",
        choices=["mean", "reflect", "roll"],
        default="mean",
    )
    parser.add_argument("--record_every", type=int, default=25)
    parser.add_argument(
        "--contrast_units",
        type=str,
        choices=["unit", "byte"],
        default="unit",
        help="'unit' means [0,1] pixels; 'byte' means [0,255] pixels.",
    )
    for stage, default_mode in (
        ("train", "max"),
        ("post_step", "max"),
        ("final", "match"),
    ):
        option = stage.replace("_", "-")
        parser.add_argument(
            f"--{option}-contrast",
            dest=f"{stage}_contrast",
            type=float,
            default=None,
        )
        parser.add_argument(
            f"--{option}-contrast-mode",
            dest=f"{stage}_contrast_mode",
            choices=["max", "match"],
            default=default_mode,
        )
        parser.add_argument(
            f"--{option}-contrast-mean",
            dest=f"{stage}_contrast_mean",
            type=float,
            default=None,
        )
    return parser

def apply_task_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.task_config is None:
        return args

    with open(args.task_config, "r") as f:
        config = json.load(f)

    if isinstance(config, list):
        defaults = {}
        tasks = config
    elif isinstance(config, dict) and "tasks" in config:
        defaults = config.get("defaults", {})
        tasks = config["tasks"]
    elif isinstance(config, dict):
        defaults = {}
        tasks = [config]
    else:
        raise ValueError("task_config must be a dict, a list, or a dict with a 'tasks' list.")

    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if task_id < 0 or task_id >= len(tasks):
        raise IndexError(f"task_id {task_id} is outside [0, {len(tasks) - 1}].")

    task = dict(defaults)
    task.update(tasks[task_id])
    for key, value in task.items():
        if not hasattr(args, key):
            raise KeyError(f"Unknown config key '{key}'.")
        setattr(args, key, value)
    args.task_id = task_id
    return args


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.smooth_weight is not None:
        args.tv_weight = args.smooth_weight
    if isinstance(args.voxel_indices, int):
        args.voxel_indices = [args.voxel_indices]

    required = ["model_name", "layer_name", "voxel_indices", "save_dir"]
    missing = [name for name in required if getattr(args, name) in (None, [])]
    if missing:
        raise ValueError(f"Missing required arguments after config merge: {', '.join(missing)}")
    if args.model_dir is None and args.best_prf_idx_path is None:
        raise ValueError("Provide either model_dir or best_prf_idx_path.")
    if args.model_name == "ADV_RN50" and not os.path.isfile(args.adv_checkpoint_path):
        raise FileNotFoundError(
            f"Cannot find ADV_RN50 backbone checkpoint: {args.adv_checkpoint_path}"
        )
    source_prf_utils = os.path.join(args.source_repo, "prf_utils.py")
    if not os.path.isfile(source_prf_utils):
        raise FileNotFoundError(f"Cannot find source repository file: {source_prf_utils}")
    return args


def build_optimization_config(args: argparse.Namespace) -> OptimizationConfig:
    unit_scale = 255.0 if args.contrast_units == "byte" else 1.0

    def constraint(stage: str) -> RMSContrastConstraint:
        target = getattr(args, f"{stage}_contrast")
        target_mean = getattr(args, f"{stage}_contrast_mean")
        return RMSContrastConstraint(
            target=None if target is None else target / unit_scale,
            mode=getattr(args, f"{stage}_contrast_mode"),
            target_mean=None if target_mean is None else target_mean / unit_scale,
        )

    config = OptimizationConfig(
        steps=args.steps,
        lr=args.lr,
        direction=args.direction,
        optimizer=args.optimizer,
        momentum=args.momentum,
        tv_weight=args.tv_weight,
        l2_weight=args.l2_weight,
        jitter=args.jitter,
        jitter_mode=args.jitter_mode,
        record_every=args.record_every,
        train_contrast=constraint("train"),
        post_step_contrast=constraint("post_step"),
        final_contrast=constraint("final"),
    )
    config.validate()
    return config


def run_one_voxel(args, checkpoint, feature_extractor, voxel_idx: int, device: torch.device) -> None:
    target = checkpoint.get_target(voxel_idx, device)
    objective = PrfVoxelObjective(
        feature_extractor=feature_extractor,
        model_name=args.model_name,
        layer_name=args.layer_name,
        targets=[target],
        device=device,
        source_repo=args.source_repo,
        which_prf_grid=args.which_prf_grid,
    )
    objective.eval()
    optimization_config = build_optimization_config(args)

    best_result = None
    best_restart = 0
    for restart_idx in range(args.num_restarts):
        restart_seed = None if args.seed is None else args.seed + restart_idx
        set_random_seed(restart_seed)
        image_param = initialize_image_param(
            init_mode=args.init_mode,
            image_size=args.image_size,
            device=device,
            init_image_path=args.init_image_path,
        )
        result = optimize_image(
            objective=objective,
            image_param=image_param,
            config=optimization_config,
        )
        if best_result is None or result.best_score > best_result.best_score:
            best_result = result
            best_restart = restart_idx

    assert best_result is not None
    voxel_prefix = f"voxel_{voxel_idx:06d}"
    png_path = os.path.join(args.save_dir, f"{voxel_prefix}_{args.direction}.png")
    npy_path = os.path.join(args.save_dir, f"{voxel_prefix}_{args.direction}.npy")
    history_path = os.path.join(args.save_dir, f"{voxel_prefix}_{args.direction}_history.csv")
    meta_path = os.path.join(args.save_dir, f"{voxel_prefix}_{args.direction}_meta.json")

    save_image_tensor(best_result.image, png_path)
    np.save(npy_path, best_result.image.detach().squeeze(0).cpu().numpy().astype(np.float32))
    save_history(best_result.history, history_path)
    save_metadata(
        {
            "model_name": args.model_name,
            "layer_name": args.layer_name,
            "source_repo": os.path.abspath(args.source_repo),
            "model_dir": checkpoint.model_dir,
            "adv_checkpoint_path": (
                os.path.abspath(args.adv_checkpoint_path)
                if args.model_name == "ADV_RN50"
                else None
            ),
            "voxel_idx": voxel_idx,
            "prf_idx": target.prf_idx,
            "direction": args.direction,
            "best_restart": best_restart,
            "best_score": best_result.best_score,
            "final_score": best_result.final_score,
            "steps": args.steps,
            "lr": args.lr,
            "tv_weight": args.tv_weight,
            "smooth_weight": args.tv_weight,
            "l2_weight": args.l2_weight,
            "jitter": args.jitter,
            "jitter_mode": args.jitter_mode,
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "contrast_units": args.contrast_units,
            "optimization_config": asdict(optimization_config),
            "saved_image_rms_contrast_unit": float(
                batch_rms_contrast(best_result.image).mean().cpu().item()
            ),
            "saved_image_rms_contrast_byte": float(
                255.0 * batch_rms_contrast(best_result.image).mean().cpu().item()
            ),
            "saved_image_luminance_mean_unit": float(
                batch_luminance_mean(best_result.image).mean().cpu().item()
            ),
            "saved_image_mean_unit": float(
                batch_luminance_mean(best_result.image).mean().cpu().item()
            ),
            "saved_image_mean_byte": float(
                255.0
                * batch_luminance_mean(best_result.image).mean().cpu().item()
            ),
            "which_prf_grid": args.which_prf_grid,
        },
        meta_path,
    )

    print(
        f"voxel={voxel_idx} prf={target.prf_idx} "
        f"best_score={best_result.best_score:.6f} saved={png_path}"
    )


def main() -> None:
    parser = build_arg_parser()
    args = finalize_args(apply_task_config(parser.parse_args()))

    set_random_seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    checkpoint = load_checkpoint(
        model_dir=args.model_dir,
        best_prf_idx_path=args.best_prf_idx_path,
    )
    voxel_indices = parse_voxel_indices(args.voxel_indices)

    print(f"Using device: {device}")
    print(f"Loading feature extractor: {args.model_name} {args.layer_name}")
    feature_extractor = build_feature_extractor(
        args.model_name,
        args.layer_name,
        device,
        adv_checkpoint_path=args.adv_checkpoint_path,
    )
    print(f"Loaded pRF checkpoint from: {checkpoint.model_dir}")

    for voxel_idx in voxel_indices:
        run_one_voxel(args, checkpoint, feature_extractor, voxel_idx, device)


if __name__ == "__main__":
    main()
