import csv
import importlib.util
import json
import os
import pickle
import random
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor


@dataclass(frozen=True)
class ModelConfig:
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    layer_shapes: Dict[str, Tuple[int, int, int]]
    return_nodes: Dict[str, str]


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RN50_PRF_LAYERS = {
    "layer1": (256, 56, 56),
    "layer2": (512, 28, 28),
    "layer3": (1024, 14, 14),
    "layer4": (2048, 7, 7),
}

MODEL_CONFIGS = {
    "CLIP_RN50": ModelConfig(
        mean=CLIP_MEAN,
        std=CLIP_STD,
        layer_shapes={
            "relu1": (32, 112, 112),
            "relu2": (32, 112, 112),
            "relu3": (64, 112, 112),
            "avgpool": (64, 56, 56),
            **RN50_PRF_LAYERS,
        },
        return_nodes={
            "relu1": "relu1",
            "relu2": "relu2",
            "relu3": "relu3",
            "avgpool": "avgpool",
            "layer1": "layer1",
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4",
        },
    ),
    "OPEN_CLIP_RN50": ModelConfig(
        mean=CLIP_MEAN,
        std=CLIP_STD,
        layer_shapes={
            "act1": (32, 112, 112),
            "act2": (32, 112, 112),
            "act3": (64, 112, 112),
            "avgpool": (64, 56, 56),
            **RN50_PRF_LAYERS,
        },
        return_nodes={
            "act1": "act1",
            "act2": "act2",
            "act3": "act3",
            "avgpool": "avgpool",
            "layer1": "layer1",
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4",
        },
    ),
    "DINO_RN50": ModelConfig(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        layer_shapes={
            "relu": (64, 112, 112),
            "maxpool": (64, 56, 56),
            **RN50_PRF_LAYERS,
        },
        return_nodes={
            "relu": "relu",
            "maxpool": "maxpool",
            "layer1": "layer1",
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4",
        },
    ),
    "SIMCLR_RN50": ModelConfig(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        layer_shapes={
            "relu": (64, 112, 112),
            "maxpool": (64, 56, 56),
            **RN50_PRF_LAYERS,
        },
        return_nodes={
            "relu": "relu",
            "maxpool": "maxpool",
            "layer1": "layer1",
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4",
        },
    ),
    "ADV_RN50": ModelConfig(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        layer_shapes={
            "relu": (64, 112, 112),
            "maxpool": (64, 56, 56),
            **RN50_PRF_LAYERS,
        },
        return_nodes={
            "relu": "relu",
            "maxpool": "maxpool",
            "layer1": "layer1",
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4",
        },
    ),
    "OPEN_CLIP_CONVNEXT_BASE": ModelConfig(
        mean=CLIP_MEAN,
        std=CLIP_STD,
        layer_shapes={
            "stem": (128, 56, 56),
            "stage1": (128, 56, 56),
            "stage2": (256, 28, 28),
            "stage3": (512, 14, 14),
            "stage4": (1024, 7, 7),
        },
        return_nodes={
            "stem": "trunk.stem.1.permute_1",
            "stage1": "trunk.stages.0.blocks.2.add",
            "stage2": "trunk.stages.1.blocks.2.add",
            "stage3": "trunk.stages.2.blocks.26.add",
            "stage4": "trunk.stages.3.blocks.2.add",
        },
    ),
}


@dataclass
class VoxelTarget:
    voxel_idx: int
    prf_idx: int
    weights: torch.Tensor
    intercept: torch.Tensor
    features_m: Optional[torch.Tensor]
    features_s: Optional[torch.Tensor]
    channel_kept: Optional[torch.Tensor]


@dataclass
class OptimizationResult:
    image: torch.Tensor
    best_score: float
    final_score: float
    history: List[Dict[str, float]]


@dataclass
class BatchOptimizationResult:
    images: torch.Tensor
    best_scores: torch.Tensor
    final_scores: torch.Tensor


@dataclass(frozen=True)
class RMSContrastConstraint:
    """Per-image RMS contrast constraint in [0, 1] pixel space."""

    target: Optional[float] = None
    mode: str = "max"
    target_mean: Optional[float] = None
    eps: float = 1e-8
    match_iterations: int = 50

    @property
    def enabled(self) -> bool:
        return self.target is not None

    def validate(self) -> None:
        if self.mode not in {"max", "match"}:
            raise ValueError("Contrast constraint mode must be 'max' or 'match'.")
        if self.target is None and self.target_mean is not None:
            raise ValueError(
                "Contrast target_mean requires an enabled contrast target."
            )
        if self.target is not None and not 0.0 < self.target <= 0.5:
            raise ValueError(
                "RMS contrast target must be in (0, 0.5] for [0, 1] pixels, "
                f"got {self.target}."
            )
        if self.target_mean is not None and not 0.0 <= self.target_mean <= 1.0:
            raise ValueError(
                "Contrast target_mean must be in [0, 1], "
                f"got {self.target_mean}."
            )
        if self.eps <= 0.0:
            raise ValueError("Contrast constraint eps must be positive.")
        if self.match_iterations <= 0:
            raise ValueError("Contrast match_iterations must be positive.")


@dataclass(frozen=True)
class OptimizationConfig:
    """Composable configuration for one MEI optimization algorithm."""

    steps: int = 500
    lr: float = 0.05
    direction: str = "maximize"
    optimizer: str = "adam"
    momentum: float = 0.9
    tv_weight: float = 1e-4
    l2_weight: float = 1e-5
    jitter: int = 8
    jitter_mode: str = "mean"
    record_every: int = 25
    train_contrast: RMSContrastConstraint = RMSContrastConstraint()
    post_step_contrast: RMSContrastConstraint = RMSContrastConstraint()
    final_contrast: RMSContrastConstraint = RMSContrastConstraint(mode="match")

    def validate(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must be non-negative.")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be 'maximize' or 'minimize'.")
        if self.optimizer not in {"adam", "adamw", "sgd"}:
            raise ValueError("optimizer must be 'adam', 'adamw', or 'sgd'.")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if self.tv_weight < 0.0 or self.l2_weight < 0.0:
            raise ValueError("Regularization weights must be non-negative.")
        if self.jitter < 0:
            raise ValueError("jitter must be non-negative.")
        if self.jitter_mode not in {"mean", "reflect", "roll"}:
            raise ValueError(
                "jitter_mode must be 'mean', 'reflect', or legacy 'roll'."
            )
        if self.record_every <= 0:
            raise ValueError("record_every must be positive.")
        self.train_contrast.validate()
        self.post_step_contrast.validate()
        self.final_contrast.validate()


def get_model_config(model_name: str) -> ModelConfig:
    if model_name not in MODEL_CONFIGS:
        supported = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unsupported model_name={model_name}. Supported: {supported}")
    return MODEL_CONFIGS[model_name]


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def set_random_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_dir(
    model_dir: Optional[str] = None,
    best_prf_idx_path: Optional[str] = None,
) -> str:
    if model_dir is None and best_prf_idx_path is None:
        raise ValueError("Provide either model_dir or best_prf_idx_path.")
    if model_dir is None:
        model_dir = os.path.dirname(os.path.abspath(best_prf_idx_path))
    if os.path.basename(model_dir) == "best_prf_idx.npy":
        model_dir = os.path.dirname(model_dir)
    return os.path.abspath(model_dir)


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_optional_array_or_pickle(model_dir: str, stem: str):
    pkl_path = os.path.join(model_dir, f"{stem}.pkl")
    npy_path = os.path.join(model_dir, f"{stem}.npy")
    if os.path.exists(pkl_path):
        return _load_pickle(pkl_path)
    if os.path.exists(npy_path):
        return np.load(npy_path, allow_pickle=True)
    return None


class FittedPrfCheckpoint:
    def __init__(self, model_dir: str):
        self.model_dir = os.path.abspath(model_dir)
        self.best_prf_idx = np.load(os.path.join(self.model_dir, "best_prf_idx.npy"))
        self.best_weights = _load_pickle(os.path.join(self.model_dir, "best_weights.pkl"))
        self.channel_kept = _load_optional_array_or_pickle(self.model_dir, "channel_kept")
        self.features_m = _load_optional_array_or_pickle(self.model_dir, "best_features_m")
        self.features_s = _load_optional_array_or_pickle(self.model_dir, "best_features_s")

    def __len__(self) -> int:
        return int(self.best_prf_idx.shape[0])

    def _lookup(self, obj, voxel_idx: int):
        if obj is None:
            return None
        if isinstance(obj, dict):
            if voxel_idx in obj:
                return obj[voxel_idx]
            if str(voxel_idx) in obj:
                return obj[str(voxel_idx)]
            return None
        return obj[voxel_idx]

    def get_target(self, voxel_idx: int, device: torch.device) -> VoxelTarget:
        if voxel_idx < 0 or voxel_idx >= len(self):
            raise IndexError(f"voxel_idx {voxel_idx} is outside [0, {len(self) - 1}].")

        weights_with_intercept = np.asarray(self.best_weights[str(voxel_idx)], dtype=np.float32)
        weights = torch.as_tensor(weights_with_intercept[:-1], device=device)
        intercept = torch.as_tensor(weights_with_intercept[-1], device=device)

        features_m = self._to_1d_tensor(self._lookup(self.features_m, voxel_idx), device)
        features_s = self._to_1d_tensor(self._lookup(self.features_s, voxel_idx), device)
        channel_kept = self._lookup(self.channel_kept, voxel_idx)
        if channel_kept is not None:
            channel_kept = torch.as_tensor(channel_kept, dtype=torch.long, device=device)

        return VoxelTarget(
            voxel_idx=voxel_idx,
            prf_idx=int(self.best_prf_idx[voxel_idx]),
            weights=weights,
            intercept=intercept,
            features_m=features_m,
            features_s=features_s,
            channel_kept=channel_kept,
        )

    @staticmethod
    def _to_1d_tensor(value, device: torch.device) -> Optional[torch.Tensor]:
        if value is None:
            return None
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        return torch.as_tensor(value, device=device)


def load_checkpoint(
    model_dir: Optional[str] = None,
    best_prf_idx_path: Optional[str] = None,
) -> FittedPrfCheckpoint:
    return FittedPrfCheckpoint(resolve_model_dir(model_dir, best_prf_idx_path))


def build_feature_extractor(
    model_name: str,
    layer_name: str,
    device: torch.device,
    adv_checkpoint_path: Optional[str] = None,
) -> nn.Module:
    config = get_model_config(model_name)
    if layer_name not in config.return_nodes:
        supported = ", ".join(config.return_nodes.keys())
        raise ValueError(f"Unsupported layer_name={layer_name} for {model_name}. Supported: {supported}")

    if model_name == "CLIP_RN50":
        import clip

        full_model, _ = clip.load("RN50", device=device, jit=False)
        backbone = full_model.visual
    elif model_name == "OPEN_CLIP_RN50":
        import open_clip

        full_model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="yfcc15m")
        backbone = full_model.visual
    elif model_name == "DINO_RN50":
        backbone = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
    elif model_name == "SIMCLR_RN50":
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(
            repo_id="lightly-ai/simclrv1-imagenet1k-resnet50-1x",
            filename="resnet50-1x.pth",
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        backbone = resnet50(weights=None)
        backbone.load_state_dict(state_dict, strict=True)
    elif model_name == "ADV_RN50":
        if not adv_checkpoint_path:
            raise ValueError("ADV_RN50 requires adv_checkpoint_path.")
        adv_checkpoint_path = os.path.abspath(os.path.expanduser(adv_checkpoint_path))
        if not os.path.isfile(adv_checkpoint_path):
            raise FileNotFoundError(f"Cannot find ADV_RN50 checkpoint: {adv_checkpoint_path}")

        try:
            checkpoint = torch.load(
                adv_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(adv_checkpoint_path, map_location="cpu")

        state_dict = checkpoint.get("model", checkpoint)
        prefix = "module.model."
        backbone_state_dict = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not backbone_state_dict:
            raise ValueError(
                "ADV_RN50 checkpoint does not contain keys prefixed with "
                f"'{prefix}': {adv_checkpoint_path}"
            )

        backbone = resnet50(weights=None)
        backbone.load_state_dict(backbone_state_dict, strict=True)
    elif model_name == "OPEN_CLIP_CONVNEXT_BASE":
        import open_clip

        full_model, _, _ = open_clip.create_model_and_transforms(
            "convnext_base",
            pretrained="laion400m_s13b_b51k",
        )
        backbone = full_model.visual
    else:
        raise ValueError(f"Unsupported model_name={model_name}")

    return_nodes = {config.return_nodes[layer_name]: layer_name}
    feature_extractor = create_feature_extractor(backbone, return_nodes=return_nodes)
    feature_extractor.eval().to(device)
    for param in feature_extractor.parameters():
        param.requires_grad_(False)
    return feature_extractor


def _ensure_bchw(feature_map: torch.Tensor, expected_shape: Tuple[int, int, int]) -> torch.Tensor:
    expected_c, expected_h, expected_w = expected_shape
    if feature_map.ndim != 4:
        raise ValueError(f"Expected 4D feature map, got shape {tuple(feature_map.shape)}")
    if feature_map.shape[1:] == (expected_c, expected_h, expected_w):
        return feature_map
    if feature_map.shape[1] == expected_c:
        return feature_map
    if feature_map.shape[-1] == expected_c:
        return feature_map.permute(0, 3, 1, 2).contiguous()
    raise ValueError(
        "Cannot infer feature layout. "
        f"Got {tuple(feature_map.shape)}, expected channels={expected_c}."
    )


@lru_cache(maxsize=None)
def load_source_prf_utils(source_repo: str) -> ModuleType:
    source_repo = os.path.abspath(os.path.expanduser(source_repo))
    module_path = os.path.join(source_repo, "prf_utils.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"Cannot find source pRF utility module: {module_path}")

    spec = importlib.util.spec_from_file_location("source_prf_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load source pRF utility module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_prf_mask(
    model_name: str,
    layer_name: str,
    prf_idx: int,
    device: torch.device,
    source_repo: str,
    which_prf_grid: str = "default-log-polar",
) -> torch.Tensor:
    _, height, width = get_model_config(model_name).layer_shapes[layer_name]
    if height != width:
        raise ValueError(f"Only square pRF feature maps are supported, got {height}x{width}.")
    prf_utils = load_source_prf_utils(source_repo)
    prf_stack = prf_utils.get_prf_stack(n_pix=height, which_prf_grid=which_prf_grid)
    if prf_idx < 0 or prf_idx >= prf_stack.shape[-1]:
        raise IndexError(f"prf_idx {prf_idx} is outside [0, {prf_stack.shape[-1] - 1}].")
    mask = torch.as_tensor(prf_stack[:, :, prf_idx], dtype=torch.float32, device=device)
    return mask.view(1, 1, height, width)


class PrfVoxelObjective(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        model_name: str,
        layer_name: str,
        targets: Sequence[VoxelTarget],
        device: torch.device,
        source_repo: str,
        which_prf_grid: str = "default-log-polar",
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.model_name = model_name
        self.layer_name = layer_name
        self.targets = list(targets)
        self.config = get_model_config(model_name)
        self.expected_shape = self.config.layer_shapes[layer_name]

        mean = torch.tensor(self.config.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(self.config.std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean)
        self.register_buffer("image_std", std)

        prf_masks = {
            target.prf_idx: make_prf_mask(
                model_name=model_name,
                layer_name=layer_name,
                prf_idx=target.prf_idx,
                device=device,
                source_repo=source_repo,
                which_prf_grid=which_prf_grid,
            )
            for target in self.targets
        }
        self.prf_masks = nn.ParameterDict(
            {
                str(prf_idx): nn.Parameter(mask, requires_grad=False)
                for prf_idx, mask in prf_masks.items()
            }
        )

    def normalize(self, image_01: torch.Tensor) -> torch.Tensor:
        return (image_01 - self.image_mean) / self.image_std

    def forward(self, image_01: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(self.normalize(image_01))[self.layer_name].float()
        features = _ensure_bchw(features, self.expected_shape)

        responses = []
        for target in self.targets:
            mask = self.prf_masks[str(target.prf_idx)]
            pooled = (features * mask).sum(dim=(2, 3))
            if target.channel_kept is not None:
                pooled = pooled.index_select(dim=1, index=target.channel_kept)
            if target.features_m is not None and target.features_s is not None:
                pooled = (pooled - target.features_m) / target.features_s
            if pooled.shape[1] != target.weights.shape[0]:
                raise ValueError(
                    f"Feature/weight mismatch for voxel {target.voxel_idx}: "
                    f"{pooled.shape[1]} features vs {target.weights.shape[0]} weights."
                )
            responses.append(pooled @ target.weights + target.intercept)
        return torch.stack(responses, dim=1)


def total_variation(image: torch.Tensor) -> torch.Tensor:
    tv_h = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    return tv_h + tv_w


def batch_total_variation(image: torch.Tensor) -> torch.Tensor:
    """Total variation reduced per image rather than across the batch."""
    tv_h = torch.mean(
        torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]),
        dim=(1, 2, 3),
    )
    tv_w = torch.mean(
        torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]),
        dim=(1, 2, 3),
    )
    return tv_h + tv_w


def image_luminance(image: torch.Tensor) -> torch.Tensor:
    """Convert BCHW RGB values to Rec.709 luma, preserving one-channel input."""
    if image.ndim != 4:
        raise ValueError(f"Expected a batched image tensor, got shape {tuple(image.shape)}.")
    if image.shape[1] == 1:
        return image
    if image.shape[1] != 3:
        raise ValueError(
            "Luminance conversion expects one or three channels, "
            f"got {image.shape[1]}."
        )
    weights = image.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def batch_luminance_mean(
    image: torch.Tensor,
    keepdim: bool = False,
) -> torch.Tensor:
    luminance = image_luminance(image)
    mean = luminance.mean(dim=(-2, -1), keepdim=True)
    if keepdim:
        return mean
    return mean.reshape(image.shape[0])


def batch_rms_contrast(image: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """Population standard deviation of Rec.709 luminance per image."""
    luminance = image_luminance(image)
    image_mean = luminance.mean(dim=(-2, -1), keepdim=True)
    contrast = torch.sqrt(
        torch.mean((luminance - image_mean) ** 2, dim=(-2, -1), keepdim=True)
    )
    if keepdim:
        return contrast
    return contrast.reshape(image.shape[0])


def apply_rms_contrast_constraint(
    image_01: torch.Tensor,
    constraint: RMSContrastConstraint,
) -> torch.Tensor:
    """Apply a differentiable per-image hard projection in [0, 1] pixel space."""
    constraint.validate()
    if not constraint.enabled:
        return image_01

    dims = (-2, -1)
    current_mean = batch_luminance_mean(image_01, keepdim=True)
    current_contrast = batch_rms_contrast(image_01, keepdim=True)
    target = torch.as_tensor(
        constraint.target,
        dtype=image_01.dtype,
        device=image_01.device,
    )
    scale = target / current_contrast.clamp_min(constraint.eps)
    if constraint.mode == "max":
        scale = torch.minimum(scale, torch.ones_like(scale))

    if constraint.target_mean is None:
        output_mean = current_mean
    else:
        output_mean = torch.as_tensor(
            constraint.target_mean,
            dtype=image_01.dtype,
            device=image_01.device,
        )
    constrained = (image_01 - current_mean) * scale + output_mean
    if constraint.mode == "max":
        return constrained.clamp(0.0, 1.0)

    # Clipping can move both the requested mean and RMS contrast. Repeating the
    # affine adjustment compensates for that effect, like force_stats in the
    # reference MEI implementation, while remaining differentiable.
    for _ in range(constraint.match_iterations):
        constrained = constrained.clamp(0.0, 1.0)
        constrained_mean = batch_luminance_mean(constrained, keepdim=True)
        constrained_contrast = batch_rms_contrast(constrained, keepdim=True)
        constrained = (
            (constrained - constrained_mean)
            * (target / constrained_contrast.clamp_min(constraint.eps))
            + output_mean
        )
    return constrained.clamp(0.0, 1.0)


def _image_to_logit(image_01: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    image_01 = image_01.clamp(eps, 1.0 - eps)
    return torch.log(image_01 / (1.0 - image_01))


def translate_image(
    image: torch.Tensor,
    shift_y: int,
    shift_x: int,
    mode: str = "mean",
) -> torch.Tensor:
    """Translate an image with configurable, differentiable boundary handling."""
    if mode == "roll":
        return torch.roll(image, shifts=(shift_y, shift_x), dims=(-2, -1))
    if mode not in {"mean", "reflect"}:
        raise ValueError("Translation mode must be 'mean', 'reflect', or 'roll'.")

    height, width = image.shape[-2:]
    if abs(shift_y) >= height or abs(shift_x) >= width:
        raise ValueError(
            f"Shift ({shift_y}, {shift_x}) must be smaller than image {height}x{width}."
        )

    pad_y = abs(shift_y)
    pad_x = abs(shift_x)
    if mode == "reflect":
        padded = torch.nn.functional.pad(
            image,
            (pad_x, pad_x, pad_y, pad_y),
            mode="reflect",
        )
        top = pad_y - shift_y
        left = pad_x - shift_x
        return padded[..., top : top + height, left : left + width]

    translated = torch.roll(
        image,
        shifts=(shift_y, shift_x),
        dims=(-2, -1),
    )
    valid = torch.ones(
        (image.shape[0], 1, height, width),
        dtype=image.dtype,
        device=image.device,
    )
    if shift_y > 0:
        valid[..., :shift_y, :] = 0.0
    elif shift_y < 0:
        valid[..., shift_y:, :] = 0.0
    if shift_x > 0:
        valid[..., :, :shift_x] = 0.0
    elif shift_x < 0:
        valid[..., :, shift_x:] = 0.0
    channel_mean = image.mean(dim=(-2, -1), keepdim=True)
    return translated * valid + channel_mean * (1.0 - valid)


def initialize_image_param(
    init_mode: str,
    image_size: int,
    device: torch.device,
    init_image_path: Optional[str] = None,
) -> torch.Tensor:
    if init_image_path:
        image = Image.open(init_image_path).convert("RGB").resize((image_size, image_size))
        image_np = np.asarray(image).astype(np.float32) / 255.0
        image_01 = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
    elif init_mode == "gray":
        image_01 = torch.full((1, 3, image_size, image_size), 0.5, device=device)
    elif init_mode == "random":
        image_01 = torch.rand((1, 3, image_size, image_size), device=device) * 0.6 + 0.2
    else:
        raise ValueError("init_mode must be 'random' or 'gray'.")

    image_param = _image_to_logit(image_01)
    image_param.requires_grad_(True)
    return image_param


def _build_optimizer(
    image_param: torch.Tensor,
    config: OptimizationConfig,
) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam([image_param], lr=config.lr)
    if config.optimizer == "adamw":
        return torch.optim.AdamW([image_param], lr=config.lr)
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            [image_param],
            lr=config.lr,
            momentum=config.momentum,
        )
    raise AssertionError(f"Unhandled optimizer: {config.optimizer}")


def _project_image_param(
    image_param: torch.Tensor,
    constraint: RMSContrastConstraint,
) -> None:
    if not constraint.enabled:
        return
    with torch.no_grad():
        image_01 = torch.sigmoid(image_param)
        projected = apply_rms_contrast_constraint(image_01, constraint)
        image_param.copy_(_image_to_logit(projected))


def optimize_image(
    objective: PrfVoxelObjective,
    image_param: torch.Tensor,
    config: Optional[OptimizationConfig] = None,
) -> OptimizationResult:
    config = config or OptimizationConfig()
    config.validate()
    direction_sign = 1.0 if config.direction == "maximize" else -1.0

    optimizer = _build_optimizer(image_param, config)
    history: List[Dict[str, float]] = []
    best_score = -float("inf")
    best_image = None
    final_score = float("nan")

    for step in range(config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        image_01 = torch.sigmoid(image_param)
        model_image = apply_rms_contrast_constraint(
            image_01,
            config.train_contrast,
        )

        if config.jitter > 0:
            shift_y = random.randint(-config.jitter, config.jitter)
            shift_x = random.randint(-config.jitter, config.jitter)
            model_image = translate_image(
                model_image,
                shift_y=shift_y,
                shift_x=shift_x,
                mode=config.jitter_mode,
            )

        responses = objective(model_image)
        raw_score = responses.mean()
        target_score = direction_sign * raw_score
        regularizer = (
            config.tv_weight * total_variation(image_01)
            + config.l2_weight * torch.mean((image_01 - 0.5) ** 2)
        )
        loss = -target_score + regularizer
        loss.backward()
        optimizer.step()
        _project_image_param(image_param, config.post_step_contrast)

        if step % config.record_every == 0 or step == config.steps:
            with torch.no_grad():
                image_eval = torch.sigmoid(image_param)
                model_image_eval = apply_rms_contrast_constraint(
                    image_eval,
                    config.train_contrast,
                )
                eval_score = direction_sign * objective(model_image_eval).mean()
                final_score = float(eval_score.detach().cpu().item())
                history.append(
                    {
                        "step": float(step),
                        "score": final_score,
                        "loss": float(loss.detach().cpu().item()),
                        "tv": float(total_variation(image_eval).detach().cpu().item()),
                        "image_rms_contrast": float(
                            batch_rms_contrast(image_eval).mean().detach().cpu().item()
                        ),
                        "model_rms_contrast": float(
                            batch_rms_contrast(model_image_eval).mean().detach().cpu().item()
                        ),
                    }
                )
                if final_score > best_score:
                    best_score = final_score
                    best_image = image_eval.detach().clone()

    assert best_image is not None
    best_image = apply_rms_contrast_constraint(
        best_image,
        config.final_contrast,
    ).detach()
    with torch.no_grad():
        final_model_image = apply_rms_contrast_constraint(
            best_image,
            config.train_contrast,
        )
        best_score = float(
            (direction_sign * objective(final_model_image).mean()).detach().cpu().item()
        )
    return OptimizationResult(
        image=best_image,
        best_score=best_score,
        final_score=final_score,
        history=history,
    )


def optimize_image_batch(
    objective: PrfVoxelObjective,
    image_param: torch.Tensor,
    config: Optional[OptimizationConfig] = None,
) -> BatchOptimizationResult:
    """Optimize independent images together for high-throughput seed sweeps."""
    config = config or OptimizationConfig()
    config.validate()
    if config.jitter != 0:
        raise ValueError(
            "Batched optimization currently requires jitter=0 so each seed "
            "has an independent, fixed spatial objective."
        )
    direction_sign = 1.0 if config.direction == "maximize" else -1.0
    optimizer = _build_optimizer(image_param, config)
    batch_size = image_param.shape[0]
    best_scores = torch.full(
        (batch_size,),
        -float("inf"),
        dtype=image_param.dtype,
        device=image_param.device,
    )
    best_images = torch.empty_like(image_param)
    final_scores = torch.full_like(best_scores, float("nan"))

    for step in range(config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        image_01 = torch.sigmoid(image_param)
        model_image = apply_rms_contrast_constraint(
            image_01,
            config.train_contrast,
        )
        raw_scores = objective(model_image).reshape(batch_size, -1).mean(dim=1)
        target_scores = direction_sign * raw_scores
        regularizers = (
            config.tv_weight * batch_total_variation(image_01)
            + config.l2_weight
            * torch.mean((image_01 - 0.5) ** 2, dim=(1, 2, 3))
        )
        loss = torch.mean(-target_scores + regularizers)
        loss.backward()
        optimizer.step()
        _project_image_param(image_param, config.post_step_contrast)

        if step % config.record_every == 0 or step == config.steps:
            with torch.no_grad():
                image_eval = torch.sigmoid(image_param)
                model_image_eval = apply_rms_contrast_constraint(
                    image_eval,
                    config.train_contrast,
                )
                eval_scores = direction_sign * objective(model_image_eval)
                eval_scores = eval_scores.reshape(batch_size, -1).mean(dim=1)
                final_scores = eval_scores
                improved = eval_scores > best_scores
                best_scores = torch.where(improved, eval_scores, best_scores)
                best_images[improved] = image_eval[improved]

    with torch.no_grad():
        best_images = apply_rms_contrast_constraint(
            best_images,
            config.final_contrast,
        )
        final_model_images = apply_rms_contrast_constraint(
            best_images,
            config.train_contrast,
        )
        best_scores = direction_sign * objective(final_model_images)
        best_scores = best_scores.reshape(batch_size, -1).mean(dim=1)
    return BatchOptimizationResult(
        images=best_images.detach(),
        best_scores=best_scores.detach(),
        final_scores=final_scores.detach(),
    )


def save_image_tensor(image_01: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image_np = image_01.detach().squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    image_uint8 = (image_np * 255.0).round().astype(np.uint8)
    Image.fromarray(image_uint8).save(path)


def save_history(history: Sequence[Dict[str, float]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not history:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_metadata(metadata: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def parse_voxel_indices(values: Iterable[int]) -> List[int]:
    return [int(v) for v in values]
