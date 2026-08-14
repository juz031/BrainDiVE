#!/usr/bin/env python3
"""Validate BrainDiVE and natural val images with a frozen encoding model.

The default validator is OpenCLIP ConvNeXt-Base pretrained on LAION-400M. Its
voxel encoding model is loaded from set 2, while the existing generation and
ranking runs under ``sub-01`` use set 1. Validation never changes image order
or overwrites ``ranking.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.models.feature_extraction import create_feature_extractor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prf_utils import get_prf_stack


DEFAULT_INPUT = Path(
    "/user_data/junruz/BrainDiVE/contrast_con/"
    "pRF_zscore_model_ranked/sub-01"
)
DEFAULT_MODEL_ROOT = Path("/user_data/junruz/prf_models/fixed/split_1_zscore")
DEFAULT_NSD_ROOT = Path("/lab_data/hendersonlab/datasets/nsd_preproc")
DEFAULT_OUTPUT_ROOT = Path("/user_data/junruz/BrainDiVE/validation")

ROI_SOURCE_LAYERS = {
    "V1": "layer2",
    "hV4": "layer3",
}

DEFAULT_VALIDATION_MODEL = "OPEN_CLIP_CONVNEXT_BASE"
DEFAULT_ADV_CHECKPOINT = Path(
    "/user_data/junruz/prf_features/imagenet_l2_3_0.pt"
)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RN50_LAYER_SIZES = {
    "relu": (64, 112, 112),
    "maxpool": (64, 56, 56),
    "layer1": (256, 56, 56),
    "layer2": (512, 28, 28),
    "layer3": (1024, 14, 14),
    "layer4": (2048, 7, 7),
}
CLIP_RN50_LAYER_SIZES = {
    "relu1": (32, 112, 112),
    "relu2": (32, 112, 112),
    "relu3": (64, 112, 112),
    "avgpool": (64, 56, 56),
    "layer1": (256, 56, 56),
    "layer2": (512, 28, 28),
    "layer3": (1024, 14, 14),
    "layer4": (2048, 7, 7),
}
OPEN_CLIP_RN50_LAYER_SIZES = {
    "act1": (32, 112, 112),
    "act2": (32, 112, 112),
    "act3": (64, 112, 112),
    "avgpool": (64, 56, 56),
    "layer1": (256, 56, 56),
    "layer2": (512, 28, 28),
    "layer3": (1024, 14, 14),
    "layer4": (2048, 7, 7),
}
CONVNEXT_LAYER_SIZES = {
    "stem": (128, 56, 56),
    "stage1": (128, 56, 56),
    "stage2": (256, 28, 28),
    "stage3": (512, 14, 14),
    "stage4": (1024, 7, 7),
}

RN50_RETURN_NODES = {
    "relu": "relu",
    "maxpool": "maxpool",
    "layer1": "layer1",
    "layer2": "layer2",
    "layer3": "layer3",
    "layer4": "layer4",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backbone: str
    pretraining_set: str
    pretrained_tag: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    layer_sizes: dict[str, tuple[int, int, int]]
    return_nodes: dict[str, str]
    default_layer_mapping: dict[str, str]


def identity_layer_mapping(layers: dict[str, tuple[int, int, int]]) -> dict[str, str]:
    return {layer: layer for layer in layers}


CONVNEXT_LAYER_MAPPING = {
    "relu1": "stage1",
    "relu2": "stage1",
    "relu3": "stage1",
    "act1": "stage1",
    "act2": "stage1",
    "act3": "stage1",
    "avgpool": "stage1",
    "relu": "stage1",
    "maxpool": "stage1",
    "layer1": "stage1",
    "layer2": "stage2",
    "layer3": "stage3",
    "layer4": "stage4",
}

MODEL_SPECS = {
    "CLIP_RN50": ModelSpec(
        name="CLIP_RN50",
        backbone="resnet50",
        pretraining_set="WebImageText-400M",
        pretrained_tag="openai",
        mean=CLIP_MEAN,
        std=CLIP_STD,
        layer_sizes=CLIP_RN50_LAYER_SIZES,
        return_nodes=identity_layer_mapping(CLIP_RN50_LAYER_SIZES),
        default_layer_mapping={
            **identity_layer_mapping(CLIP_RN50_LAYER_SIZES),
            "act1": "relu1",
            "act2": "relu2",
            "act3": "relu3",
            "relu": "relu3",
            "maxpool": "avgpool",
        },
    ),
    "OPEN_CLIP_RN50": ModelSpec(
        name="OPEN_CLIP_RN50",
        backbone="resnet50",
        pretraining_set="YFCC-15M",
        pretrained_tag="yfcc15m",
        mean=CLIP_MEAN,
        std=CLIP_STD,
        layer_sizes=OPEN_CLIP_RN50_LAYER_SIZES,
        return_nodes=identity_layer_mapping(OPEN_CLIP_RN50_LAYER_SIZES),
        default_layer_mapping={
            **identity_layer_mapping(OPEN_CLIP_RN50_LAYER_SIZES),
            "relu1": "act1",
            "relu2": "act2",
            "relu3": "act3",
            "relu": "act3",
            "maxpool": "avgpool",
        },
    ),
    "DINO_RN50": ModelSpec(
        name="DINO_RN50",
        backbone="resnet50",
        pretraining_set="ImageNet-1K",
        pretrained_tag="dino_resnet50",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        layer_sizes=RN50_LAYER_SIZES,
        return_nodes=RN50_RETURN_NODES,
        default_layer_mapping={
            **identity_layer_mapping(RN50_LAYER_SIZES),
            "relu1": "relu",
            "relu2": "relu",
            "relu3": "relu",
            "act1": "relu",
            "act2": "relu",
            "act3": "relu",
            "avgpool": "maxpool",
        },
    ),
    "SIMCLR_RN50": ModelSpec(
        name="SIMCLR_RN50",
        backbone="resnet50",
        pretraining_set="ImageNet-1K",
        pretrained_tag="simclrv1-imagenet1k-resnet50-1x",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        layer_sizes=RN50_LAYER_SIZES,
        return_nodes=RN50_RETURN_NODES,
        default_layer_mapping={},
    ),
    "ADV_RN50": ModelSpec(
        name="ADV_RN50",
        backbone="resnet50",
        pretraining_set="ImageNet-1K",
        pretrained_tag="imagenet_l2_3_0",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        layer_sizes=RN50_LAYER_SIZES,
        return_nodes=RN50_RETURN_NODES,
        default_layer_mapping={},
    ),
    "OPEN_CLIP_CONVNEXT_BASE": ModelSpec(
        name="OPEN_CLIP_CONVNEXT_BASE",
        backbone="convnext_base",
        pretraining_set="LAION-400M",
        pretrained_tag="laion400m_s13b_b51k",
        mean=CLIP_MEAN,
        std=CLIP_STD,
        layer_sizes=CONVNEXT_LAYER_SIZES,
        return_nodes={
            "trunk.stem.1.permute_1": "stem",
            "trunk.stages.0.blocks.2.add": "stage1",
            "trunk.stages.1.blocks.2.add": "stage2",
            "trunk.stages.2.blocks.26.add": "stage3",
            "trunk.stages.3.blocks.2.add": "stage4",
        },
        default_layer_mapping=CONVNEXT_LAYER_MAPPING,
    ),
}

for model_name in ("SIMCLR_RN50", "ADV_RN50"):
    spec = MODEL_SPECS[model_name]
    MODEL_SPECS[model_name] = ModelSpec(
        **{
            **spec.__dict__,
            "default_layer_mapping": MODEL_SPECS["DINO_RN50"].default_layer_mapping,
        }
    )


@dataclass
class ValidationResources:
    model: ModelSpec
    layer: str
    encoder: torch.nn.Module
    best_prf_idx: np.ndarray
    best_weights: dict[str, np.ndarray]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    r2: dict[str, Any]
    prf_stack: np.ndarray


def safe_component(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    return cleaned.strip("-.") or "unnamed"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def model_description(name: str) -> dict[str, str]:
    spec = MODEL_SPECS.get(name)
    if spec is None:
        raise ValueError(
            f"No independence metadata is registered for model {name!r}"
        )
    return {
        "name": spec.name,
        "backbone": spec.backbone,
        "pretraining_set": spec.pretraining_set,
        "pretrained_tag": spec.pretrained_tag,
    }


def check_independence(
    run_config: dict[str, Any],
    validation_model: str,
    validation_set: int,
) -> dict[str, Any]:
    generation = model_description(run_config["generation_model"]["name"])
    ranking = model_description(run_config["ranking_model"]["name"])
    validation = model_description(validation_model)
    generation_set = int(run_config["split"])

    checks = {
        "different_model_from_generation": validation["name"] != generation["name"],
        "different_model_from_ranking": validation["name"] != ranking["name"],
        "different_backbone_from_generation": (
            validation["backbone"] != generation["backbone"]
        ),
        "different_backbone_from_ranking": (
            validation["backbone"] != ranking["backbone"]
        ),
        "different_pretraining_set_from_generation": (
            validation["pretraining_set"] != generation["pretraining_set"]
        ),
        "different_pretraining_set_from_ranking": (
            validation["pretraining_set"] != ranking["pretraining_set"]
        ),
        "different_encoding_fit_set": validation_set != generation_set,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "generation": generation,
        "ranking": ranking,
        "validation": validation,
        "generation_encoding_fit_set": generation_set,
        "validation_encoding_fit_set": validation_set,
    }


def infer_validation_layer(
    run_config: dict[str, Any],
    model: ModelSpec,
    explicit_layer: str | None,
    roi: str,
) -> str:
    source_layer = ROI_SOURCE_LAYERS[roi]
    run_layer = run_config["generation_model"]["layer"]
    if run_layer != source_layer:
        raise ValueError(
            f"ROI {roi} requires generation layer {source_layer}, but "
            f"run_config.json uses {run_layer}"
        )
    if explicit_layer is not None:
        if explicit_layer not in model.layer_sizes:
            raise ValueError(
                f"{model.name} has no layer {explicit_layer!r}; "
                f"choose from {sorted(model.layer_sizes)}"
            )
        return explicit_layer
    try:
        return model.default_layer_mapping[source_layer]
    except KeyError as error:
        raise ValueError(
            f"No default {model.name} validation layer maps from "
            f"{source_layer!r}; "
            "pass --validation-layer explicitly"
        ) from error


def build_validation_encoder(
    model: ModelSpec,
    device: torch.device,
    adv_checkpoint: Path,
) -> torch.nn.Module:
    if model.name == "CLIP_RN50":
        import clip

        loaded_model, _ = clip.load("RN50", device=device)
        backbone = loaded_model.visual
        del loaded_model.transformer
    elif model.name == "OPEN_CLIP_RN50":
        import open_clip

        loaded_model, _, _ = open_clip.create_model_and_transforms(
            "RN50",
            pretrained=model.pretrained_tag,
        )
        backbone = loaded_model.visual
        del loaded_model.transformer
    elif model.name == "OPEN_CLIP_CONVNEXT_BASE":
        import open_clip

        loaded_model, _, _ = open_clip.create_model_and_transforms(
            model.backbone,
            pretrained=model.pretrained_tag,
        )
        backbone = loaded_model.visual
    elif model.name == "DINO_RN50":
        backbone = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_resnet50",
        )
    elif model.name == "SIMCLR_RN50":
        from huggingface_hub import hf_hub_download
        from torchvision.models import resnet50

        checkpoint_path = hf_hub_download(
            repo_id="lightly-ai/simclrv1-imagenet1k-resnet50-1x",
            filename="resnet50-1x.pth",
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        backbone = resnet50(weights=None)
        backbone.load_state_dict(state_dict, strict=True)
    elif model.name == "ADV_RN50":
        from robustness.datasets import ImageNet
        from robustness.model_utils import make_and_restore_model
        from torchvision.models import resnet50

        wrapped_model, _ = make_and_restore_model(
            arch="resnet50",
            dataset=ImageNet("/tmp"),
            resume_path=str(adv_checkpoint),
            parallel=False,
        )
        state_dict = {
            key: value.detach().cpu()
            for key, value in wrapped_model.model.state_dict().items()
        }
        backbone = resnet50(weights=None)
        backbone.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError(f"No encoder builder is registered for {model.name}")

    backbone.eval()
    encoder = create_feature_extractor(
        backbone,
        return_nodes=model.return_nodes,
    )
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder.eval().to(device)


def load_validation_resources(
    *,
    layer: str,
    model: ModelSpec,
    subject: int,
    validation_set: int,
    model_root: Path,
    encoder: torch.nn.Module,
) -> ValidationResources:
    if layer not in model.layer_sizes:
        raise ValueError(
            f"Unknown {model.name} validation layer {layer!r}; "
            f"choose from {sorted(model.layer_sizes)}"
        )
    model_dir = (
        model_root
        / f"S{subject}"
        / f"{model.name}_set{validation_set}"
        / layer
    )
    required = [
        "best_prf_idx.npy",
        "best_weights.pkl",
        "best_features_m.npy",
        "best_features_s.npy",
        "r2.pkl",
    ]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Validation model is incomplete in {model_dir}; missing {missing}"
        )

    with (model_dir / "best_weights.pkl").open("rb") as handle:
        best_weights = pickle.load(handle)
    with (model_dir / "r2.pkl").open("rb") as handle:
        r2 = pickle.load(handle)

    _, height, width = model.layer_sizes[layer]
    if height != width:
        raise ValueError(f"Expected square feature maps for {layer}")

    return ValidationResources(
        model=model,
        layer=layer,
        encoder=encoder,
        best_prf_idx=np.load(model_dir / "best_prf_idx.npy"),
        best_weights=best_weights,
        feature_mean=np.load(model_dir / "best_features_m.npy"),
        feature_std=np.load(model_dir / "best_features_s.npy"),
        r2=r2,
        prf_stack=get_prf_stack(
            n_pix=height,
            which_prf_grid="default-log-polar",
        ),
    )


def resolve_image_path(result: dict[str, Any], ranking_path: Path) -> Path:
    stored = result.get("image_path")
    if stored and Path(stored).is_file():
        return Path(stored)
    relative = result.get("image_file")
    if relative:
        candidate = ranking_path.parent / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot resolve image for rank {result.get('rank')} in {ranking_path}"
    )


def validate_ranking_voxel_identity(
    ranking: dict[str, Any],
    ranking_path: Path,
    resources: ValidationResources,
) -> int:
    """Ensure the MEI directory, metadata, and validation readout agree."""
    voxel_id = int(ranking["voxel_id"])
    voxel_rank = int(ranking["voxel_rank_in_topk"])
    match = re.fullmatch(
        r"voxel-rank-(\d+)__id-(\d+)",
        ranking_path.parent.name,
    )
    if match is None:
        raise ValueError(
            f"Cannot verify voxel identity from MEI directory "
            f"{ranking_path.parent.name!r}"
        )
    directory_rank = int(match.group(1))
    directory_voxel_id = int(match.group(2))
    if directory_voxel_id != voxel_id:
        raise ValueError(
            f"MEI voxel mismatch in {ranking_path}: directory has voxel "
            f"{directory_voxel_id}, ranking.json has voxel {voxel_id}"
        )
    if directory_rank != voxel_rank:
        raise ValueError(
            f"MEI voxel-rank mismatch in {ranking_path}: directory has rank "
            f"{directory_rank}, ranking.json has rank {voxel_rank}"
        )
    if str(voxel_id) not in resources.best_weights:
        raise KeyError(
            f"Validation model has no readout for MEI voxel {voxel_id}"
        )
    if not 0 <= voxel_id < len(resources.best_prf_idx):
        raise IndexError(
            f"MEI voxel {voxel_id} is outside best_prf_idx with length "
            f"{len(resources.best_prf_idx)}"
        )
    if not 0 <= voxel_id < len(resources.feature_mean):
        raise IndexError(
            f"MEI voxel {voxel_id} is outside feature means with length "
            f"{len(resources.feature_mean)}"
        )
    if not 0 <= voxel_id < len(resources.feature_std):
        raise IndexError(
            f"MEI voxel {voxel_id} is outside feature stds with length "
            f"{len(resources.feature_std)}"
        )
    return voxel_id


def load_image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
    return torch.from_numpy(array).permute(2, 0, 1) / 255.0


def preprocess(batch: torch.Tensor, model: ModelSpec) -> torch.Tensor:
    mean = torch.as_tensor(
        model.mean,
        device=batch.device,
        dtype=batch.dtype,
    ).view(1, 3, 1, 1)
    std = torch.as_tensor(
        model.std,
        device=batch.device,
        dtype=batch.dtype,
    ).view(1, 3, 1, 1)
    return (batch - mean) / std


def score_images(
    *,
    image_paths: list[Path],
    voxel_id: int,
    resources: ValidationResources,
    device: torch.device,
    batch_size: int,
) -> tuple[list[float], int]:
    weights_with_intercept = np.asarray(
        resources.best_weights[str(voxel_id)]
    ).reshape(-1)
    weights = torch.as_tensor(
        weights_with_intercept[:-1],
        device=device,
        dtype=torch.float64,
    )
    intercept = torch.as_tensor(
        weights_with_intercept[-1],
        device=device,
        dtype=torch.float64,
    )
    feature_mean = torch.as_tensor(
        resources.feature_mean[voxel_id],
        device=device,
        dtype=torch.float64,
    )
    feature_std = torch.as_tensor(
        resources.feature_std[voxel_id],
        device=device,
        dtype=torch.float64,
    )
    prf_idx = int(resources.best_prf_idx[voxel_id])
    prf_kernel = torch.as_tensor(
        resources.prf_stack[:, :, prf_idx],
        device=device,
        dtype=torch.float64,
    )

    expected_channels = resources.model.layer_sizes[resources.layer][0]
    if not (
        len(weights) == len(feature_mean) == len(feature_std) == expected_channels
    ):
        raise ValueError(
            f"Channel mismatch for voxel {voxel_id}, layer {resources.layer}: "
            f"weights={len(weights)}, mean={len(feature_mean)}, "
            f"std={len(feature_std)}, expected={expected_channels}"
        )

    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch = torch.stack(
                [load_image_tensor(path) for path in batch_paths]
            ).to(device)
            features = resources.encoder(
                preprocess(batch, resources.model)
            )[resources.layer]
            pooled = torch.einsum(
                "bchw,hw->bc",
                features.to(torch.float64),
                prf_kernel,
            )
            standardized = (pooled - feature_mean) / feature_std
            predictions = standardized @ weights + intercept
            scores.extend(float(value) for value in predictions.cpu().numpy())
    return scores, prf_idx


def load_validation_image_ids(
    *,
    model_root: Path,
    subject: int,
    validation_set: int,
) -> np.ndarray:
    split_path = model_root / f"S{subject}" / f"data_splits_S{subject}.pkl"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    with split_path.open("rb") as handle:
        split_payload = pickle.load(handle)
    try:
        validation_ids = np.asarray(
            split_payload[validation_set]["val"],
            dtype=np.int64,
        ).reshape(-1)
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Cannot read val IDs for set {validation_set} from {split_path}"
        ) from error
    if len(validation_ids) == 0:
        raise ValueError(
            f"Validation set {validation_set} in {split_path} is empty"
        )
    return validation_ids


def natural_image_batch(stimuli: Any, image_ids: np.ndarray) -> torch.Tensor:
    images = []
    for image_id in image_ids:
        image = np.asarray(stimuli[int(image_id)], dtype=np.float32)
        if image.ndim != 3:
            raise ValueError(
                f"Natural image {int(image_id)} has shape {image.shape}, expected 3D"
            )
        if image.shape[0] == 3:
            pass
        elif image.shape[-1] == 3:
            image = np.moveaxis(image, -1, 0)
        else:
            raise ValueError(
                f"Natural image {int(image_id)} has shape {image.shape}, "
                "expected CHW or HWC RGB"
            )
        images.append(torch.from_numpy(image) / 255.0)
    batch = torch.stack(images)
    if tuple(batch.shape[-2:]) != (224, 224):
        batch = torch.nn.functional.interpolate(
            batch,
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    return batch


def save_natural_validation_predictions(
    *,
    source_run_dir: Path,
    output_run_dir: Path,
    records: list[dict[str, Any]],
    resources: ValidationResources,
    validation_set: int,
    model_root: Path,
    nsd_root: Path,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> Path:
    """Predict selected voxels for every natural image in the val partition."""
    if not records:
        raise ValueError(f"No voxel records were found for {source_run_dir}")

    subject = int(records[0]["subject"])
    if any(int(record["subject"]) != subject for record in records):
        raise ValueError(f"Run contains multiple subjects: {source_run_dir}")

    validation_ids = load_validation_image_ids(
        model_root=model_root,
        subject=subject,
        validation_set=validation_set,
    )
    voxel_ids = np.asarray(
        [int(record["voxel_id"]) for record in records],
        dtype=np.int64,
    )

    weights_with_intercept = [
        np.asarray(resources.best_weights[str(voxel_id)]).reshape(-1)
        for voxel_id in voxel_ids
    ]
    expected_channels = resources.model.layer_sizes[resources.layer][0]
    if any(
        len(values) != expected_channels + 1
        for values in weights_with_intercept
    ):
        raise ValueError(
            f"Natural-validation readout channel mismatch for {resources.model.name}/"
            f"{resources.layer}; expected {expected_channels} weights plus intercept"
        )

    weights = torch.as_tensor(
        np.stack([values[:-1] for values in weights_with_intercept]),
        device=device,
        dtype=torch.float64,
    )
    intercepts = torch.as_tensor(
        np.asarray([values[-1] for values in weights_with_intercept]),
        device=device,
        dtype=torch.float64,
    )
    feature_mean = torch.as_tensor(
        resources.feature_mean[voxel_ids],
        device=device,
        dtype=torch.float64,
    )
    feature_std = torch.as_tensor(
        resources.feature_std[voxel_ids],
        device=device,
        dtype=torch.float64,
    )
    prf_indices = np.asarray(
        resources.best_prf_idx[voxel_ids],
        dtype=np.int64,
    )
    prf_kernels = torch.as_tensor(
        np.moveaxis(resources.prf_stack[:, :, prf_indices], -1, 0),
        device=device,
        dtype=torch.float64,
    )

    expected_shape = (len(voxel_ids), expected_channels)
    if tuple(weights.shape) != expected_shape:
        raise ValueError(
            f"Weights have shape {tuple(weights.shape)}, expected {expected_shape}"
        )
    if tuple(feature_mean.shape) != expected_shape:
        raise ValueError(
            f"Feature means have shape {tuple(feature_mean.shape)}, expected {expected_shape}"
        )
    if tuple(feature_std.shape) != expected_shape:
        raise ValueError(
            f"Feature stds have shape {tuple(feature_std.shape)}, expected {expected_shape}"
        )

    stimulus_path = nsd_root / "stimuli" / f"S{subject}_stimuli_224.h5py"
    if not stimulus_path.is_file():
        raise FileNotFoundError(stimulus_path)

    import h5py

    prediction_chunks = []
    with h5py.File(stimulus_path, "r") as handle, torch.inference_mode():
        if "stimuli" not in handle:
            raise KeyError(f"{stimulus_path} has no 'stimuli' dataset")
        stimuli = handle["stimuli"]
        if (
            int(np.max(validation_ids)) >= len(stimuli)
            or int(np.min(validation_ids)) < 0
        ):
            raise IndexError(
                f"Validation IDs are outside the stimulus range [0, {len(stimuli)})"
            )
        for start in range(0, len(validation_ids), batch_size):
            batch_ids = validation_ids[start : start + batch_size]
            batch = natural_image_batch(stimuli, batch_ids).to(device)
            features = resources.encoder(
                preprocess(batch, resources.model)
            )[resources.layer].to(torch.float64)
            pooled = torch.einsum("bchw,vhw->bvc", features, prf_kernels)
            standardized = (
                pooled - feature_mean.unsqueeze(0)
            ) / feature_std.unsqueeze(0)
            predictions = torch.einsum("bvc,vc->bv", standardized, weights)
            predictions = predictions + intercepts.unsqueeze(0)
            prediction_chunks.append(predictions.cpu().numpy().astype(np.float32))

    prediction_matrix = np.concatenate(prediction_chunks, axis=0).T
    expected_prediction_shape = (len(voxel_ids), len(validation_ids))
    if prediction_matrix.shape != expected_prediction_shape:
        raise RuntimeError(
            f"Prediction matrix has shape {prediction_matrix.shape}, "
            f"expected {expected_prediction_shape}"
        )

    model_tag = safe_component(
        f"{resources.model.name}-{resources.layer}-set{validation_set}"
    )
    output_path = (
        output_run_dir / f"{model_tag}__natural-val-predictions.npz"
    )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(1, dtype=np.int64),
        subject=np.asarray(subject, dtype=np.int64),
        validation_set=np.asarray(validation_set, dtype=np.int64),
        validation_model=np.asarray(resources.model.name),
        validation_layer=np.asarray(resources.layer),
        stimulus_file=np.asarray(str(stimulus_path.resolve())),
        val_ids=validation_ids,
        voxel_ids=voxel_ids,
        source_ranking_files=np.asarray(
            [str(record["source_ranking_file"]) for record in records]
        ),
        regions=np.asarray([str(record["region"]) for record in records]),
        voxel_indices_in_region=np.asarray(
            [int(record["voxel_index_in_region"]) for record in records],
            dtype=np.int64,
        ),
        voxel_ranks_in_topk=np.asarray(
            [int(record["voxel_rank_in_topk"]) for record in records],
            dtype=np.int64,
        ),
        prf_indices=prf_indices,
        predictions=prediction_matrix,
    )
    return output_path


def rank_descending(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    ranks = [0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if np.std(x_array) == 0 or np.std(y_array) == 0:
        return None
    return float(np.corrcoef(x_array, y_array)[0, 1])


def spearman_from_ranks(x_ranks: list[int], y_ranks: list[int]) -> float | None:
    return pearson([float(value) for value in x_ranks], [float(value) for value in y_ranks])


def finite_or_none(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def validation_r2(
    resources: ValidationResources,
    region: str,
    voxel_index_in_region: int,
) -> float | None:
    try:
        values = np.asarray(resources.r2[region]["r2_voxels"]).reshape(-1)
        return finite_or_none(float(values[voxel_index_in_region]))
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def validate_ranking_file(
    *,
    ranking_path: Path,
    source_run_dir: Path,
    output_run_dir: Path,
    run_config: dict[str, Any],
    resources: ValidationResources,
    validation_set: int,
    independence: dict[str, Any],
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    ranking = load_json(ranking_path)
    voxel_id = validate_ranking_voxel_identity(
        ranking,
        ranking_path,
        resources,
    )
    results = sorted(ranking["results"], key=lambda item: int(item["rank"]))
    image_paths = [
        resolve_image_path(result, ranking_path) for result in results
    ]
    scores, prf_idx = score_images(
        image_paths=image_paths,
        voxel_id=voxel_id,
        resources=resources,
        device=device,
        batch_size=batch_size,
    )
    validation_ranks = rank_descending(scores)
    original_ranks = [int(result["rank"]) for result in results]
    rank_scores = [float(result["rank_score"]) for result in results]
    gen_scores = [float(result["gen_score"]) for result in results]

    validated_results = []
    for result, image_path, score, new_rank in zip(
        results,
        image_paths,
        scores,
        validation_ranks,
    ):
        validated_results.append(
            {
                "seed": int(result["seed"]),
                "original_rank": int(result["rank"]),
                "validation_rank": int(new_rank),
                "gen_score": float(result["gen_score"]),
                "rank_score": float(result["rank_score"]),
                "validation_score": float(score),
                "image_file": result.get("image_file"),
                "image_path": str(image_path.resolve()),
            }
        )

    best_validation_index = int(np.argmax(np.asarray(scores)))
    agreement = {
        "spearman_original_vs_validation_rank": finite_or_none(
            spearman_from_ranks(original_ranks, validation_ranks)
        ),
        "pearson_rank_score_vs_validation_score": finite_or_none(
            pearson(rank_scores, scores)
        ),
        "pearson_gen_score_vs_validation_score": finite_or_none(
            pearson(gen_scores, scores)
        ),
        "original_top1_is_validation_top1": validation_ranks[0] == 1,
        "validation_top1_original_rank": original_ranks[best_validation_index],
        "validation_top1_seed": int(results[best_validation_index]["seed"]),
    }

    model_tag = (
        f"{resources.model.name}-{resources.layer}-set{validation_set}"
    )
    output_path = (
        output_run_dir
        / ranking_path.parent.relative_to(source_run_dir)
        / f"{safe_component(model_tag)}.json"
    )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )

    payload = {
        "schema_version": 1,
        "source_ranking_file": str(ranking_path.resolve()),
        "source_run_config": run_config,
        "independence": independence,
        "validation_model": {
            "name": resources.model.name,
            "backbone": resources.model.backbone,
            "pretraining_set": resources.model.pretraining_set,
            "pretrained_tag": resources.model.pretrained_tag,
            "layer": resources.layer,
            "encoding_fit_set": validation_set,
        },
        "region": ranking["region"],
        "subject": int(ranking["subject"]),
        "voxel_id": voxel_id,
        "voxel_index_in_region": int(ranking["voxel_index_in_region"]),
        "voxel_rank_in_topk": int(ranking["voxel_rank_in_topk"]),
        "generation_model_r2": float(ranking["voxel_r2"]),
        "validation_model_r2": validation_r2(
            resources,
            ranking["region"],
            int(ranking["voxel_index_in_region"]),
        ),
        "validation_prf_index": prf_idx,
        "num_images": len(results),
        "agreement": agreement,
        "results": validated_results,
    }
    write_json(output_path, payload)
    return output_path, payload


def discover_runs(
    input_root: Path,
    model_name: str,
    roi: str,
) -> list[Path]:
    runs = []
    required_layer = ROI_SOURCE_LAYERS[roi]
    for config_path in input_root.rglob("run_config.json"):
        config = load_json(config_path)
        if config.get("generation_model", {}).get("name") != model_name:
            continue
        if roi not in config.get("regions", []):
            continue
        if config["generation_model"].get("layer") != required_layer:
            continue
        runs.append(config_path.parent)
    return sorted(runs)


def resolve_output_run_dir(
    output_root: Path,
    source_run_dir: Path,
    subject: int,
) -> Path:
    subject_component = f"sub-{subject:02d}"
    for candidate in (source_run_dir, *source_run_dir.parents):
        if candidate.name == subject_component:
            return output_root / subject_component / source_run_dir.relative_to(candidate)
    return output_root / subject_component / source_run_dir.name


def write_run_summary(
    *,
    source_run_dir: Path,
    output_run_dir: Path,
    model: ModelSpec,
    layer: str,
    validation_set: int,
    independence: dict[str, Any],
    records: list[dict[str, Any]],
    natural_predictions_path: Path,
) -> tuple[Path, Path]:
    model_tag = safe_component(
        f"{model.name}-{layer}-set{validation_set}"
    )
    output_run_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_run_dir / f"{model_tag}__summary.json"
    csv_path = output_run_dir / f"{model_tag}__summary.csv"

    top1_values = [
        bool(record["agreement"]["original_top1_is_validation_top1"])
        for record in records
    ]
    spearman_values = [
        record["agreement"]["spearman_original_vs_validation_rank"]
        for record in records
        if record["agreement"]["spearman_original_vs_validation_rank"] is not None
    ]
    summary = {
        "schema_version": 2,
        "run_dir": str(source_run_dir.resolve()),
        "validation_model": {
            "name": model.name,
            "backbone": model.backbone,
            "pretraining_set": model.pretraining_set,
            "pretrained_tag": model.pretrained_tag,
            "layer": layer,
            "encoding_fit_set": validation_set,
        },
        "independence": independence,
        "natural_validation_predictions_file": str(
            natural_predictions_path.resolve()
        ),
        "num_voxels": len(records),
        "mean_spearman_original_vs_validation_rank": (
            float(np.mean(spearman_values)) if spearman_values else None
        ),
        "top1_agreement_fraction": (
            float(np.mean(top1_values)) if top1_values else None
        ),
        "voxels": [
            {
                "region": record["region"],
                "voxel_id": record["voxel_id"],
                "voxel_rank_in_topk": record["voxel_rank_in_topk"],
                "generation_model_r2": record["generation_model_r2"],
                "validation_model_r2": record["validation_model_r2"],
                **record["agreement"],
            }
            for record in records
        ],
    }
    write_json(json_path, summary)

    fieldnames = [
        "region",
        "voxel_id",
        "voxel_rank_in_topk",
        "generation_model_r2",
        "validation_model_r2",
        "spearman_original_vs_validation_rank",
        "pearson_rank_score_vs_validation_score",
        "pearson_gen_score_vs_validation_score",
        "original_top1_is_validation_top1",
        "validation_top1_original_rank",
        "validation_top1_seed",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in summary["voxels"]:
            writer.writerow({field: record.get(field) for field in fieldnames})
    return json_path, csv_path


def check_run_resources(
    *,
    run_dir: Path,
    run_config: dict[str, Any],
    model: ModelSpec,
    layer: str,
    validation_set: int,
    model_root: Path,
    nsd_root: Path,
    roi: str,
) -> list[str]:
    subject = int(run_config["subject"])
    model_dir = (
        model_root
        / f"S{subject}"
        / f"{model.name}_set{validation_set}"
        / layer
    )
    messages = [f"run={run_dir}", f"validation_model={model_dir}"]
    split_path = model_root / f"S{subject}" / f"data_splits_S{subject}.pkl"
    stimulus_path = nsd_root / "stimuli" / f"S{subject}_stimuli_224.h5py"
    messages.extend(
        [
            f"validation_split={split_path}",
            f"natural_stimuli={stimulus_path}",
        ]
    )
    ranking_files = sorted(
        run_dir.glob(f"roi-{safe_component(roi)}/voxel-rank-*/ranking.json")
    )
    messages.append(f"ranking_files={len(ranking_files)}")
    if not ranking_files:
        raise FileNotFoundError(f"No ranking.json files found under {run_dir}")
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    if not stimulus_path.is_file():
        raise FileNotFoundError(stimulus_path)
    for name in (
        "best_prf_idx.npy",
        "best_weights.pkl",
        "best_features_m.npy",
        "best_features_s.npy",
        "r2.pkl",
    ):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(model_dir / name)
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--model-name",
        required=True,
        choices=sorted(MODEL_SPECS),
        help="Generation model whose runs should be validated",
    )
    parser.add_argument(
        "--roi",
        required=True,
        choices=sorted(ROI_SOURCE_LAYERS),
        help="ROI to validate (V1 uses layer2; hV4 uses layer3)",
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for all validation outputs",
    )
    parser.add_argument(
        "--nsd-root",
        type=Path,
        default=DEFAULT_NSD_ROOT,
        help="NSD preprocessing root containing stimuli/S*_stimuli_224.h5py",
    )
    parser.add_argument(
        "--validation-model",
        choices=sorted(MODEL_SPECS),
        default=DEFAULT_VALIDATION_MODEL,
        help="Frozen backbone/encoding-model family used for validation",
    )
    parser.add_argument("--validation-set", type=int, default=2)
    parser.add_argument(
        "--validation-layer",
        default=None,
        help=(
            "Backbone layer to validate. By default, an analogous layer is "
            "selected from the generation layer."
        ),
    )
    parser.add_argument(
        "--adv-checkpoint",
        type=Path,
        default=DEFAULT_ADV_CHECKPOINT,
        help="Checkpoint used only when --validation-model ADV_RN50",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, for example cuda, cuda:0, or cpu",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check run metadata and required files without loading the network",
    )
    parser.add_argument(
        "--allow-nonindependent",
        action="store_true",
        help="Proceed even if an explicit independence check fails",
    )
    args = parser.parse_args()

    runs = discover_runs(args.input, args.model_name, args.roi)
    if not runs:
        parser.error(
            f"No runs found for generation model {args.model_name} and "
            f"ROI {args.roi} under {args.input}"
        )

    validation_model = MODEL_SPECS[args.validation_model]
    device = torch.device(args.device)
    resource_cache: dict[tuple[int, str], ValidationResources] = {}
    validation_encoder: torch.nn.Module | None = None
    total_outputs = 0

    for run_dir in runs:
        run_config = load_json(run_dir / "run_config.json")
        layer = infer_validation_layer(
            run_config,
            validation_model,
            args.validation_layer,
            args.roi,
        )
        independence = check_independence(
            run_config,
            validation_model.name,
            args.validation_set,
        )
        if not independence["passed"] and not args.allow_nonindependent:
            failed = [
                name
                for name, passed in independence["checks"].items()
                if not passed
            ]
            raise ValueError(
                f"Independent-validation checks failed for {run_dir}: {failed}"
            )

        for message in check_run_resources(
            run_dir=run_dir,
            run_config=run_config,
            model=validation_model,
            layer=layer,
            validation_set=args.validation_set,
            model_root=args.model_root,
            nsd_root=args.nsd_root,
            roi=args.roi,
        ):
            print(message)
        if args.dry_run:
            continue

        subject = int(run_config["subject"])
        output_run_dir = resolve_output_run_dir(
            args.output_root,
            run_dir,
            subject,
        )
        cache_key = (subject, layer)
        if cache_key not in resource_cache:
            if validation_encoder is None:
                validation_encoder = build_validation_encoder(
                    validation_model,
                    device,
                    args.adv_checkpoint,
                )
            resource_cache[cache_key] = load_validation_resources(
                layer=layer,
                model=validation_model,
                subject=subject,
                validation_set=args.validation_set,
                model_root=args.model_root,
                encoder=validation_encoder,
            )
        resources = resource_cache[cache_key]

        run_records = []
        ranking_files = sorted(
            run_dir.glob(
                f"roi-{safe_component(args.roi)}/voxel-rank-*/ranking.json"
            )
        )
        for ranking_path in ranking_files:
            output_path, payload = validate_ranking_file(
                ranking_path=ranking_path,
                source_run_dir=run_dir,
                output_run_dir=output_run_dir,
                run_config=run_config,
                resources=resources,
                validation_set=args.validation_set,
                independence=independence,
                device=device,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
            )
            run_records.append(payload)
            total_outputs += 1
            print(output_path)

        natural_predictions_path = save_natural_validation_predictions(
            source_run_dir=run_dir,
            output_run_dir=output_run_dir,
            records=run_records,
            resources=resources,
            validation_set=args.validation_set,
            model_root=args.model_root,
            nsd_root=args.nsd_root,
            device=device,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        print(natural_predictions_path)

        summary_json, summary_csv = write_run_summary(
            source_run_dir=run_dir,
            output_run_dir=output_run_dir,
            model=validation_model,
            layer=layer,
            validation_set=args.validation_set,
            independence=independence,
            records=run_records,
            natural_predictions_path=natural_predictions_path,
        )
        print(summary_json)
        print(summary_csv)

    if args.dry_run:
        print(f"Dry run passed for {len(runs)} runs")
    else:
        print(
            f"Validated {total_outputs} voxel rankings across {len(runs)} runs"
        )


if __name__ == "__main__":
    main()
