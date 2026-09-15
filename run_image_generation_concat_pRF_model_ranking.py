"""Generate and cross-model-rank MEIs with concatenated-layer pRF models.

Each voxel uses one selected pRF ID across every layer recorded in the fitted
checkpoint. Each layer is pRF-pooled at its native spatial resolution, and the
pooled channel vectors are concatenated in ``concat_metadata.json`` order.
"""

import sys
import os
import time
import json
import h5py

import numpy as np
import pandas as pd
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, LMSDiscreteScheduler
from brain_guide_pipeline import mypipelineSAG
from generation_timing_utils import merge_roi_generation_timing
from joint_luma_contrast_utils import validate_luminance_settings
from luma_rgb_clip_utils import validate_luma_contrast_method, method_metadata
from seed_utils import (build_seed_plan, save_seed_record, save_json_atomic,
                        summarize_generation_times, GENERATION_TIMING_SCOPE,
                        collect_device_metadata)
from dual_scope_contrast_utils import (
    DualScopeContrastConvergenceError,
    global_luma_mean_and_rms,
    normalized_prf_weight,
    weighted_luma_mean_and_rms,
)
import pickle
import gc

import nibabel as nib
import os

import argparse
from torchvision.models.feature_extraction import create_feature_extractor
from prf_utils import get_prf_grid, get_prf_stack
from voxel_selection import select_voxels_by_rank
from PIL import Image
from torchvision import transforms
from skimage.transform import resize

import clip
import open_clip
from torchvision.models import resnet50
from huggingface_hub import hf_hub_download
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model


import model_fitting_utils


_PRF_STACK_CACHE = {}


MODEL_PROVENANCE = {
    "CLIP_RN50": {
        "library": "clip",
        "architecture": "RN50",
        "pretrained": "OpenAI",
    },
    "OPEN_CLIP_RN50": {
        "library": "open_clip",
        "architecture": "RN50",
        "pretrained": "yfcc15m",
        "force_quick_gelu": True,
    },
    "DINO_RN50": {
        "library": "torch.hub",
        "architecture": "dino_resnet50",
        "repository": "facebookresearch/dino:main",
    },
    "SIMCLR_RN50": {
        "library": "huggingface_hub/torchvision",
        "architecture": "resnet50",
        "repository": "lightly-ai/simclrv1-imagenet1k-resnet50-1x",
        "checkpoint": "resnet50-1x.pth",
    },
    "ADV_RN50": {
        "library": "robustness/torchvision",
        "architecture": "resnet50",
        "checkpoint": "/user_data/junruz/prf_features/imagenet_l2_3_0.pt",
    },
    "OPEN_CLIP_CONVNEXT_BASE": {
        "library": "open_clip",
        "architecture": "convnext_base",
        "pretrained": "laion400m_s13b_b51k",
    },
}



class LayerSize:
    pass

setattr(LayerSize, "CLIP_RN50_layer_size", {
    'relu': [64, 112, 112],
    'relu1': [32, 112, 112],
    'relu2': [32, 112, 112],
    'relu3': [64, 112, 112],
    'avgpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
})

setattr(LayerSize, "OPEN_CLIP_RN50_layer_size", {
    'relu': [64, 112, 112],
    'act1': [32, 112, 112],
    'act2': [32, 112, 112],
    'act3': [64, 112, 112],
    'avgpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
})

setattr(LayerSize, "DINO_RN50_layer_size", {
    'relu': [64, 112, 112],
    'maxpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
})

setattr(LayerSize, "SIMCLR_RN50_layer_size", {
    'relu': [64, 112, 112],
    'maxpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
})

setattr(LayerSize, "ADV_RN50_layer_size", {
    'relu': [64, 112, 112],
    'maxpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
})

setattr(LayerSize, "OPEN_CLIP_CONVNEXT_BASE_layer_size", {
    'stem': [128, 56, 56],
    'stage1': [128, 56, 56],
    'stage2': [256, 28, 28],
    'stage3': [512, 14, 14],
    'stage4': [1024, 7, 7]
})

class Normalize:
    pass

setattr(Normalize, "CLIP_RN50_MEAN", np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.single)[:, None, None])
setattr(Normalize, "CLIP_RN50_STD", np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.single)[:, None, None])

setattr(Normalize, "OPEN_CLIP_RN50_MEAN", np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.single)[:, None, None])
setattr(Normalize, "OPEN_CLIP_RN50_STD", np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.single)[:, None, None])

setattr(Normalize, "DINO_RN50_MEAN", np.array((0.485, 0.456, 0.406), dtype=np.single)[:, None, None])
setattr(Normalize, "DINO_RN50_STD", np.array((0.229, 0.224, 0.225), dtype=np.single)[:, None, None])

setattr(Normalize, "SIMCLR_RN50_MEAN", np.array((0.485, 0.456, 0.406), dtype=np.single)[:, None, None])
setattr(Normalize, "SIMCLR_RN50_STD", np.array((0.229, 0.224, 0.225), dtype=np.single)[:, None, None])

setattr(Normalize, "ADV_RN50_MEAN", np.array((0.485, 0.456, 0.406), dtype=np.single)[:, None, None])
setattr(Normalize, "ADV_RN50_STD", np.array((0.229, 0.224, 0.225), dtype=np.single)[:, None, None])

setattr(Normalize, "OPEN_CLIP_CONVNEXT_BASE_MEAN", np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.single)[:, None, None])
setattr(Normalize, "OPEN_CLIP_CONVNEXT_BASE_STD", np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.single)[:, None, None])


def normalize_image(input_ndarray, args):
    # print(input_ndarray.dtype, np.max(input_ndarray), np.min(input_ndarray), input_ndarray.shape)
    # exit()
    input_ndarray = input_ndarray.transpose((1, 2, 0))
    image_resized = resize(input_ndarray, (224, 224), preserve_range=True)
    scaled_image = image_resized.astype(np.single).transpose((2, 0, 1))/(255.0) #*random.uniform(0.95, 1.05)
    # print(scaled_image.shape, OPENAI_CLIP_STD.shape, OPENAI_CLIP_MEAN.shape, "SHAPES")
    return (scaled_image-getattr(Normalize, f"{args.gen_model}_MEAN"))/getattr(Normalize, f"{args.gen_model}_STD")


def load_nsd_data(data_folder, labels_folder, ss):
    """
    Return:
    - voxel_data: array of shape [num_voxels, num_features] (e.g.. S1 [10000, 19738]), the fmri data for each voxel
    - good_values: array of shape [num_voxels], the indices of the voxels that have valid fmri data
    """

    # load the preprocessed data files, made using code in nsd_preproc/code 
    
    # load info about images on each trial
    info_fn = os.path.join(labels_folder, 'S%d_image_info.csv'%(ss))
    print(info_fn)
    info = pd.read_csv(info_fn)

    image_order = np.array(info['unique_ims'])
    n_reps = np.array(info['n_reps'])

    # load fmri data
    data_filename = os.path.join(data_folder, 'S%d_betas_avg_bigmask.hdf5'%ss)
    print(data_filename)

    t = time.time()
    with h5py.File(data_filename, 'r') as data_set:
        values = np.copy(data_set['/betas'])
        data_set.close() 
    elapsed = time.time() - t
    print('Took %.5f seconds to load file'%elapsed)
    # data is organized as:
    # [images x voxels]

    # Some of these values may be nans, only for some subjects
    # this is for subjects who didn't complete all 40 sessions of NSD experiment.
    # make sure we remove the nans now.
    good_values = ~np.isnan(values[:,0])
    # print(values.shape)
    # print(np.sum(~good_values))

    # check that nans are exactly where we expect
    # nans happen when n_reps=0
    assert(np.all(good_values[n_reps>0]))
    assert(np.all(~good_values[n_reps==0]))

    voxel_data = values[good_values,:]
    # print(voxel_data.shape)
    
    return voxel_data, good_values


def load_from_nii(nii_file):
    return nib.load(nii_file).get_fdata()


def load_nsd_rois(ss, args):
    roi_path = os.path.join(args.roi_path.format(ss))
    print(f"Loading rois from: {roi_path}")

    roi_info = np.load(roi_path, allow_pickle=True).item()
    noise_ceiling = roi_info['noise_ceiling_avgreps'] / 100.

    big_mask = roi_info['voxel_mask']
    roi_keys = ['roi_labels_kastner', 'roi_labels_retino', 'roi_labels_face', 'roi_labels_place', 'roi_labels_body']
    roi_names = ['kastner_atlas_roi_names', 'ret_prf_roi_names',  'floc_face_roi_names', 'floc_place_roi_names', 'floc_body_roi_names']
    roi_masks = dict()
    for key, name in zip(roi_keys, roi_names):
        roi_labels = roi_info[key][big_mask]
        roi_name = roi_info[name]
        for name in roi_name.keys():
            roi_masks[name] = roi_labels==roi_name[name]
    
    # combine ventral and dorsal retinotopic regions  
    for name in ["V1", "V2", "V3"]:
        roi_masks[name] = roi_masks[name + "v"] | roi_masks[name + "d"]

    # combine FFA-1 and FFA-2
    roi_masks["FFA"] = roi_masks["FFA-1"] | roi_masks["FFA-2"]

    print(f"Loaded {len(roi_masks)} rois: {list(roi_masks.keys())}")
        
    return roi_masks, noise_ceiling


def build_brain_encoder(model_name, device):
    if model_name == "CLIP_RN50":
        visual_encoder, _ = clip.load("RN50", device=device)
        CNN_backbone = visual_encoder.visual.to(device)

        del visual_encoder.transformer
        torch.cuda.empty_cache()
        
        assert not visual_encoder.training

        return_nodes = {
            'relu1': 'relu1', #torch.Size([1, 32, 112, 112])
            'relu2': 'relu2', #torch.Size([1, 32, 112, 112])
            'relu3': 'relu', #torch.Size([1, 64, 112, 112])
            'avgpool': 'avgpool', #torch.Size([1, 64, 56, 56])
            'layer1': 'layer1', #torch.Size([1, 256, 56, 56])
            'layer2': 'layer2', #torch.Size([1, 512, 28, 28])
            'layer3': 'layer3', #torch.Size([1, 1024, 14, 14])
            'layer4': 'layer4' #torch.Size([1, 2048, 7, 7])
        }
    
    elif model_name == "OPEN_CLIP_RN50":
        model, _, preprocess = open_clip.create_model_and_transforms(
            'RN50', pretrained='yfcc15m', force_quick_gelu=True
        )
        CNN_backbone = model.visual
        del model.transformer
        torch.cuda.empty_cache()

        # train_nodes, val_nodes = get_graph_node_names(CNN_backbone)
        
        return_nodes = {
            'act1': 'act1', #torch.Size([1, 32, 112, 112])
            'act2': 'act2', #torch.Size([1, 32, 112, 112])
            'act3': 'relu', #torch.Size([1, 64, 112, 112])
            'avgpool': 'avgpool', #torch.Size([1, 64, 56, 56])
            'layer1': 'layer1', #torch.Size([1, 256, 56, 56])
            'layer2': 'layer2', #torch.Size([1, 512, 28, 28])
            'layer3': 'layer3', #torch.Size([1, 1024, 14, 14])
            'layer4': 'layer4' #torch.Size([1, 2048, 7, 7])
        }
    
    elif model_name == "DINO_RN50":
        CNN_backbone = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        return_nodes = {
            'relu': 'relu', #torch.Size([1, 64, 112, 112])
            'maxpool': 'maxpool', #torch.Size([1, 64, 56, 56])
            'layer1': 'layer1', #torch.Size([1, 256, 56, 56])
            'layer2': 'layer2', #torch.Size([1, 512, 28, 28])
            'layer3': 'layer3', #torch.Size([1, 1024, 14, 14])
            'layer4': 'layer4' #torch.Size([1, 2048, 7, 7])
        }
    
    elif model_name == "SIMCLR_RN50":
        repo_id = "lightly-ai/simclrv1-imagenet1k-resnet50-1x"
        filename = "resnet50-1x.pth"

        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        ckpt = torch.load(ckpt_path, map_location="cpu")

        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        CNN_backbone = resnet50(weights=None)

        CNN_backbone.load_state_dict(state_dict, strict=True)
        return_nodes = {
            'relu': 'relu', #torch.Size([1, 64, 112, 112])
            'maxpool': 'maxpool', #torch.Size([1, 64, 56, 56])
            'layer1': 'layer1', #torch.Size([1, 256, 56, 56])
            'layer2': 'layer2', #torch.Size([1, 512, 28, 28])
            'layer3': 'layer3', #torch.Size([1, 1024, 14, 14])
            'layer4': 'layer4' #torch.Size([1, 2048, 7, 7])
        }

    elif model_name == "ADV_RN50":
        dataset = ImageNet("/tmp")  # dummy path

        wrapped_model, _ = make_and_restore_model(
            arch="resnet50",
            dataset=dataset,
            resume_path="/user_data/junruz/prf_features/imagenet_l2_3_0.pt",
            parallel=False,
        )

        # Extract weights from the MadryLab ResNet
        robust_state_dict = {
            key: value.detach().cpu()
            for key, value in wrapped_model.model.state_dict().items()
        }

        # Create a normal torchvision ResNet-50
        CNN_backbone = resnet50(weights=None)

        # The parameter names and shapes should match exactly
        CNN_backbone.load_state_dict(robust_state_dict, strict=True)

        # CNN_backbone = wrapped_model.model
        return_nodes = {
            'relu': 'relu', #torch.Size([1, 64, 112, 112])
            'maxpool': 'maxpool', #torch.Size([1, 64, 56, 56])
            'layer1': 'layer1', #torch.Size([1, 256, 56, 56])
            'layer2': 'layer2', #torch.Size([1, 512, 28, 28])
            'layer3': 'layer3', #torch.Size([1, 1024, 14, 14])
            'layer4': 'layer4' #torch.Size([1, 2048, 7, 7])
        }

    elif model_name == "OPEN_CLIP_CONVNEXT_BASE":
        model, _, preprocess = open_clip.create_model_and_transforms('convnext_base', pretrained='laion400m_s13b_b51k')
        CNN_backbone = model.visual
        CNN_backbone.eval()
        # train, eval = get_graph_node_names(CNN_backbone)
        return_nodes = {
            "trunk.stem.1.permute_1": "stem",
            "trunk.stages.0.blocks.2.add": "stage1",
            "trunk.stages.1.blocks.2.add": "stage2",
            "trunk.stages.2.blocks.26.add": "stage3",
            "trunk.stages.3.blocks.2.add": "stage4"
        }

    
    feature_extractor = create_feature_extractor(CNN_backbone, return_nodes=return_nodes)
    for name, param in feature_extractor.named_parameters():
        param.requires_grad = False

    feature_extractor.eval().to(device)
    print(f"Created {model_name} and moved to {device}, feature extractor created")

    return feature_extractor


def load_concat_metadata(model_path, expected_model):
    """Load and validate the layer order used by a concatenated checkpoint."""
    metadata_path = os.path.join(model_path, "concat_metadata.json")
    with open(metadata_path, "r") as file:
        metadata = json.load(file)
    if metadata.get("model_name") != expected_model:
        raise ValueError(
            f"Checkpoint metadata model {metadata.get('model_name')!r} does "
            f"not match requested model {expected_model!r}: {metadata_path}"
        )
    if not metadata.get("layers"):
        raise ValueError(f"No concatenated layers recorded in {metadata_path}")
    return metadata


def validate_concat_layout(metadata, layer_sizes, description):
    """Ensure live encoder channels match the fitted concatenation layout."""
    expected_slices = {}
    start = 0
    for layer_name in metadata["layers"]:
        stop = start + int(layer_sizes[layer_name][0])
        expected_slices[layer_name] = [start, stop]
        start = stop
    if metadata.get("feature_slices") != expected_slices:
        raise ValueError(
            f"{description} feature layout mismatch: checkpoint has "
            f"{metadata.get('feature_slices')}, encoder requires "
            f"{expected_slices}"
        )
    if int(metadata.get("num_features", -1)) != start:
        raise ValueError(
            f"{description} metadata reports {metadata.get('num_features')} "
            f"features, but its layers contain {start}"
        )


def load_concatenated_features(features_model_folder, layer_names, prf_idx):
    """Load the same pRF ID from all layers and concatenate channel columns."""
    arrays = []
    num_images = None
    for layer_name in layer_names:
        feature_path = os.path.join(
            features_model_folder,
            layer_name,
            f"features_prf_{int(prf_idx)}.npy",
        )
        features = np.load(feature_path)
        if features.ndim != 2:
            raise ValueError(
                f"Expected [images, channels] in {feature_path}, got "
                f"{features.shape}"
            )
        if num_images is None:
            num_images = features.shape[0]
        elif features.shape[0] != num_images:
            raise ValueError(f"Image count mismatch in {feature_path}")
        arrays.append(features)
    return np.concatenate(arrays, axis=1)


def build_prf_kernels(layer_names, layer_sizes, prf_idx, device):
    """Build the same pRF ID at each selected layer's spatial resolution."""
    kernels = {}
    for layer_name in layer_names:
        n_pix = layer_sizes[layer_name][-1]
        if n_pix not in _PRF_STACK_CACHE:
            _PRF_STACK_CACHE[n_pix] = get_prf_stack(
                n_pix=n_pix, which_prf_grid="default-log-polar"
            )
        stack = _PRF_STACK_CACHE[n_pix]
        if not 0 <= int(prf_idx) < stack.shape[-1]:
            raise IndexError(
                f"pRF ID {prf_idx} is outside the grid for {layer_name} "
                f"(size {stack.shape[-1]})"
            )
        kernels[layer_name] = torch.as_tensor(
            stack[:, :, int(prf_idx)], device=device, dtype=torch.double
        )
    return kernels


def pool_concatenated_features(extracted, layer_names, prf_kernels):
    """pRF-pool every layer and concatenate channels in checkpoint order."""
    pooled = []
    for layer_name in layer_names:
        layer_features = extracted[layer_name].double()
        pooled.append(
            torch.einsum(
                "bchw,hw->bc", layer_features, prf_kernels[layer_name]
            )
        )
    return torch.cat(pooled, dim=1)


def save_online_ranking_model(
    output_folder,
    *,
    subject_id,
    region,
    voxel_id,
    voxel_index_in_region,
    voxel_rank,
    voxel_r2,
    generation_model_name,
    model_name,
    split_id,
    metadata,
    layer_sizes,
    prf_idx,
    prf_grid_name,
    prf_params,
    weights,
    intercept,
    feature_mean,
    feature_std,
    channel_indices,
    lambda_candidates,
    best_lambda_index,
    nested_sse,
    train_ids,
    val_ids,
    nest_ids,
    features_model_folder,
    ranking_model_path,
    data_splits_path,
):
    """Save a complete, per-voxel online ranking-model artifact."""
    os.makedirs(output_folder, exist_ok=True)

    parameters_filename = "parameters.npz"
    metadata_filename = "model.json"
    parameters_path = os.path.join(output_folder, parameters_filename)
    metadata_path = os.path.join(output_folder, metadata_filename)

    weights_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    intercept_array = np.asarray(intercept, dtype=np.float64).reshape(())
    feature_mean_array = np.asarray(feature_mean, dtype=np.float64).reshape(-1)
    feature_std_array = np.asarray(feature_std, dtype=np.float64).reshape(-1)
    channel_indices_array = np.asarray(channel_indices, dtype=np.int64).reshape(-1)
    lambda_candidates_array = np.asarray(
        lambda_candidates, dtype=np.float64
    ).reshape(-1)
    best_lambda_index = int(
        np.asarray(best_lambda_index).reshape(-1)[0]
    )
    nested_sse = float(np.asarray(nested_sse).reshape(-1)[0])

    num_features = int(metadata["num_features"])
    expected_shape = (num_features,)
    for array_name, array in (
        ("weights", weights_array),
        ("feature_mean", feature_mean_array),
        ("feature_std", feature_std_array),
        ("channel_indices", channel_indices_array),
    ):
        if array.shape != expected_shape:
            raise ValueError(
                f"Online ranking {array_name} has shape {array.shape}; "
                f"expected {expected_shape}"
            )
    if not 0 <= best_lambda_index < len(lambda_candidates_array):
        raise IndexError(
            f"Best lambda index {best_lambda_index} is outside a grid of "
            f"length {len(lambda_candidates_array)}"
        )

    train_ids_array = np.asarray(train_ids, dtype=np.int64).reshape(-1)
    val_ids_array = np.asarray(val_ids, dtype=np.int64).reshape(-1)
    nest_ids_array = np.asarray(nest_ids, dtype=np.int64).reshape(-1)
    partial_parameters_path = parameters_path + ".partial.npz"
    np.savez_compressed(
        partial_parameters_path,
        weights=weights_array,
        intercept=intercept_array,
        feature_mean=feature_mean_array,
        feature_std=feature_std_array,
        channel_indices=channel_indices_array,
        lambda_candidates=lambda_candidates_array,
        best_lambda_index=np.asarray(best_lambda_index, dtype=np.int64),
        best_lambda=np.asarray(
            lambda_candidates_array[best_lambda_index], dtype=np.float64
        ),
        nested_sse=np.asarray(nested_sse, dtype=np.float64),
        train_ids=train_ids_array,
        validation_ids=val_ids_array,
        nested_ids=nest_ids_array,
    )

    layer_names = list(metadata["layers"])
    layer_layout = []
    for layer_name in layer_names:
        channels, height, width = (
            int(value) for value in layer_sizes[layer_name]
        )
        layer_layout.append({
            "name": layer_name,
            "channels": channels,
            "height": height,
            "width": width,
            "feature_slice": list(metadata["feature_slices"][layer_name]),
        })

    prf_x, prf_y, prf_sigma = (
        float(value) for value in np.asarray(prf_params).reshape(3)
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "braindive_online_voxel_ranking_model",
        "subject": int(subject_id),
        "region": str(region),
        "voxel": {
            "id": int(voxel_id),
            "index_in_region": int(voxel_index_in_region),
            "generation_model_r2_rank": int(voxel_rank),
            "generation_model_r2": float(voxel_r2),
        },
        "ranking_encoder": {
            "name": model_name,
            "online_voxel_readout_fit": True,
            "pretrained_backbone_parameters_included": False,
            "provenance": MODEL_PROVENANCE.get(model_name, {}),
            "checkpoint_folder": os.path.abspath(ranking_model_path),
            "checkpoint_concat_metadata": metadata,
            "input": {
                "color_space": "RGB",
                "range": [0.0, 1.0],
                "resize": [224, 224],
                "resize_mode": "bilinear",
                "antialias": True,
                "mean": getattr(
                    Normalize, f"{model_name}_MEAN"
                ).reshape(-1).astype(float).tolist(),
                "std": getattr(
                    Normalize, f"{model_name}_STD"
                ).reshape(-1).astype(float).tolist(),
            },
            "concatenation_order": layer_names,
            "layers": layer_layout,
            "num_features": num_features,
        },
        "prf": {
            "source": "generation_model_best_prf_idx",
            "generation_model_name": generation_model_name,
            "shared_with_generation_model": True,
            "grid_name": prf_grid_name,
            "id": int(prf_idx),
            "x_aperture_units": prf_x,
            "y_aperture_units": prf_y,
            "sigma_aperture_units": prf_sigma,
            "nsd_aperture_degrees": 8.4,
            "x_degrees": prf_x * 8.4,
            "y_degrees": prf_y * 8.4,
            "sigma_degrees": prf_sigma * 8.4,
            "pooling": "normalized_isotropic_gaussian_weighted_mean",
            "rasterized_at_each_layer_native_resolution": True,
        },
        "fit": {
            "model_type": "ridge_regression_with_intercept",
            "target": "NSD averaged beta response for this voxel",
            "split_id": int(split_id),
            "feature_normalization": "per-channel z-score",
            "normalization_statistics_partition": "train_plus_nested",
            "weights_fit_partition": "train",
            "lambda_selection_partition": "nested",
            "validation_partition_used_during_fit": False,
            "intercept_in_ridge_penalty": True,
            "ridge_solver_epsilon": 1e-4,
            "best_lambda_index": best_lambda_index,
            "best_lambda": float(
                lambda_candidates_array[best_lambda_index]
            ),
            "best_nested_sse": nested_sse,
            "partition_counts": {
                "train": int(len(train_ids_array)),
                "validation": int(len(val_ids_array)),
                "nested": int(len(nest_ids_array)),
            },
            "prediction_equation": (
                "score = dot((features - feature_mean) / feature_std, "
                "weights) + intercept"
            ),
        },
        "source_data": {
            "precomputed_feature_folder": os.path.abspath(
                features_model_folder
            ),
            "data_splits_file": os.path.abspath(data_splits_path),
        },
        "artifacts": {
            "parameters_file": parameters_filename,
            "parameter_arrays": {
                "weights": "float64[num_features]",
                "intercept": "float64 scalar",
                "feature_mean": "float64[num_features]",
                "feature_std": "float64[num_features]",
                "channel_indices": "int64[num_features]",
                "lambda_candidates": "float64[num_lambdas]",
                "best_lambda_index": "int64 scalar",
                "best_lambda": "float64 scalar",
                "nested_sse": "float64 scalar",
                "train_ids": "int64[num_train_images]",
                "validation_ids": "int64[num_validation_images]",
                "nested_ids": "int64[num_nested_images]",
            },
        },
    }
    partial_metadata_path = metadata_path + ".partial"
    with open(partial_metadata_path, "w") as metadata_handle:
        json.dump(payload, metadata_handle, indent=2, allow_nan=False)
    os.replace(partial_parameters_path, parameters_path)
    os.replace(partial_metadata_path, metadata_path)

    return metadata_path, parameters_path


def rank_model_fitting(voxel_id, best_prf_idx, voxel_data, train_ids, val_ids, nest_ids, features_model_folder, layer_names, device):
    """Fit one online concatenated-feature ridge readout for ranking."""

    train_voxel = voxel_data[train_ids, voxel_id]
    nest_voxel = voxel_data[nest_ids, voxel_id]

    train_voxel = torch.from_numpy(train_voxel).float().unsqueeze(1).to(device)
    nest_voxel = torch.from_numpy(nest_voxel).float().unsqueeze(1).to(device)

    features = load_concatenated_features(
        features_model_folder, layer_names, best_prf_idx
    )

    # split train and nest features, then z-score the features across images axis 0 using model_fitting_utils.split_normalize_feats
    # train_features, val_features, nest_features, features_s, features_m = model_fitting_utils.split_normalize_feats(features, train_ids, val_ids, nest_ids)
    train_features, val_features, nest_features, features_s, features_m = (
        model_fitting_utils.split_normalize_feats(
            features, train_ids, val_ids, nest_ids
        )
    )
    channel_kept = np.arange(features.shape[1], dtype=int)
    # train_features, val_features, nest_features, keep_idx = model_fitting_utils.split_feats(features, train_ids, val_ids, nest_ids)

    # add the intercept: a column of ones
    train_features = np.concatenate([train_features, np.ones(shape=(len(train_features), 1), dtype=train_features.dtype)], axis=1)
    nest_features = np.concatenate([nest_features, np.ones(shape=(len(nest_features), 1), dtype=nest_features.dtype)], axis=1)

    train_features = torch.from_numpy(train_features).float().to(device)
    nest_features = torch.from_numpy(nest_features).float().to(device)
    n_lambdas = 20
    small_value = 0.0001
    lambdas = np.logspace(np.log(small_value),np.log(10**10+small_value),n_lambdas, \
                        dtype=np.float32, base=np.e) - small_value
    
    best_weights, best_lambda_idx, best_nest_loss = model_fitting_utils.solve_ridge(train_features, train_voxel, nest_features, nest_voxel, lambdas, eps=1e-4, return_loss=True)
    # best_weights, best_lambda_idx, best_nest_loss = model_fitting_utils.solve_ridge_svd(train_features, train_voxel, nest_features, nest_voxel, lambdas, eps=1e-4, return_loss=True)


    return (
        best_weights,
        channel_kept,
        best_lambda_idx,
        best_nest_loss,
        lambdas,
        features_s,
        features_m,
    )


def build_image_save_metadata(mode, stages, ranking_enabled):
    """Explain image provenance independently of the folder's legacy name."""
    adjusted = any(stage != 'none' for stage in stages)
    return {
        'mode': mode,
        'constraint_timings': dict(zip(
            ('contrast_full_image', 'contrast_prf', 'luminance_full_image', 'luminance_prf'), stages)),
        'pre_adjustment': {
            'saving_enabled': mode != 'constrained_only' and adjusted,
            'folder': 'unconstrained_images',
            'capture_stage': 'final_decode_before_last_constraint_adjustment',
            'guidance_already_used_constraints': 'every_step' in stages,
            'meaning': 'Before the last adjustment, not a separate unconstrained generation.',
        },
        'post_adjustment': {
            'saving_enabled': mode != 'unconstrained_only',
            'folder': 'images',
        },
        'format': 'PNG', 'color_mode': 'RGB', 'bits_per_channel': 8,
        'quantization': 'round(clip(float_pixels, 0, 1) * 255) to uint8',
        'exact_float_decode_preserved': False,
        'final_adjustment_configured': adjusted,
        'final_adjustment_enabled': adjusted and mode != 'unconstrained_only',
        'save_mode_skips_final_adjustment': mode == 'unconstrained_only',
        'final_target_validation_applied': adjusted and mode != 'unconstrained_only',
        'final_adjustment_rejected_seeds_saved': False,
        'selection': 'ranking_model_top_n' if ranking_enabled else 'all_valid_candidates_in_seed_attempt_order',
        'scores_computed_on': (
            ('unadjusted_final_decode' if mode == 'unconstrained_only'
             else 'pipeline_output_after_last_adjustment_if_enabled') if ranking_enabled else None),
    }


def generation_summary_for_mode(summary, ranking_enabled):
    """Keep ranking metadata only when scoring/ranking actually ran."""
    summary['ranking_enabled'] = ranking_enabled
    if not ranking_enabled:
        for key in ('ranking_definitions', 'candidate_rankings', 'ranking_top_n_requested',
                    'ranking_top_n_returned', 'ranking_top_n', 'online_ranking_model'):
            summary.pop(key, None)
        summary['image_order'] = 'valid_seed_attempt_order'
        summary['valid_seed_definition'] = 'passed_image_and_constraint_checks_without_post_generation_scoring'
        summary['rejected_candidates'] = {
            key: value for key, value in summary['rejected_candidates'].items()
            if key not in ('nonfinite_gen_score', 'nonfinite_rank_score')
        }
    return summary


def parse_ranking_top_n(value):
    """Accept a positive saved-image count or the literal 'all'."""
    if value == "all":
        return "all"
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("ranking_top_n must be a positive integer or 'all'") from error
    if count < 1:
        raise argparse.ArgumentTypeError("ranking_top_n must be positive or 'all'")
    return count


def resolve_luminance_targets(args):
    """Fill omitted active targets with the subject's mean image luma (byte units)."""
    sources = {}
    missing = []
    for scope in ('full_image', 'prf'):
        enabled = getattr(args, f'luminance_constraint_{scope}') != 'none'
        target = getattr(args, f'target_luminance_{scope}')
        sources[scope] = 'explicit' if enabled and target is not None else None
        if enabled and target is None:
            missing.append(scope)
    metadata = {'target_sources': sources, 'statistics_path': None}
    if not missing:
        return metadata

    path = args.contrast_stats_json or os.path.join(
        args.model_root, f'S{args.subject_id}', 'nsd_contrast_statistics.json')
    try:
        with open(path) as handle:
            statistics = json.load(handle)
        if 'subject' in statistics and statistics['subject'] != args.subject_id:
            raise ValueError(
                f"statistics subject {statistics['subject']} does not match "
                f"requested subject {args.subject_id}")
        mean = float(statistics['luminance_mean']['byte_0_255']['mean'])
        if not np.isfinite(mean) or not 0 <= mean <= 255:
            raise ValueError('mean luminance must be finite and in [0, 255]')
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Cannot load mean luminance from {path}: {error}. "
            "Run calculate_nsd_contrast_statistics.py for this subject, pass "
            "--contrast_stats_json, or supply explicit luminance targets."
        ) from error

    for scope in missing:
        setattr(args, f'target_luminance_{scope}', mean)
        sources[scope] = 'luminance_mean.byte_0_255.mean'
    metadata.update(statistics_path=os.path.abspath(path),
                    statistics_subject=statistics.get('subject'),
                    statistics_selection=statistics.get('selection'),
                    statistics_image_count=statistics.get('image_count'),
                    dataset_mean_luminance_byte=mean)
    return metadata


def main():
    ### PATHS ############################################################
    nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
    data_folder = os.path.join(nsd_path, 'data')
    labels_folder = os.path.join(nsd_path, 'labels')
    stim_folder = os.path.join(nsd_path, 'stimuli')
    rois_folder = os.path.join(nsd_path, 'rois')

    ### ARGUMENTS ############################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=int, default=1)
    parser.add_argument('--roi_path', type=str, default=os.path.join(rois_folder, 'S{}_voxel_roi_info.npy'))
    parser.add_argument('--gen_model', type=str, default="DINO_RN50")
    parser.add_argument('--rank_model', type=str, default="OPEN_CLIP_RN50")
    parser.add_argument('--enable_ranking', choices=['true', 'false'], default='true',
                        help="Enable online ranking and post-generation scoring; false saves all pixel-valid candidates")
    parser.add_argument('--split_id', type=int, choices=[1, 2], default=1)
    parser.add_argument('--feature_root', type=str, default="/user_data/junruz/prf_features")
    parser.add_argument('--model_root', type=str, default="/user_data/junruz/prf_models/concat/split_1_zscore")
    parser.add_argument('--roi', nargs='+', default=["V1"], type=str)
    parser.add_argument('--k', type=int, default=3, help="Top k voxels to keep in the region")
    parser.add_argument(
        '--voxel_rank_range', nargs=2, type=int, metavar=('START', 'END'),
        help=("Inclusive 1-based voxel rank range; overrides --k. "
              "For example, --voxel_rank_range 6 10 selects ranks 6-10."),
    )
    parser.add_argument(
        '--r2_noise_ceiling_threshold',
        type=float,
        default=0.0,
        help=(
            "Noise-ceiling threshold used when r2.pkl was created. The pRF "
            "evaluation script defaults to 0.0 and stores R² values only for "
            "ROI voxels above this threshold."
        ),
    )
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--brain_guidance_scale', type=float, default=30)
    parser.add_argument(
        '--sag_scale',
        type=float,
        default=0.75,
        help=(
            "Self-attention guidance scale. 0 disables SAG; BrainDiVE default "
            "is 0.75."
        ),
    )
    parser.add_argument(
        '--contrast_constraint_method',
        choices=['max', 'range', 'match'],
        default='range',
        help=(
            "Which whole-image projection to use: 'max' caps RMS contrast, "
            "'range' keeps it between lower/upper bounds, and 'match' "
            "projects every image to one target RMS contrast. Ignored when "
            "--contrast_constraint_full_image is 'none'. The soft-pRF "
            "constraint always matches an exact target."
        ),
    )
    parser.add_argument(
        '--contrast_constraint_full_image',
        choices=['none', 'every_step', 'final'],
        default='every_step',
        help=(
            "When to constrain whole-image RMS contrast. 'every_step' "
            "projects before every brain-guidance evaluation and again "
            "after the last diffusion step, 'final' projects only after "
            "the last step, and 'none' disables the constraint."
        ),
    )
    parser.add_argument(
        '--contrast_constraint_prf',
        choices=['none', 'every_step', 'final'],
        default='none',
        help=(
            "When to additionally match soft-pRF-weighted RMS contrast to "
            "the same target, using the same stage names. It cannot run at "
            "a stage where --contrast_constraint_full_image does not, "
            "because the dual-scope solver drives both statistics to one "
            "shared target, and it requires "
            "--contrast_constraint_method match."
        ),
    )
    parser.add_argument(
        '--contrast_stats_json',
        '--contrast_stats_path',
        dest='contrast_stats_json',
        type=str,
        default=None,
        help=(
            "NSD contrast and mean-luminance statistics JSON. Defaults to "
            "MODEL_ROOT/SUBJECT/nsd_contrast_statistics.json"
        ),
    )
    parser.add_argument('--min_contrast_percentile', type=float, default=5.0)
    parser.add_argument('--max_contrast_percentile', type=float, default=95.0)
    parser.add_argument(
        '--target_contrast_percentile',
        type=float,
        default=50.0,
        help=(
            "Statistics percentile used as the target for the 'match' method "
            "when --target_rms_contrast is omitted"
        ),
    )
    parser.add_argument(
        '--min_rms_contrast',
        type=float,
        default=None,
        help="Optional lower-bound override in 0-255 pixel units",
    )
    parser.add_argument(
        '--max_rms_contrast',
        type=float,
        default=None,
        help="Optional upper-bound override in 0-255 pixel units",
    )
    parser.add_argument(
        '--target_rms_contrast',
        type=float,
        default=None,
        help=(
            "Exact RMS contrast for the 'match' method in 0-255 pixel units"
        ),
    )
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--seed_mode', choices=['random', 'file', 'fixed'], default='random',
                        help="Fresh random sequence, JSON seed file, or fixed-generator sequence")
    parser.add_argument('--seed_file', type=str, default=None,
                        help="JSON list or saved seeds.json; required for seed_mode=file")
    parser.add_argument('--seed_file_key', choices=['all_seeds_attempted', 'valid_seeds', 'seed_set'],
                        default='all_seeds_attempted', help="List to load from a JSON object")
    parser.add_argument('--seed_generator_seed', type=int, default=None,
                        help="Seed for generating the image-seed sequence in fixed mode")
    parser.add_argument('--ranking_top_n', type=parse_ranking_top_n, default=5,
                        help="Save the top N accepted images, or 'all' to save every accepted image")
    parser.add_argument('--image_save_mode',
                        choices=['unconstrained_only', 'constrained_only', 'both'], default='both',
                        help="PNG versions to save; unconstrained_only skips the last adjustment and its target checks")
    parser.add_argument('--luminance_constraint_full_image',
                        choices=['none', 'every_step', 'final'], default='none')
    parser.add_argument('--luminance_constraint_prf',
                        choices=['none', 'every_step', 'final'], default='none')
    parser.add_argument('--target_luminance_full_image', type=float, default=None,
                        help="Override whole-image mean luma (0-255); defaults to the NSD dataset mean")
    parser.add_argument('--target_luminance_prf', type=float, default=None,
                        help="Override soft-pRF mean luma (0-255); defaults to the same NSD dataset mean")
    parser.add_argument('--luminance_validation_tolerance', type=float, default=0.01,
                        help=("Allowed mean-luma error in 0-255 units: pre-RGB-clipping "
                              "luma for luma_rgb_clip, clipped RGB for joint_solver"))
    parser.add_argument('--luma_contrast_method', choices=['luma_rgb_clip', 'joint_solver'],
                        default='luma_rgb_clip', help=(
                            "luma_rgb_clip: match luma then restore original chroma and clip RGB; "
                            "requires paired mean/RMS timings per scope and method=match. "
                            "joint_solver retains strict post-RGB-clipping matching."))
    parser.add_argument('--luma_rgb_clip_match_iterations', type=int, default=20,
                        help="Full-only luma loop: at most N+1 passes, matching the reference")
    parser.add_argument('--luminance_shift_limit', type=float, default=1.0,
                        help="Bound on local/global brightness shifts in normalized [0,1] units")
    parser.add_argument(
        '--generation_timing_file', type=str, default=None,
        help=("Optional timing JSON mirror; written under a roi-<name> subfolder "
              "of the supplied parent directory. CUDA synchronized around every pipeline call."),
    )
    parser.add_argument(
        '--max_generation_attempts',
        type=int,
        default=None,
        help=(
            "Maximum seed attempts per voxel, including replacements for "
            "invalid candidates. Defaults to num_seeds + max(10%%, 10)."
        ),
    )
    parser.add_argument(
        '--contrast_validation_tolerance',
        type=float,
        default=0.01,
        help=(
            "Allowed unquantized-float RMS-contrast error in 0-255 pixel "
            "units. luma_rgb_clip checks projected luma before RGB clipping; "
            "joint_solver checks clipped RGB. The reference tolerance is 0.01."
        ),
    )
    parser.add_argument('--dual_scope_solver_max_nfev', type=int, default=300)
    parser.add_argument(
        '--dual_scope_every_step_iterations', type=int, default=10
    )
    parser.add_argument('--dual_scope_parameter_limit', type=float, default=8.0)
    parser.add_argument('--dual_scope_envelope_power', type=float, default=0.5)
    parser.add_argument('--save_root', type=str, default="/user_data/junruz/BrainDiVE/pRF_concat_model_ranked")
    args = parser.parse_args()

    try:
        validate_luma_contrast_method(
            args.luma_contrast_method, args.contrast_constraint_full_image,
            args.contrast_constraint_prf, args.luminance_constraint_full_image,
            args.luminance_constraint_prf, args.contrast_constraint_method,
            args.luma_rgb_clip_match_iterations,
        )
        luminance_target_metadata = resolve_luminance_targets(args)
    except ValueError as error:
        parser.error(str(error))

    luminance_options = dict(
        apply_final_adjustment=args.image_save_mode != 'unconstrained_only',
        luma_contrast_method=args.luma_contrast_method,
        luma_rgb_clip_match_iterations=args.luma_rgb_clip_match_iterations,
        luminance_constraint_full_image=args.luminance_constraint_full_image,
        luminance_constraint_prf=args.luminance_constraint_prf,
        target_luminance_full_image=(args.target_luminance_full_image / 255
                                    if args.target_luminance_full_image is not None else None),
        target_luminance_prf=(args.target_luminance_prf / 255
                             if args.target_luminance_prf is not None else None),
        luminance_validation_tolerance=args.luminance_validation_tolerance / 255,
        luminance_shift_limit=args.luminance_shift_limit,
    )
    try:
        validate_luminance_settings(
            args.luminance_constraint_full_image, args.luminance_constraint_prf,
            luminance_options['target_luminance_full_image'],
            luminance_options['target_luminance_prf'],
            luminance_options['luminance_validation_tolerance'],
            args.contrast_constraint_full_image, args.contrast_constraint_method,
            args.luminance_shift_limit,
        )
    except ValueError as error:
        parser.error(str(error))
    stage_order = {'none': 0, 'final': 1, 'every_step': 2}
    constraint_method_metadata = method_metadata(
        args.luma_contrast_method,
        max((args.contrast_constraint_full_image, args.luminance_constraint_full_image), key=stage_order.get),
        max((args.contrast_constraint_prf, args.luminance_constraint_prf), key=stage_order.get),
        args.luma_rgb_clip_match_iterations,
    )
    acceptance_domain = constraint_method_metadata['acceptance_domain']
    if args.image_save_mode == 'unconstrained_only':
        constraint_method_metadata['configured_final_solver'] = constraint_method_metadata['effective_solver']['final']
        constraint_method_metadata['effective_solver']['final'] = 'skipped_unconstrained_only'
        acceptance_domain = 'unadjusted_decode_pixel_validity_only'
    print(f"Luma/contrast method: {constraint_method_metadata}")
    luminance_metadata = {
        **luminance_target_metadata,
        'quantity': 'gamma_coded_Rec709_mean_luma',
        'full_image': args.luminance_constraint_full_image,
        'prf': args.luminance_constraint_prf,
        'target_full_image_byte': (args.target_luminance_full_image
                                   if args.luminance_constraint_full_image != 'none' else None),
        'target_prf_byte': (args.target_luminance_prf
                            if args.luminance_constraint_prf != 'none' else None),
        'acceptance_domain': acceptance_domain,
        'final_target_validation_applied': args.image_save_mode != 'unconstrained_only',
        'projection_method': constraint_method_metadata,
        'acceptance_tolerance_byte': args.luminance_validation_tolerance,
        'shift_limit_normalized': args.luminance_shift_limit,
        'joint_solver_tolerance': 1e-8,
        'joint_solver_max_nfev': args.dual_scope_solver_max_nfev,
        'joint_every_step_iterations': args.dual_scope_every_step_iterations,
        'prf_source': 'highest_resolution_generation_layer',
    }

    ss = args.subject_id
    if args.num_seeds < 1:
        parser.error("--num_seeds must be at least 1")
    if (
        args.max_generation_attempts is not None
        and args.max_generation_attempts < args.num_seeds
    ):
        parser.error("--max_generation_attempts cannot be less than --num_seeds")
    if args.contrast_validation_tolerance <= 0:
        parser.error("--contrast_validation_tolerance must be positive")
    contrast_stage_rank = {'none': 0, 'final': 1, 'every_step': 2}
    if (
        contrast_stage_rank[args.contrast_constraint_prf]
        > contrast_stage_rank[args.contrast_constraint_full_image]
    ):
        parser.error(
            "--contrast_constraint_prf cannot run at a stage where "
            "--contrast_constraint_full_image does not, because the "
            "dual-scope solver drives whole-image and soft-pRF contrast to "
            f"the same target; got prf={args.contrast_constraint_prf} with "
            f"full_image={args.contrast_constraint_full_image}"
        )
    if (
        args.contrast_constraint_prf != 'none'
        and args.contrast_constraint_method != 'match'
    ):
        parser.error(
            "--contrast_constraint_prf requires "
            "--contrast_constraint_method match"
        )
    if args.dual_scope_solver_max_nfev <= 0:
        parser.error("--dual_scope_solver_max_nfev must be positive")
    if args.dual_scope_every_step_iterations <= 0:
        parser.error("--dual_scope_every_step_iterations must be positive")
    if args.dual_scope_parameter_limit <= 0:
        parser.error("--dual_scope_parameter_limit must be positive")
    if args.dual_scope_envelope_power <= 0:
        parser.error("--dual_scope_envelope_power must be positive")

    contrast_stats_path = None
    try:
        seed_plan = build_seed_plan(
            args.seed_mode, args.num_seeds, args.max_generation_attempts,
            args.seed_file, args.seed_generator_seed, args.seed_file_key)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Invalid seed configuration: {error}")
    print(f"Seed mode: {seed_plan['mode']}; plan: {seed_plan['plan_id']}; "
          f"attempt budget per voxel: {seed_plan['effective_max_attempts']}")
    percentile_values = None
    needs_contrast_statistics = args.contrast_constraint_full_image != 'none' and (
        (args.contrast_constraint_method == 'max' and args.max_rms_contrast is None)
        or (
            args.contrast_constraint_method == 'range'
            and (
                args.min_rms_contrast is None
                or args.max_rms_contrast is None
            )
        )
        or (
            args.contrast_constraint_method == 'match'
            and args.target_rms_contrast is None
        )
    )
    if needs_contrast_statistics:
        contrast_stats_path = args.contrast_stats_json or os.path.join(
            args.model_root,
            f"S{ss}",
            "nsd_contrast_statistics.json",
        )
        try:
            with open(contrast_stats_path, "r") as contrast_stats_file:
                contrast_stats = json.load(contrast_stats_file)
            percentile_values = contrast_stats["rms_contrast"]["byte_0_255"][
                "percentiles"
            ]
        except FileNotFoundError:
            parser.error(
                f"Contrast-statistics JSON not found: {contrast_stats_path}"
            )
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            parser.error(
                f"Invalid contrast-statistics JSON {contrast_stats_path}: {error}"
            )

    def get_contrast_percentile(percentile):
        if percentile_values is None:
            raise RuntimeError("Contrast statistics were not loaded")
        for key, value in percentile_values.items():
            if float(key) == percentile:
                return float(value)
        raise ValueError(
            f"Percentile {percentile:g} is not present in {contrast_stats_path}; "
            f"available percentiles: {list(percentile_values)}"
        )

    if args.contrast_constraint_full_image == 'none':
        args.min_rms_contrast = None
        args.max_rms_contrast = None
        args.target_rms_contrast = None
        contrast_description = "disabled"
    elif args.contrast_constraint_method == 'max':
        if args.max_rms_contrast is None:
            args.max_rms_contrast = get_contrast_percentile(
                args.max_contrast_percentile
            )
        if not 0.0 < args.max_rms_contrast <= 127.5:
            parser.error("--max_rms_contrast must be in (0, 127.5]")
        args.min_rms_contrast = None
        args.target_rms_contrast = None
        contrast_description = f"maximum {args.max_rms_contrast:.6g}"
    elif args.contrast_constraint_method == 'range':
        if args.min_rms_contrast is None:
            args.min_rms_contrast = get_contrast_percentile(
                args.min_contrast_percentile
            )
        if args.max_rms_contrast is None:
            args.max_rms_contrast = get_contrast_percentile(
                args.max_contrast_percentile
            )
        if not 0.0 <= args.min_rms_contrast <= args.max_rms_contrast <= 127.5:
            parser.error(
                "RMS contrast bounds must satisfy "
                "0 <= min <= max <= 127.5 in 0-255 pixel units; got "
                f"{args.min_rms_contrast} and {args.max_rms_contrast}"
            )
        args.target_rms_contrast = None
        contrast_description = (
            f"range [{args.min_rms_contrast:.6g}, "
            f"{args.max_rms_contrast:.6g}]"
        )
    elif args.contrast_constraint_method == 'match':
        if args.target_rms_contrast is None:
            args.target_rms_contrast = get_contrast_percentile(
                args.target_contrast_percentile
            )
        if not 0.0 < args.target_rms_contrast <= 127.5:
            parser.error("--target_rms_contrast must be in (0, 127.5]")
        args.min_rms_contrast = None
        args.max_rms_contrast = None
        contrast_description = f"target {args.target_rms_contrast:.6g}"

    constraint_stages = (
        args.contrast_constraint_full_image, args.contrast_constraint_prf,
        args.luminance_constraint_full_image, args.luminance_constraint_prf,
    )
    image_saving_metadata = build_image_save_metadata(
        args.image_save_mode, constraint_stages, args.enable_ranking == 'true')
    if args.image_save_mode == 'unconstrained_only' and all(stage == 'none' for stage in constraint_stages):
        parser.error("--image_save_mode unconstrained_only requires an enabled constraint (every_step or final)")
    save_constrained_images = args.image_save_mode != 'unconstrained_only'
    save_unconstrained_images = (
        args.image_save_mode != 'constrained_only' and any(stage != 'none' for stage in constraint_stages)
    )
    if not save_unconstrained_images:
        unconstrained_image_content = None
    elif 'every_step' in constraint_stages:
        unconstrained_image_content = "pre_final_projection_decode_with_constrained_guidance"
    else:
        unconstrained_image_content = "raw_decode"

    print(
        f"Using Rec.709 RMS contrast method "
        f"{args.contrast_constraint_method!r}: {contrast_description} "
        f"in 0-255 pixel units; "
        f"full_image={args.contrast_constraint_full_image!r}; "
        f"prf={args.contrast_constraint_prf!r}; float tolerance="
        f"{args.contrast_validation_tolerance:g} byte; "
        f"unconstrained images saved={save_unconstrained_images}"
    )
    print(f"Luminance settings: {luminance_metadata}; keep={args.ranking_top_n}")

    def path_component(value):
        """Return a readable, filesystem-safe output path component."""
        cleaned = "".join(
            character if character.isalnum() or character in "-_." else "-"
            for character in str(value).strip()
        )
        return cleaned.strip("-.") or "unnamed"

    gen_layer_sizes = getattr(LayerSize, f"{args.gen_model}_layer_size")
    if args.enable_ranking == 'false':
        print("Ranking disabled: ignoring --ranking_top_n; saving all pixel-valid candidates in seed-attempt order.")
    rank_layer_sizes = (getattr(LayerSize, f"{args.rank_model}_layer_size")
                        if args.enable_ranking == 'true' else None)
    n_pix_img = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware_metadata = collect_device_metadata(device, torch)
    print(f"Generation hardware: {hardware_metadata}")

    ###### BRAIN ENCODER ###################################################
    print("Building brain encoder")
    brain_encoder = build_brain_encoder(args.gen_model, device)
    rank_encoder = None
    if args.enable_ranking == 'true':
        print("Building rank encoder")
        rank_encoder = build_brain_encoder(args.rank_model, device)


    def preprocess_encoder_input(image_tensor, model_name):
        """Resize a BCHW [0, 1] tensor and apply model-specific normalization."""
        image_tensor = torch.nn.functional.interpolate(
            image_tensor.float(),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        mean = torch.as_tensor(
            getattr(Normalize, f"{model_name}_MEAN"),
            device=image_tensor.device,
            dtype=image_tensor.dtype,
        )
        std = torch.as_tensor(
            getattr(Normalize, f"{model_name}_STD"),
            device=image_tensor.device,
            dtype=image_tensor.dtype,
        )
        return (image_tensor - mean) / std


    def score_generated_images_for_voxel(
        image_records,
        layer_names,
        prf_kernels,
        voxel_weights,
        voxel_intercept,
        features_s,
        features_m,
        model_name,
        batch_size=8,
        encoder=rank_encoder,
    ):
        if len(image_records) == 0:
            return []

        scores = []
        with torch.no_grad():
            for batch_start in range(0, len(image_records), batch_size):
                batch_records = image_records[batch_start: batch_start + batch_size]
                batch_tensor = torch.stack(
                    [item["tensor"] for item in batch_records], dim=0
                ).to(device)
                batch_tensor = preprocess_encoder_input(batch_tensor, model_name)
                image_features = encoder(batch_tensor)
                prf_features = pool_concatenated_features(
                    image_features, layer_names, prf_kernels
                )
                prf_features = (prf_features - features_m) / features_s
                voxel_responses = (
                    torch.matmul(prf_features, voxel_weights.reshape(-1))
                    + voxel_intercept.reshape(())
                )
                scores.extend(voxel_responses.detach().cpu().numpy().tolist())

        return scores



    ############################## Brain PRF Model #########################################

    model_path = os.path.join(
        args.model_root,
        f"S{args.subject_id}",
        f"{args.gen_model}_set{args.split_id}",
    )
    weights_path = os.path.join(model_path, 'best_weights.pkl')
    prf_idx_path = os.path.join(model_path, 'best_prf_idx.npy')
    lambda_path = os.path.join(model_path, 'best_lambda.npy')
    r2_path = os.path.join(model_path, 'r2.pkl')
    features_s_path = os.path.join(model_path, 'best_features_s.npy')
    features_m_path = os.path.join(model_path, 'best_features_m.npy')

    gen_metadata = load_concat_metadata(model_path, args.gen_model)
    gen_layer_names = gen_metadata["layers"]
    unknown_gen_layers = set(gen_layer_names) - set(gen_layer_sizes)
    if unknown_gen_layers:
        parser.error(
            f"Unknown generation layers in concat metadata: "
            f"{sorted(unknown_gen_layers)}"
        )
    validate_concat_layout(gen_metadata, gen_layer_sizes, "Generation")

    rank_layer_names = []
    if args.enable_ranking == 'true':
        rank_model_path = os.path.join(
            args.model_root,
            f"S{args.subject_id}",
            f"{args.rank_model}_set{args.split_id}",
        )
        rank_metadata = load_concat_metadata(rank_model_path, args.rank_model)
        rank_layer_names = rank_metadata["layers"]
        unknown_rank_layers = set(rank_layer_names) - set(rank_layer_sizes)
        if unknown_rank_layers:
            parser.error(
                f"Unknown ranking layers in concat metadata: "
                f"{sorted(unknown_rank_layers)}"
            )
        validate_concat_layout(rank_metadata, rank_layer_sizes, "Ranking")
    print(
        f"Generation concatenation: {args.gen_model} layers "
        f"{gen_layer_names} ({gen_metadata['num_features']} features)"
    )
    if args.enable_ranking == 'true':
        print(
            f"Ranking concatenation: {args.rank_model} layers "
            f"{rank_layer_names} ({rank_metadata['num_features']} features)"
        )


    ####################### Turn on if zscore is used #######################
    # channel_mean = np.load(os.path.join(feature_path, 'channel_mean.npy'))
    # channel_std = np.load(os.path.join(feature_path, 'channel_std.npy'))
    
    with open(weights_path, "rb") as f:
        best_weights = pickle.load(f)
    best_prf_idx = np.load(prf_idx_path)
    with open(r2_path, "rb") as f:
        r2 = pickle.load(f)
    best_features_s = np.load(features_s_path)
    best_features_m = np.load(features_m_path)

    best_features_s = torch.from_numpy(best_features_s).to(device)
    best_features_m = torch.from_numpy(best_features_m).to(device)
    if best_features_m.shape[1] != int(gen_metadata["num_features"]):
        raise ValueError(
            f"Generation normalization has {best_features_m.shape[1]} "
            f"features, metadata expects {gen_metadata['num_features']}"
        )
    # Load rois masks
    roi_masks, noise_ceiling = load_nsd_rois(ss, args)
    if args.enable_ranking == 'true':
        split_dir = os.path.join(args.model_root, f'S{ss}')
        data_splits_path = os.path.join(split_dir, f'data_splits_S{ss}.pkl')
        with open(data_splits_path, 'rb') as f:
            data_splits = pickle.load(f)
        ids = data_splits[args.split_id]
        train_ids, val_ids, nest_ids = ids['train'], ids['val'], ids['nest']
        voxel_data, good_values = load_nsd_data(data_folder, labels_folder, ss)


    ########################## Stable Diffusion Pipeline #########################################
    repo_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = mypipelineSAG.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)


    ############################## Loop over regions #########################################
    regions = args.roi
    # random.seed(a=b64encode(os.urandom(5)).decode('utf-8'))
    # random.shuffle(regions)

    generation_timing = {"seconds": 0.0, "calls": 0, "voxels": {}}
    prf_grid_params, prf_grid_name = get_prf_grid("default-log-polar")
    for region in regions:
        print("Starting S{} {}".format(ss, region))
        # random.seed(a=b64encode(os.urandom(5)).decode('utf-8'))
        # random.shuffle(region_seeds)
        try:
            del region_mask
        except:
            pass

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()


        region_mask = roi_masks[region]
        # r2.pkl is written in the ordering of the ROI after the evaluation
        # script applies its noise-ceiling filter. Recreate that exact local
        # index space before translating an R² index to a global voxel ID.
        r2_region_mask = region_mask & (
            noise_ceiling > args.r2_noise_ceiling_threshold
        )
        region_voxel_ids = np.flatnonzero(r2_region_mask)
        # region_mask = region_mask.unsqueeze(0).to(device)
        # Apply region_mask to best_weights
        # This will create a new dict of best_weights only for voxels where region_mask is True
        best_prf_idx_region = best_prf_idx[r2_region_mask]
        
        r2_voxels = np.squeeze(r2[region]['r2_voxels'])
        r2_voxels = np.atleast_1d(r2_voxels)
        if r2_voxels.size != region_voxel_ids.size:
            raise ValueError(
                f"R²/voxel index mismatch for {region}: r2.pkl contains "
                f"{r2_voxels.size} voxels, but ROI & "
                f"(noise_ceiling > {args.r2_noise_ceiling_threshold}) contains "
                f"{region_voxel_ids.size}. Set --r2_noise_ceiling_threshold "
                "to the value used to create r2.pkl."
            )
        if 'noise_ceiling' in r2[region]:
            stored_noise_ceiling = np.asarray(
                r2[region]['noise_ceiling']
            ).reshape(-1)
            selected_noise_ceiling = noise_ceiling[r2_region_mask]
            if (
                stored_noise_ceiling.shape != selected_noise_ceiling.shape
                or not np.allclose(
                    stored_noise_ceiling,
                    selected_noise_ceiling,
                    equal_nan=True,
                )
            ):
                raise ValueError(
                    f"Noise-ceiling values for {region} do not match the "
                    "filtered voxel ordering reconstructed from the ROI file. "
                    "Check that r2.pkl and --roi_path belong to the same "
                    "subject/model evaluation."
                )
        topk_indices, topk_values, voxel_ranks = select_voxels_by_rank(
            r2_voxels, args.k, args.voxel_rank_range
        )
        print(f"Voxel ranks {voxel_ranks[0]}-{voxel_ranks[-1]} r2 values "
              f"for {region}: {topk_values}")
        print(f"Corresponding indices: {topk_indices}")

        
        for i, (voxel_rank, idx) in enumerate(zip(voxel_ranks, topk_indices)):
            voxel_id = int(region_voxel_ids[idx])
            voxel_key = str(voxel_id)
            weights = best_weights[voxel_key][:-1]
            intercept = best_weights[voxel_key][-1]

            weights = torch.from_numpy(weights).unsqueeze(0).double().to(device)
            intercept = torch.tensor(intercept).unsqueeze(0).double().to(device)

            prf_idx = best_prf_idx_region[idx]
            if int(prf_idx) not in set(gen_metadata["prf_ids"]):
                raise ValueError(
                    f"Selected pRF ID {prf_idx} for voxel {voxel_id} is not "
                    "listed in the generation checkpoint metadata"
                )
            if args.enable_ranking == 'true' and int(prf_idx) not in set(rank_metadata["prf_ids"]):
                raise ValueError(
                    f"Generation-selected pRF ID {prf_idx} is unavailable in "
                    "the ranking model metadata"
                )
            gen_prf_kernels = build_prf_kernels(
                gen_layer_names, gen_layer_sizes, prf_idx, device
            )
            contrast_prf_layer = max(
                gen_layer_names,
                key=lambda layer_name: gen_layer_sizes[layer_name][-1],
            )
            contrast_prf_weight = gen_prf_kernels[contrast_prf_layer]
            contrast_prf_native_size = list(
                gen_layer_sizes[contrast_prf_layer][-2:]
            )
            if weights.shape[1] != int(gen_metadata["num_features"]):
                raise ValueError(
                    f"Generation weights for voxel {voxel_id} contain "
                    f"{weights.shape[1]} channels, but concatenated metadata "
                    f"requires {gen_metadata['num_features']}"
                )
            features_m = best_features_m[voxel_id].double()
            features_s = best_features_s[voxel_id].double()

            # Turn on if zscore is used
            # mean = channel_mean[prf_idx, channel_used]
            # std = channel_std[prf_idx, channel_used]
            # mean = torch.from_numpy(mean).double().to(device)
            # std = torch.from_numpy(std).double().to(device)

            def single_voxel_objective(image_input):
                # Aim: Maximize the predicted mean response for the region
                encoder_input = preprocess_encoder_input(
                    image_input, args.gen_model
                )
                image_features = brain_encoder(encoder_input)
                prf_features = pool_concatenated_features(
                    image_features, gen_layer_names, gen_prf_kernels
                ).squeeze(0)
                prf_features = (prf_features - features_m) / features_s

                voxel_response = weights @ prf_features + intercept
                
                return -voxel_response

            # def guider_objective(image_input):
            #     # Aim: Maximize the predicted mean response for the region
            #     image_features = feature_extractor(image_input)
            #     image_features = image_features['layer2'].float()
            #     image_features = torch.nn.functional.avg_pool2d(image_features, kernel_size=2, stride=2)
            #     image_features = image_features.view(image_features.size(0), -1)
            #     image_features = image_features/(image_features.norm(dim=-1, keepdim=True)+1e-10)
            #     # pred_response = brain_projector(image_features)
            #     pred_response = pred_response[region_mask]
            #     # pred_response = pred_response[r2_mask]
            #     return -torch.mean(pred_response)

            # def activate_deactivate_objective(image_input):
            #     # Aim: Maximize the predicted mean response for the region
            #     area_activate = region
            #     area_deactivate = list(set(regions) - set([region]))[0]

            #     area_activate_mask = torch.tensor(functional_dict[area_activate]).unsqueeze(0).bool().to("cuda")
            #     area_deactivate_mask = torch.tensor(functional_dict[area_deactivate]).unsqueeze(0).bool().to("cuda")
            #     image_featrues = CNN_backbone.encode_image(image_input)
            #     image_featrues = image_featrues/image_featrues.norm(dim=-1, keepdim=True)
            #     pred_voxels = brain_projector(image_featrues.float())
            #     activate_response = pred_voxels[area_activate_mask]
            #     deactivate_response = pred_voxels[area_deactivate_mask]
            #     return -torch.mean(activate_response) + torch.mean(deactivate_response)

            num_steps = args.num_steps
            brain_guidance_scale = args.brain_guidance_scale
            sag_scale = args.sag_scale
            model_folder = (
                f"gen-{path_component(args.gen_model)}-concat"
                + (f"__rank-{path_component(args.rank_model)}-concat" if args.enable_ranking == 'true' else "")
            )
            if args.contrast_constraint_full_image == 'none':
                contrast_folder = "none"
            elif args.contrast_constraint_method == 'max':
                contrast_folder = f"max-rms{args.max_rms_contrast:.2f}"
            elif args.contrast_constraint_method == 'range':
                contrast_folder = (
                    f"range-rms{args.min_rms_contrast:.2f}"
                    f"-{args.max_rms_contrast:.2f}"
                )
            else:
                contrast_folder = f"match-rms{args.target_rms_contrast:.2f}"
            luma_full = args.luminance_constraint_full_image
            luma_prf = args.luminance_constraint_prf
            if luma_full != "none":
                luma_full += f"-{args.target_luminance_full_image:g}"
            if luma_prf != "none":
                luma_prf += f"-{args.target_luminance_prf:g}"
            settings_folder = (
                f"split-{args.split_id:02d}"
                f"__steps-{num_steps}"
                f"__scale-{brain_guidance_scale:g}"
                f"__sag-{sag_scale:g}"
                f"__contrast-{contrast_folder}"
                f"__full-{args.contrast_constraint_full_image}"
                f"__prf-{args.contrast_constraint_prf}"
                f"__luma-full-{luma_full}__luma-prf-{luma_prf}"
                f"__seeds-{args.num_seeds}"
                f"__keep-{args.ranking_top_n if args.enable_ranking == 'true' else 'all'}"
            )
            # Method-specific branch also separates old, unlabelled experiment paths.
            # Keep the already-long settings component below filesystem name limits.
            settings_folder = os.path.join(
                'ranking-enabled' if args.enable_ranking == 'true' else 'ranking-disabled',
                f"method-{args.luma_contrast_method}", settings_folder)
            experiment_folder = os.path.join(
                args.save_root,
                f"sub-{args.subject_id:02d}",
                model_folder,
                settings_folder,
                f"seed-{args.seed_mode}-{seed_plan['plan_id']}",
            )
            region_folder = os.path.join(
                experiment_folder,
                f"roi-{path_component(region)}",
            )
            voxel_folder = os.path.join(
                region_folder,
                f"voxel-rank-{voxel_rank:02d}__id-{voxel_id:06d}",
            )
            images_folder = os.path.join(voxel_folder, "images") if save_constrained_images else None
            if images_folder is not None:
                os.makedirs(images_folder, exist_ok=True)

            # Pre-projection decodes sit beside "images" in the voxel folder.
            unconstrained_images_folder = None
            if save_unconstrained_images:
                unconstrained_images_folder = os.path.join(
                    voxel_folder, "unconstrained_images"
                )
                os.makedirs(unconstrained_images_folder, exist_ok=True)

            config_file = os.path.join(region_folder, "run_config.json")
            if not os.path.exists(config_file):
                with open(config_file, "w") as config_handle:
                    json.dump(
                        {
                            "subject": int(args.subject_id),
                            "ranking_enabled": args.enable_ranking == 'true',
                            "image_saving": image_saving_metadata,
                            "seed_plan": seed_plan,
                            "split": int(args.split_id),
                            "generation_model": {
                                "name": args.gen_model,
                                "layers": list(gen_layer_names),
                                "concatenated": True,
                            },
                            "ranking_model": {
                                "name": args.rank_model,
                                "layers": list(rank_layer_names),
                                "concatenated": True,
                            } if args.enable_ranking == 'true' else None,
                            "generation": {
                                "num_steps": int(num_steps),
                                "brain_guidance_scale": float(brain_guidance_scale),
                                "sag_scale": float(sag_scale),
                                "num_seeds": int(args.num_seeds),
                                "top_k_voxels": int(len(voxel_ranks)),
                                "voxel_rank_start": int(voxel_ranks[0]),
                                "voxel_rank_end": int(voxel_ranks[-1]),
                                "voxel_count": int(len(voxel_ranks)),
                                "ranking_top_n": args.ranking_top_n if args.enable_ranking == 'true' else None,
                                "max_generation_attempts": (
                                    int(args.max_generation_attempts)
                                    if args.max_generation_attempts is not None
                                    else None
                                ),
                                "contrast_validation_tolerance_byte": float(
                                    args.contrast_validation_tolerance
                                ),
                            },
                            "luminance": luminance_metadata,
                            "constraint_projection": constraint_method_metadata,
                            "contrast": {
                                "method": args.contrast_constraint_method,
                                "full_image": (
                                    args.contrast_constraint_full_image
                                ),
                                "prf": args.contrast_constraint_prf,
                                "acceptance_domain": acceptance_domain,
                                "acceptance_tolerance_byte": float(
                                    args.contrast_validation_tolerance
                                ),
                                "image_save_mode": args.image_save_mode,
                                "constrained_image_saved": save_constrained_images,
                                "unconstrained_image_saved": bool(
                                    save_unconstrained_images
                                ),
                                "unconstrained_image_content": (
                                    unconstrained_image_content
                                ),
                                "statistics_json": (
                                    os.path.abspath(contrast_stats_path)
                                    if contrast_stats_path is not None
                                    else None
                                ),
                                "min_percentile": float(
                                    args.min_contrast_percentile
                                ),
                                "max_percentile": float(
                                    args.max_contrast_percentile
                                ),
                                "target_percentile": float(
                                    args.target_contrast_percentile
                                ),
                                "min_rms_byte": (
                                    float(args.min_rms_contrast)
                                    if args.min_rms_contrast is not None
                                    else None
                                ),
                                "max_rms_byte": (
                                    float(args.max_rms_contrast)
                                    if args.max_rms_contrast is not None
                                    else None
                                ),
                                "target_rms_byte": (
                                    float(args.target_rms_contrast)
                                    if args.target_rms_contrast is not None
                                    else None
                                ),
                                "dual_scope": {
                                    "prf_source": (
                                        "generation_model_selected_prf"
                                    ),
                                    "prf_layer_selection": (
                                        "highest_spatial_resolution_"
                                        "concatenated_generation_layer"
                                    ),
                                    "prf_layer": contrast_prf_layer,
                                    "prf_native_size": contrast_prf_native_size,
                                    "resize_to_decoded_image": "bilinear",
                                    "normalize_weight_sum": 1.0,
                                    "solver_max_nfev": int(
                                        args.dual_scope_solver_max_nfev
                                    ),
                                    "every_step_iterations": int(
                                        args.dual_scope_every_step_iterations
                                    ),
                                    "parameter_limit": float(
                                        args.dual_scope_parameter_limit
                                    ),
                                    "adjustment_envelope_power": float(
                                        args.dual_scope_envelope_power
                                    ),
                                },
                            },
                            "region": region,
                            "regions": [region],
                            "invocation_regions": list(args.roi),
                        },
                        config_handle,
                        indent=2,
                        allow_nan=False,
                    )
            pipe.brain_tweak = single_voxel_objective

            image_records = []
            attempted_seeds = []
            seed_generation_times = []
            seed_record_path = os.path.join(voxel_folder, 'seeds.json')

            def checkpoint_seeds(status='in_progress'):
                save_seed_record(seed_record_path, seed_plan, attempted_seeds,
                                 [item['seed'] for item in image_records], status,
                                 generation_times=seed_generation_times,
                                 ranking_enabled=args.enable_ranking == 'true')
                voxel_timing = {
                    'subject': int(ss), 'region': region, 'voxel_id': int(voxel_id),
                    'seed_record': os.path.abspath(seed_record_path), 'status': status,
                    'hardware': hardware_metadata,
                    **summarize_generation_times(seed_generation_times),
                }
                generation_timing['voxels'][os.path.abspath(seed_record_path)] = voxel_timing
                merge_roi_generation_timing(
                    os.path.join(region_folder, 'generation_timing.json'), region, [voxel_timing])
                if args.generation_timing_file is not None:
                    mirror_path = os.path.join(
                        os.path.dirname(os.path.abspath(args.generation_timing_file)),
                        os.path.basename(region_folder), os.path.basename(args.generation_timing_file))
                    merge_roi_generation_timing(mirror_path, region, [voxel_timing])

            checkpoint_seeds()
            rejected_candidates = {
                "nonfinite_pixels": 0,
                "invalid_shape": 0,
                "out_of_range_pixels": 0,
                "degenerate_contrast": 0,
                "contrast_mismatch": 0,
                "global_contrast_mismatch": 0,
                "prf_contrast_mismatch": 0,
                "dual_scope_solver_failure": 0,
                "nonfinite_dual_scope_parameters": 0,
                "nonfinite_gen_score": 0,
                "nonfinite_rank_score": 0,
                "global_luminance_mismatch": 0,
                "prf_luminance_mismatch": 0,
                "joint_solver_failure": 0,
                "nonfinite_joint_parameters": 0,
                "luma_rgb_clip_solver_failure": 0,
            }
            max_generation_attempts = seed_plan['effective_max_attempts']
            contrast_tolerance = args.contrast_validation_tolerance / 255.0

            def candidate_rejection_reason(image_array):
                if image_array.ndim != 3 or image_array.shape[-1] != 3:
                    return "invalid_shape", None
                if not np.isfinite(image_array).all():
                    return "nonfinite_pixels", None
                if image_array.min() < -1e-6 or image_array.max() > 1.0 + 1e-6:
                    return "out_of_range_pixels", None

                float_tensor = torch.from_numpy(
                    np.ascontiguousarray(image_array)
                ).permute(2, 0, 1)[None].to(dtype=torch.float64)
                global_mean, global_rms = global_luma_mean_and_rms(float_tensor)
                global_mean = float(global_mean[0])
                global_rms = float(global_rms[0])
                prf_rms = None
                validation_weight = None
                prf_mean = None
                if (args.contrast_constraint_prf != "none"
                        or args.luminance_constraint_prf != "none"):
                    validation_weight = normalized_prf_weight(
                        contrast_prf_weight.detach().cpu(), float_tensor
                    )
                    prf_mean_value, prf_value = weighted_luma_mean_and_rms(
                        float_tensor, validation_weight
                    )
                    prf_rms = float(prf_value[0])
                    prf_mean = float(prf_mean_value[0])

                quantized_array = (
                    np.rint(np.clip(image_array, 0, 1) * 255.0) / 255.0
                )
                quantized_tensor = torch.from_numpy(
                    np.ascontiguousarray(quantized_array)
                ).permute(2, 0, 1)[None].to(dtype=torch.float64)
                quantized_mean, quantized_global = global_luma_mean_and_rms(
                    quantized_tensor
                )
                quantized_global = float(quantized_global[0])
                quantized_prf = None
                quantized_prf_mean = None
                if validation_weight is not None:
                    quantized_prf_mean_value, quantized_prf_value = weighted_luma_mean_and_rms(
                        quantized_tensor, validation_weight
                    )
                    quantized_prf = float(quantized_prf_value[0])
                    quantized_prf_mean = float(quantized_prf_mean_value[0])

                metrics = {
                    "float_global_mean_byte": global_mean * 255,
                    "float_prf_mean_byte": prf_mean * 255 if prf_mean is not None else None,
                    "png_global_mean_byte": float(quantized_mean[0]) * 255,
                    "png_prf_mean_byte": (quantized_prf_mean * 255
                                          if quantized_prf_mean is not None else None),
                    "float_global_rms_byte": global_rms * 255.0,
                    "float_prf_rms_byte": (
                        prf_rms * 255.0 if prf_rms is not None else None
                    ),
                    "png_global_rms_byte": quantized_global * 255.0,
                    "png_prf_rms_byte": (
                        quantized_prf * 255.0
                        if quantized_prf is not None
                        else None
                    ),
                }
                if args.image_save_mode == 'unconstrained_only':
                    # The final projection was deliberately skipped. Measure
                    # targets for reporting only; do not reject their mismatch.
                    metrics['acceptance_domain'] = 'unadjusted_decode_pixel_validity_only'
                    metrics['final_target_validation_applied'] = False
                    return None, metrics
                if (args.luma_contrast_method == 'luma_rgb_clip'
                        and args.contrast_constraint_full_image != 'none'):
                    # The projector already enforced all active pre-RGB luma targets.
                    # Final clipped RGB measurements are diagnostics, not acceptance
                    # criteria for this method. Shape/range/finite checks still apply.
                    metrics['acceptance_domain'] = acceptance_domain
                    metrics['postclip_mismatch_policy'] = 'record_only'
                    return None, metrics
                if (
                    not np.isfinite(global_rms)
                    or (global_rms <= 1e-8 and (
                        args.contrast_constraint_full_image != "none"
                        or args.luminance_constraint_full_image == "none"
                    ))
                    or (
                        args.contrast_constraint_prf != "none"
                        and (not np.isfinite(prf_rms) or prf_rms <= 1e-8)
                    )
                ):
                    return "degenerate_contrast", metrics

                for scope, stage, measured, target in (
                    ("global", args.luminance_constraint_full_image, global_mean,
                     luminance_options['target_luminance_full_image']),
                    ("prf", args.luminance_constraint_prf, prf_mean,
                     luminance_options['target_luminance_prf']),
                ):
                    if stage != "none":
                        error_byte = abs(measured - target) * 255
                        metrics[f"{scope}_mean_error_byte"] = error_byte
                        if error_byte > args.luminance_validation_tolerance:
                            return f"{scope}_luminance_mismatch", metrics
                if args.contrast_constraint_full_image == "none":
                    valid_contrast = True
                elif args.contrast_constraint_method == "match":
                    target = args.target_rms_contrast / 255.0
                    if abs(global_rms - target) > contrast_tolerance:
                        return "global_contrast_mismatch", metrics
                    if (
                        args.contrast_constraint_prf != "none"
                        and abs(prf_rms - target) > contrast_tolerance
                    ):
                        return "prf_contrast_mismatch", metrics
                elif args.contrast_constraint_method == "max":
                    valid_contrast = (
                        global_rms
                        <= args.max_rms_contrast / 255.0 + contrast_tolerance
                    )
                elif args.contrast_constraint_method == "range":
                    valid_contrast = (
                        global_rms
                        >= args.min_rms_contrast / 255.0 - contrast_tolerance
                        and global_rms
                        <= args.max_rms_contrast / 255.0 + contrast_tolerance
                    )
                else:
                    valid_contrast = True
                if (
                    args.contrast_constraint_method != "match"
                    and not valid_contrast
                ):
                    return "contrast_mismatch", metrics
                return None, metrics

            def unconstrained_candidate_image(raw_image):
                """Return the pre-projection decode and its global RMS.

                The pipeline records one before the last adjustment whenever
                any mean/RMS constraint is enabled (every_step or final).
                This decode can come from an already constrained trajectory.
                """
                if not save_unconstrained_images or raw_image is None:
                    return None, None
                raw_tensor = torch.from_numpy(
                    np.ascontiguousarray(
                        np.asarray(raw_image[0], dtype=np.float32)
                    )
                ).permute(2, 0, 1)
                _, raw_rms = global_luma_mean_and_rms(
                    raw_tensor[None].to(dtype=torch.float64)
                )
                return raw_tensor, float(raw_rms[0]) * 255.0

            def generate_pixel_valid_candidates(number_needed):
                generated = []
                while (
                    len(generated) < number_needed
                    and len(attempted_seeds) < max_generation_attempts
                ):
                    seed = seed_plan['seed_set'][len(attempted_seeds)]
                    attempt_number = len(attempted_seeds) + 1
                    attempted_seeds.append(seed)
                    checkpoint_seeds()
                    print(
                        f"Starting {attempt_number:05d} "
                        f"(accepted {len(image_records) + len(generated)}/"
                        f"{args.num_seeds})"
                    )
                    g = torch.Generator(device=device).manual_seed(int(seed))
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    generation_start = time.perf_counter()
                    pipeline_returned = False
                    try:
                        try:
                            image = pipe(
                                "",
                                sag_scale=sag_scale,
                                guidance_scale=0.0,
                                num_inference_steps=num_steps,
                                generator=g,
                                clip_guidance_scale=brain_guidance_scale,
                                contrast_constraint_method=(
                                    args.contrast_constraint_method
                                ),
                                contrast_constraint_full_image=(
                                    args.contrast_constraint_full_image
                                ),
                                contrast_constraint_prf=(
                                    args.contrast_constraint_prf
                                ),
                                contrast_prf_weight=contrast_prf_weight,
                                min_rms_contrast=(
                                    args.min_rms_contrast / 255.0
                                    if args.min_rms_contrast is not None
                                    else None
                                ),
                                max_rms_contrast=(
                                    args.max_rms_contrast / 255.0
                                    if args.max_rms_contrast is not None
                                    else None
                                ),
                                target_rms_contrast=(
                                    args.target_rms_contrast / 255.0
                                    if args.target_rms_contrast is not None
                                    else None
                                ),
                                dual_scope_tolerance=contrast_tolerance,
                                dual_scope_solver_max_nfev=(
                                    args.dual_scope_solver_max_nfev
                                ),
                                dual_scope_every_step_iterations=(
                                    args.dual_scope_every_step_iterations
                                ),
                                dual_scope_parameter_limit=(
                                    args.dual_scope_parameter_limit
                                ),
                                dual_scope_envelope_power=(
                                    args.dual_scope_envelope_power
                                ),
                                output_type="np",
                                **luminance_options,
                            )
                            pipeline_returned = True
                        finally:
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            elapsed = time.perf_counter() - generation_start
                            generation_timing["seconds"] += elapsed
                            generation_timing["calls"] += 1
                            seed_generation_times.append({
                                'seed': int(seed), 'attempt_number': attempt_number,
                                'generation_seconds': elapsed,
                                'pipeline_returned': pipeline_returned,
                            })
                            checkpoint_seeds()
                    except DualScopeContrastConvergenceError as error:
                        rejected_candidates[error.reason] += 1
                        print(
                            f"Rejected seed {seed}: {error.reason} ({error})"
                        )
                        continue
                    image_array = np.asarray(image.images[0], dtype=np.float32)
                    raw_tensor, raw_rms_contrast_byte = (
                        unconstrained_candidate_image(
                            pipe.last_unconstrained_image
                        )
                    )
                    reason, contrast_metrics = candidate_rejection_reason(
                        image_array
                    )
                    if reason is not None:
                        rejected_candidates[reason] += 1
                        print(
                            f"Rejected seed {seed}: {reason}"
                            + (
                                " (global RMS="
                                f"{contrast_metrics['float_global_rms_byte']:.6f}"
                                " byte)"
                                if contrast_metrics is not None
                                else ""
                            )
                        )
                        continue
                    generated.append({
                        "seed": int(seed),
                        "tensor": torch.from_numpy(
                            np.ascontiguousarray(image_array)
                        ).permute(2, 0, 1),
                        "raw_tensor": raw_tensor,
                        "raw_rms_contrast_byte": raw_rms_contrast_byte,
                        "rms_contrast_byte": contrast_metrics[
                            "float_global_rms_byte"
                        ],
                        "contrast_diagnostics": contrast_metrics,
                        "dual_scope_solver": (
                            dict(pipe.last_contrast_diagnostics)
                            if pipe.last_contrast_diagnostics is not None
                            else None
                        ),
                        "joint_luma_rms_solver": pipe.last_joint_diagnostics,
                        "luma_rgb_clip_diagnostics": (
                            pipe.last_joint_diagnostics
                            if args.luma_contrast_method == 'luma_rgb_clip' else None),
                        "guidance_constraint_diagnostics": dict(
                            pipe.last_guidance_constraint_diagnostics),
                    })
                return generated

            pending_records = generate_pixel_valid_candidates(args.num_seeds)
            if len(pending_records) < args.num_seeds:
                print(
                    "Generation attempt limit reached before collecting the "
                    "requested number of pixel-valid candidates; processing the "
                    "accepted candidates that are available. "
                    f"Only generated {len(pending_records)} pixel-valid "
                    f"candidates for {region} voxel {voxel_id} after "
                    f"{len(attempted_seeds)} attempts "
                    f"(limit {max_generation_attempts}). "
                    f"Rejections: {rejected_candidates}"
                )


            if args.enable_ranking == 'false':
                image_records.extend(pending_records)
            if args.enable_ranking == 'true':
                ############## Train the ranking model based on voxel id #########################################
                features_model_folder = os.path.join(
                    args.feature_root,
                    f"S{args.subject_id}",
                    args.rank_model,
                )
                print(
                    f"Loading concatenated ranking features from: "
                    f"{features_model_folder} ({', '.join(rank_layer_names)})"
                )

                (
                    rank_weights,
                    rank_channel_kept,
                    best_lambda_idx,
                    best_nest_loss,
                    rank_lambdas,
                    rank_features_s,
                    rank_features_m,
                ) = rank_model_fitting(
                    voxel_id,
                    prf_idx,
                    voxel_data,
                    train_ids,
                    val_ids,
                    nest_ids,
                    features_model_folder,
                    rank_layer_names,
                    device,
                )

                rank_weights = rank_weights[:, 0]
                rank_intercept = rank_weights[-1].double()
                rank_weights = rank_weights[:-1].double()
                rank_features_s = torch.as_tensor(
                    rank_features_s, device=device, dtype=torch.double
                ).squeeze(0)
                rank_features_m = torch.as_tensor(
                    rank_features_m, device=device, dtype=torch.double
                ).squeeze(0)
                rank_prf_kernels = build_prf_kernels(
                    rank_layer_names, rank_layer_sizes, prf_idx, device
                )
                if rank_weights.numel() != int(rank_metadata["num_features"]):
                    raise ValueError(
                        f"Online ranking fit has {rank_weights.numel()} weights, "
                        f"but metadata expects {rank_metadata['num_features']}"
                    )

                ranking_model_folder = os.path.join(
                    voxel_folder, "ranking_model"
                )
                ranking_model_metadata_path, ranking_model_parameters_path = (
                    save_online_ranking_model(
                        ranking_model_folder,
                        subject_id=ss,
                        region=region,
                        voxel_id=voxel_id,
                        voxel_index_in_region=idx,
                        voxel_rank=voxel_rank,
                        voxel_r2=topk_values[i],
                        generation_model_name=args.gen_model,
                        model_name=args.rank_model,
                        split_id=args.split_id,
                        metadata=rank_metadata,
                        layer_sizes=rank_layer_sizes,
                        prf_idx=prf_idx,
                        prf_grid_name=prf_grid_name,
                        prf_params=prf_grid_params[int(prf_idx)],
                        weights=rank_weights.detach().cpu().numpy(),
                        intercept=rank_intercept.detach().cpu().numpy(),
                        feature_mean=rank_features_m.detach().cpu().numpy(),
                        feature_std=rank_features_s.detach().cpu().numpy(),
                        channel_indices=rank_channel_kept,
                        lambda_candidates=rank_lambdas,
                        best_lambda_index=best_lambda_idx,
                        nested_sse=best_nest_loss,
                        train_ids=train_ids,
                        val_ids=val_ids,
                        nest_ids=nest_ids,
                        features_model_folder=features_model_folder,
                        ranking_model_path=rank_model_path,
                        data_splits_path=data_splits_path,
                    )
                )
                print(
                    f"Saved online ranking model for {region} voxel {voxel_id}: "
                    f"{ranking_model_metadata_path}"
                )

                while (
                    len(image_records) < args.num_seeds
                    and len(pending_records) > 0
                ):
                    gen_scores = score_generated_images_for_voxel(
                        image_records=pending_records,
                        layer_names=gen_layer_names,
                        prf_kernels=gen_prf_kernels,
                        voxel_weights=weights,
                        voxel_intercept=intercept,
                        features_s=features_s,
                        features_m=features_m,
                        model_name=args.gen_model,
                        batch_size=8,
                        encoder=brain_encoder,
                    )
                    pred_scores = score_generated_images_for_voxel(
                        image_records=pending_records,
                        layer_names=rank_layer_names,
                        prf_kernels=rank_prf_kernels,
                        voxel_weights=rank_weights,
                        voxel_intercept=rank_intercept,
                        features_s=rank_features_s,
                        features_m=rank_features_m,
                        model_name=args.rank_model,
                        batch_size=8,
                    )
                    for item, gen_score, rank_score in zip(
                        pending_records, gen_scores, pred_scores
                    ):
                        gen_score = float(gen_score)
                        rank_score = float(rank_score)
                        score_is_valid = True
                        if not np.isfinite(gen_score):
                            rejected_candidates["nonfinite_gen_score"] += 1
                            score_is_valid = False
                        if not np.isfinite(rank_score):
                            rejected_candidates["nonfinite_rank_score"] += 1
                            score_is_valid = False
                        if score_is_valid:
                            item["gen_score"] = gen_score
                            item["rank_score"] = rank_score
                            image_records.append(item)
                        else:
                            print(
                                f"Rejected seed {item['seed']}: non-finite score "
                                f"(gen={gen_score}, rank={rank_score})"
                            )
                            item["tensor"] = None
                            item["raw_tensor"] = None

                    checkpoint_seeds()
                    if len(image_records) >= args.num_seeds:
                        break
                    missing = args.num_seeds - len(image_records)
                    pending_records = generate_pixel_valid_candidates(missing)
                    if not pending_records:
                        print(
                            "Generation attempt limit reached before collecting "
                            "the requested number of fully valid candidates; "
                            "ranking the accepted candidates that are available. "
                            f"Only generated {len(image_records)} fully valid "
                            f"candidates for {region} voxel {voxel_id} after "
                            f"{len(attempted_seeds)} attempts "
                            f"(limit {max_generation_attempts}). "
                            f"Rejections: {rejected_candidates}"
                        )
                        break

            checkpoint_seeds('complete' if len(image_records) >= args.num_seeds
                             else 'partial_seed_budget_exhausted')
            ranked_indices = sorted(
                range(len(image_records)),
                key=lambda j: image_records[j]["rank_score"],
                reverse=True,
            ) if args.enable_ranking == 'true' else list(range(len(image_records)))
            generation_ranked_indices = sorted(
                range(len(image_records)),
                key=lambda j: image_records[j]["gen_score"],
                reverse=True,
            ) if args.enable_ranking == 'true' else []
            generation_ranks = {
                img_idx: rank for rank, img_idx in enumerate(generation_ranked_indices, start=1)
            }
            ranking_model_ranks = {
                img_idx: rank for rank, img_idx in enumerate(ranked_indices, start=1)
            } if args.enable_ranking == 'true' else {}

            requested_top_n = args.ranking_top_n if args.enable_ranking == 'true' else 'all'
            top_n = (len(ranked_indices) if requested_top_n == "all"
                     else min(requested_top_n, len(ranked_indices)))
            top_ranked = []
            selected_indices = set(ranked_indices[:top_n])
            candidate_rankings = [
                {
                    'seed': item['seed'],
                    'gen_rank': generation_ranks[img_idx],
                    'rank_model_rank': ranking_model_ranks[img_idx],
                    'gen_score': item['gen_score'],
                    'rank_score': item['rank_score'],
                    'saved': img_idx in selected_indices,
                }
                for img_idx, item in enumerate(image_records)
            ] if args.enable_ranking == 'true' else []
            for rank_idx, img_idx in enumerate(ranked_indices[:top_n], start=1):
                item = image_records[img_idx]
                ranked_image_name = (
                    f"seed-{item['seed']:010d}.png"
                )
                ranked_image_path = None
                if images_folder is not None:
                    ranked_image_path = os.path.join(images_folder, ranked_image_name)
                    image_array = item["tensor"].permute(1, 2, 0).numpy()
                    pil_image = Image.fromarray(
                        np.rint(np.clip(image_array, 0, 1) * 255).astype(np.uint8),
                        mode="RGB",
                    )
                    pil_image.save(ranked_image_path, format="PNG", compress_level=6)
                    pil_image.close()
                raw_image_path = None
                if (
                    item["raw_tensor"] is not None
                    and unconstrained_images_folder is not None
                ):
                    # Same basename as the constrained image so the pair can
                    # be matched across the two folders by filename alone.
                    raw_image_path = os.path.join(
                        unconstrained_images_folder, ranked_image_name
                    )
                    raw_array = item["raw_tensor"].permute(1, 2, 0).numpy()
                    raw_pil_image = Image.fromarray(
                        np.rint(
                            np.clip(raw_array, 0, 1) * 255
                        ).astype(np.uint8),
                        mode="RGB",
                    )
                    raw_pil_image.save(
                        raw_image_path, format="PNG", compress_level=6
                    )
                    raw_pil_image.close()
                top_ranked.append({
                    "seed": item["seed"],
                    **({
                        "rank": rank_idx,
                        "gen_rank": generation_ranks[img_idx],
                        "rank_model_rank": ranking_model_ranks[img_idx],
                        "gen_score": item["gen_score"],
                        "rank_score": item["rank_score"],
                    } if args.enable_ranking == 'true' else {}),
                    "rms_contrast_byte": item["rms_contrast_byte"],
                    "contrast_diagnostics": item["contrast_diagnostics"],
                    "dual_scope_solver": item["dual_scope_solver"],
                    "joint_luma_rms_solver": item["joint_luma_rms_solver"],
                    "luma_rgb_clip_diagnostics": item.get("luma_rgb_clip_diagnostics"),
                    "guidance_constraint_diagnostics": item["guidance_constraint_diagnostics"],
                    "image_file": os.path.join("images", ranked_image_name) if ranked_image_path else None,
                    "image_path": os.path.abspath(ranked_image_path) if ranked_image_path else None,
                    "raw_rms_contrast_byte": item["raw_rms_contrast_byte"],
                    "raw_image_file": (
                        os.path.join(
                            "unconstrained_images", ranked_image_name
                        )
                        if raw_image_path is not None
                        else None
                    ),
                    "raw_image_path": (
                        os.path.abspath(raw_image_path)
                        if raw_image_path is not None
                        else None
                    ),
                })

            ranking_file = os.path.join(
                voxel_folder, "ranking.json" if args.enable_ranking == 'true' else "generation.json")
            with open(ranking_file, "w") as f:
                json.dump(
                    generation_summary_for_mode({
                        "schema_version": 15,
                        "image_saving": image_saving_metadata,
                        "generation_model": {"name": args.gen_model, "layers": list(gen_layer_names)},
                        "split": int(args.split_id),
                        "ranking_definitions": {
                            "gen_rank": {"model": args.gen_model, "score": "gen_score"},
                            "rank_model_rank": {"model": args.rank_model, "score": "rank_score"},
                            "scope": "all_fully_valid_candidates_for_this_voxel",
                            "order": "descending_score_1_is_best",
                            "ties": "accepted_candidate_order",
                            "selection": "rank_model_rank",
                            "score_image_version": (
                                "unadjusted_final_decode" if args.image_save_mode == 'unconstrained_only'
                                else "pipeline_output_after_final_adjustment_if_enabled"),
                            "legacy_rank_field": "alias_of_rank_model_rank",
                        },
                        "candidate_rankings": candidate_rankings,
                        "region": region,
                        "subject": int(ss),
                        "voxel_id": int(voxel_id),
                        "voxel_index_in_region": int(idx),
                        "voxel_rank": int(voxel_rank),
                        "voxel_rank_in_topk": int(voxel_rank),
                        "voxel_r2": float(topk_values[i]),
                        "prf_idx": int(prf_idx),
                        "prf_grid_name": prf_grid_name,
                        "prf_x": float(prf_grid_params[int(prf_idx), 0]),
                        "prf_y": float(prf_grid_params[int(prf_idx), 1]),
                        "prf_sigma": float(prf_grid_params[int(prf_idx), 2]),
                        "contrast_prf_layer": contrast_prf_layer,
                        "contrast_prf_native_size": contrast_prf_native_size,
                        "generation_status": (
                            "complete"
                            if len(image_records) >= args.num_seeds
                            else "partial_max_attempts_reached"
                        ),
                        "requested_candidates": int(args.num_seeds),
                        "accepted_candidates": int(len(image_records)),
                        "candidate_shortfall": int(
                            max(0, args.num_seeds - len(image_records))
                        ),
                        # Retained for compatibility with existing readers.
                        "num_candidates": int(len(image_records)),
                        "generation_attempts": int(len(attempted_seeds)),
                        "seed_record": "seeds.json",
                        "generation_timing": summarize_generation_times(seed_generation_times),
                        "seed_mode": args.seed_mode,
                        "seed_plan_id": seed_plan['plan_id'],
                        "max_generation_attempts": int(max_generation_attempts),
                        "max_generation_attempts_reached": bool(
                            len(attempted_seeds) >= max_generation_attempts
                        ),
                        "contrast_validation_tolerance_byte": float(
                            args.contrast_validation_tolerance
                        ),
                        "contrast_constraint_method": (
                            args.contrast_constraint_method
                        ),
                        "contrast_constraint_full_image": (
                            args.contrast_constraint_full_image
                        ),
                        "contrast_constraint_prf": (
                            args.contrast_constraint_prf
                        ),
                        "contrast_acceptance_domain": acceptance_domain,
                        "constraint_projection": constraint_method_metadata,
                        "luminance": luminance_metadata,
                        "image_save_mode": args.image_save_mode,
                        "constrained_image_saved": save_constrained_images,
                        "unconstrained_image_saved": bool(
                            save_unconstrained_images
                        ),
                        "unconstrained_image_content": (
                            unconstrained_image_content
                        ),
                        "unconstrained_images_folder": (
                            os.path.abspath(unconstrained_images_folder)
                            if unconstrained_images_folder is not None
                            else None
                        ),
                        "contrast_target_rms_byte": (
                            float(args.target_rms_contrast)
                            if args.target_rms_contrast is not None
                            else None
                        ),
                        "dual_scope_settings": {
                            "solver_max_nfev": int(
                                args.dual_scope_solver_max_nfev
                            ),
                            "every_step_iterations": int(
                                args.dual_scope_every_step_iterations
                            ),
                            "parameter_limit": float(
                                args.dual_scope_parameter_limit
                            ),
                            "adjustment_envelope_power": float(
                                args.dual_scope_envelope_power
                            ),
                            "prf_resize": "bilinear",
                            "prf_weight_sum": 1.0,
                        },
                        "rejected_candidates": rejected_candidates,
                        "ranking_top_n_requested": requested_top_n,
                        "saved_image_count": int(top_n),
                        "ranking_top_n_returned": int(top_n),
                        # Retained for compatibility with existing readers.
                        "ranking_top_n": int(top_n),
                        "online_ranking_model": {
                            "metadata_file": os.path.relpath(
                                ranking_model_metadata_path, voxel_folder
                            ),
                            "parameters_file": os.path.relpath(
                                ranking_model_parameters_path, voxel_folder
                            ),
                        } if args.enable_ranking == 'true' else None,
                        "results": top_ranked,
                    }, args.enable_ranking == 'true'),
                    f,
                    indent=2,
                    allow_nan=False,
                )

            # Release temporary image tensors after each voxel.
            for item in image_records:
                item["tensor"] = None
                item["raw_tensor"] = None
            del ranked_indices
            del image_records
            gc.collect()
            torch.cuda.empty_cache()

            if len(top_ranked) > 0 and args.enable_ranking == 'true':
                print(
                    f"Voxel ranking complete for {region} voxel {voxel_id}: "
                    f"best_seed={top_ranked[0]['seed']}, "
                    f"best_score={top_ranked[0]['rank_score']:.6f}, "
                    f"kept_top_n={top_n}, "
                    f"saved={ranking_file}"
                )
            elif args.enable_ranking == 'false':
                print(f"Generation complete for {region} voxel {voxel_id}: saved={top_n}; metadata={ranking_file}")


    if args.generation_timing_file is not None:
        # Each ROI mirror was already checkpointed; do not overwrite with process totals.
        print(f"Generation-only timing mirrors saved in ROI subfolders beside: {args.generation_timing_file}")



if __name__ == "__main__":
    main()
