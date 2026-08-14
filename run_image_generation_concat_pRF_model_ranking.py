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
import pickle
import gc

import nibabel as nib
import os

import argparse
from torchvision.models.feature_extraction import create_feature_extractor
from prf_utils import get_prf_stack
import random
from base64 import b64encode
random.seed(a=b64encode(os.urandom(5)).decode('utf-8'))
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
    roi_keys = ['roi_labels_retino', 'roi_labels_kastner', 'roi_labels_face', 'roi_labels_place', 'roi_labels_body']
    roi_names = ['ret_prf_roi_names', 'kastner_atlas_roi_names', 'floc_face_roi_names', 'floc_place_roi_names', 'floc_body_roi_names']
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


def rank_model_fitting(voxel_id, best_prf_idx, voxel_data, train_ids, val_ids, nest_ids, features_model_folder, layer_names, device):
    """
    Return:
    - best_prf_idx: list of length of num_voxels, the index of the best pRF for each voxel
    - best_lambda: list of length of num_voxels, the best lambda for each voxel for the best pRF
    - best_loss: list of length of num_voxels, the best r2 for each voxel for the best pRF
    - best_weights: array of length of [num_voxels, num_features + 1], the best weights for each voxel for the best pRF, with the intercept
    """

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


    return best_weights, channel_kept, best_lambda_idx, features_s, features_m


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
    parser.add_argument('--split_id', type=int, choices=[1, 2], default=1)
    parser.add_argument('--feature_root', type=str, default="/user_data/junruz/prf_features")
    parser.add_argument('--model_root', type=str, default="/user_data/junruz/prf_models/concat/split_1_zscore")
    parser.add_argument('--roi', nargs='+', default=["V1"], type=str)
    parser.add_argument('--k', type=int, default=3, help="Top k voxels to keep in the region")
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
        '--contrast_constraint_method',
        choices=['none', 'max', 'range', 'match'],
        default='range',
        help=(
            "Contrast projection to use: 'none' disables it, 'max' caps RMS "
            "contrast, 'range' keeps it between lower/upper bounds, and "
            "'match' projects every image to one target RMS contrast."
        ),
    )
    parser.add_argument(
        '--contrast_stats_json',
        '--contrast_stats_path',
        dest='contrast_stats_json',
        type=str,
        default=None,
        help=(
            "NSD contrast-statistics JSON. Defaults to "
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
    parser.add_argument('--ranking_top_n', type=int, default=5)
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
        default=0.5,
        help=(
            "Allowed RMS-contrast error for generated candidates in 0-255 "
            "pixel units."
        ),
    )
    parser.add_argument('--save_root', type=str, default="/user_data/junruz/BrainDiVE/pRF_concat_model_ranked")
    args = parser.parse_args()

    ss = args.subject_id
    if args.num_seeds < 1:
        parser.error("--num_seeds must be at least 1")
    if (
        args.max_generation_attempts is not None
        and args.max_generation_attempts < args.num_seeds
    ):
        parser.error("--max_generation_attempts cannot be less than --num_seeds")
    if args.contrast_validation_tolerance < 0:
        parser.error("--contrast_validation_tolerance must be non-negative")

    contrast_stats_path = None
    percentile_values = None
    needs_contrast_statistics = (
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

    if args.contrast_constraint_method == 'max':
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
    else:
        args.min_rms_contrast = None
        args.max_rms_contrast = None
        args.target_rms_contrast = None
        contrast_description = "disabled"

    print(
        f"Using Rec.709 RMS contrast method "
        f"{args.contrast_constraint_method!r}: {contrast_description} "
        "in 0-255 pixel units"
    )

    def path_component(value):
        """Return a readable, filesystem-safe output path component."""
        cleaned = "".join(
            character if character.isalnum() or character in "-_." else "-"
            for character in str(value).strip()
        )
        return cleaned.strip("-.") or "unnamed"

    gen_layer_sizes = getattr(LayerSize, f"{args.gen_model}_layer_size")
    rank_layer_sizes = getattr(LayerSize, f"{args.rank_model}_layer_size")
    n_pix_img = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###### BRAIN ENCODER ###################################################
    print("Building brain encoder")
    brain_encoder = build_brain_encoder(args.gen_model, device)
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
    split_dir = os.path.join(args.model_root, f'S{ss}')
    with open(os.path.join(split_dir, f'data_splits_S{ss}.pkl'), 'rb') as f:
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
        # Get the top 5 values and their indices in r2_voxels
        topk_indices = np.argsort(r2_voxels)[-args.k:][::-1]
        topk_values = r2_voxels[topk_indices]
        print(f"Top 5 r2_voxels values for {region}: {topk_values}")
        print(f"Corresponding indices: {topk_indices}")

        
        for i, idx in enumerate(topk_indices):
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
            if int(prf_idx) not in set(rank_metadata["prf_ids"]):
                raise ValueError(
                    f"Generation-selected pRF ID {prf_idx} is unavailable in "
                    "the ranking model metadata"
                )
            gen_prf_kernels = build_prf_kernels(
                gen_layer_names, gen_layer_sizes, prf_idx, device
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
            model_folder = (
                f"gen-{path_component(args.gen_model)}-concat"
                f"__rank-{path_component(args.rank_model)}-concat"
            )
            if args.contrast_constraint_method == 'max':
                contrast_folder = f"max-rms{args.max_rms_contrast:.2f}"
            elif args.contrast_constraint_method == 'range':
                contrast_folder = (
                    f"range-rms{args.min_rms_contrast:.2f}"
                    f"-{args.max_rms_contrast:.2f}"
                )
            elif args.contrast_constraint_method == 'match':
                contrast_folder = f"match-rms{args.target_rms_contrast:.2f}"
            else:
                contrast_folder = "none"
            settings_folder = (
                f"split-{args.split_id:02d}"
                f"__steps-{num_steps}"
                f"__scale-{brain_guidance_scale:g}"
                f"__contrast-{contrast_folder}"
                f"__k-{args.k}__seeds-{args.num_seeds}"
                f"__keep-{args.ranking_top_n}"
            )
            experiment_folder = os.path.join(
                args.save_root,
                f"sub-{args.subject_id:02d}",
                model_folder,
                settings_folder,
            )
            region_folder = os.path.join(
                experiment_folder,
                f"roi-{path_component(region)}",
            )
            voxel_folder = os.path.join(
                region_folder,
                f"voxel-rank-{i + 1:02d}__id-{voxel_id:06d}",
            )
            images_folder = os.path.join(voxel_folder, "images")
            os.makedirs(images_folder, exist_ok=True)

            config_file = os.path.join(experiment_folder, "run_config.json")
            if not os.path.exists(config_file):
                with open(config_file, "w") as config_handle:
                    json.dump(
                        {
                            "subject": int(args.subject_id),
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
                            },
                            "generation": {
                                "num_steps": int(num_steps),
                                "brain_guidance_scale": float(brain_guidance_scale),
                                "num_seeds": int(args.num_seeds),
                                "top_k_voxels": int(args.k),
                                "ranking_top_n": int(args.ranking_top_n),
                                "max_generation_attempts": (
                                    int(args.max_generation_attempts)
                                    if args.max_generation_attempts is not None
                                    else None
                                ),
                                "contrast_validation_tolerance_byte": float(
                                    args.contrast_validation_tolerance
                                ),
                            },
                            "contrast": {
                                "method": args.contrast_constraint_method,
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
                            },
                            "regions": list(args.roi),
                        },
                        config_handle,
                        indent=2,
                        allow_nan=False,
                    )
            pipe.brain_tweak = single_voxel_objective

            image_records = []
            attempted_seeds = set()
            rejected_candidates = {
                "nonfinite_pixels": 0,
                "invalid_shape": 0,
                "out_of_range_pixels": 0,
                "degenerate_contrast": 0,
                "contrast_mismatch": 0,
                "nonfinite_gen_score": 0,
                "nonfinite_rank_score": 0,
            }
            max_generation_attempts = args.max_generation_attempts
            if max_generation_attempts is None:
                max_generation_attempts = (
                    args.num_seeds + max(10, int(np.ceil(args.num_seeds * 0.10)))
                )
            contrast_tolerance = args.contrast_validation_tolerance / 255.0

            def candidate_rejection_reason(image_array):
                if image_array.ndim != 3 or image_array.shape[-1] != 3:
                    return "invalid_shape", None
                if not np.isfinite(image_array).all():
                    return "nonfinite_pixels", None
                if image_array.min() < -1e-6 or image_array.max() > 1.0 + 1e-6:
                    return "out_of_range_pixels", None

                # Validate the quantized pixels that will actually be saved,
                # rather than only the higher-precision pipeline output.
                validation_array = (
                    np.rint(np.clip(image_array, 0, 1) * 255.0) / 255.0
                )
                luminance = np.einsum(
                    "hwc,c->hw",
                    validation_array.astype(np.float64, copy=False),
                    np.array((0.2126, 0.7152, 0.0722), dtype=np.float64),
                )
                rms_contrast = float(luminance.std())
                if not np.isfinite(rms_contrast) or rms_contrast <= 1e-8:
                    return "degenerate_contrast", rms_contrast

                if args.contrast_constraint_method == "match":
                    valid_contrast = (
                        abs(rms_contrast - args.target_rms_contrast / 255.0)
                        <= contrast_tolerance
                    )
                elif args.contrast_constraint_method == "max":
                    valid_contrast = (
                        rms_contrast
                        <= args.max_rms_contrast / 255.0 + contrast_tolerance
                    )
                elif args.contrast_constraint_method == "range":
                    valid_contrast = (
                        rms_contrast
                        >= args.min_rms_contrast / 255.0 - contrast_tolerance
                        and rms_contrast
                        <= args.max_rms_contrast / 255.0 + contrast_tolerance
                    )
                else:
                    valid_contrast = True
                if not valid_contrast:
                    return "contrast_mismatch", rms_contrast
                return None, rms_contrast

            def generate_pixel_valid_candidates(number_needed):
                generated = []
                while (
                    len(generated) < number_needed
                    and len(attempted_seeds) < max_generation_attempts
                ):
                    seed = random.randint(0, 2**32 - 1)
                    while seed in attempted_seeds:
                        seed = random.randint(0, 2**32 - 1)
                    attempt_number = len(attempted_seeds) + 1
                    attempted_seeds.add(seed)
                    print(
                        f"Starting {attempt_number:05d} "
                        f"(accepted {len(image_records) + len(generated)}/"
                        f"{args.num_seeds})"
                    )
                    g = torch.Generator(device=device).manual_seed(int(seed))
                    image = pipe(
                        "",
                        sag_scale=0.75,
                        guidance_scale=0.0,
                        num_inference_steps=num_steps,
                        generator=g,
                        clip_guidance_scale=brain_guidance_scale,
                        contrast_constraint_method=args.contrast_constraint_method,
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
                        output_type="np",
                    )
                    image_array = np.asarray(image.images[0], dtype=np.float32)
                    reason, rms_contrast = candidate_rejection_reason(image_array)
                    if reason is not None:
                        rejected_candidates[reason] += 1
                        print(
                            f"Rejected seed {seed}: {reason}"
                            + (
                                f" (RMS={rms_contrast * 255.0:.4f})"
                                if rms_contrast is not None
                                else ""
                            )
                        )
                        continue
                    generated.append({
                        "seed": int(seed),
                        "tensor": torch.from_numpy(
                            np.ascontiguousarray(image_array)
                        ).permute(2, 0, 1),
                        "rms_contrast_byte": rms_contrast * 255.0,
                    })
                return generated

            pending_records = generate_pixel_valid_candidates(args.num_seeds)
            if len(pending_records) < args.num_seeds:
                raise RuntimeError(
                    f"Only generated {len(pending_records)} pixel-valid "
                    f"candidates for {region} voxel {voxel_id} after "
                    f"{len(attempted_seeds)} attempts "
                    f"(limit {max_generation_attempts}). "
                    f"Rejections: {rejected_candidates}"
                )
            
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

            rank_weights, rank_channel_kept, best_lambda_idx, rank_features_s, rank_features_m = rank_model_fitting(
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

            while len(image_records) < args.num_seeds:
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

                if len(image_records) >= args.num_seeds:
                    break
                missing = args.num_seeds - len(image_records)
                pending_records = generate_pixel_valid_candidates(missing)
                if not pending_records:
                    raise RuntimeError(
                        f"Only generated {len(image_records)} fully valid "
                        f"candidates for {region} voxel {voxel_id} after "
                        f"{len(attempted_seeds)} attempts "
                        f"(limit {max_generation_attempts}). "
                        f"Rejections: {rejected_candidates}"
                    )

            ranked_indices = sorted(
                range(len(image_records)),
                key=lambda j: image_records[j]["rank_score"],
                reverse=True,
            )

            top_n = max(1, int(args.ranking_top_n))
            top_n = min(top_n, len(ranked_indices))
            top_ranked = []
            for rank_idx, img_idx in enumerate(ranked_indices[:top_n], start=1):
                item = image_records[img_idx]
                ranked_image_name = (
                    f"rank-{rank_idx:02d}__seed-{item['seed']:010d}.png"
                )
                ranked_image_path = os.path.join(images_folder, ranked_image_name)
                image_array = item["tensor"].permute(1, 2, 0).numpy()
                pil_image = Image.fromarray(
                    np.rint(np.clip(image_array, 0, 1) * 255).astype(np.uint8),
                    mode="RGB",
                )
                pil_image.save(ranked_image_path, format="PNG", compress_level=6)
                pil_image.close()
                top_ranked.append({
                    "seed": item["seed"],
                    "rank": rank_idx,
                    "gen_score": item["gen_score"],
                    "rank_score": item["rank_score"],
                    "rms_contrast_byte": item["rms_contrast_byte"],
                    "image_file": os.path.join("images", ranked_image_name),
                    "image_path": os.path.abspath(ranked_image_path),
                })

            ranking_file = os.path.join(voxel_folder, "ranking.json")
            with open(ranking_file, "w") as f:
                json.dump(
                    {
                        "schema_version": 3,
                        "region": region,
                        "subject": int(ss),
                        "voxel_id": int(voxel_id),
                        "voxel_index_in_region": int(idx),
                        "voxel_rank_in_topk": int(i + 1),
                        "voxel_r2": float(topk_values[i]),
                        "num_candidates": int(len(image_records)),
                        "generation_attempts": int(len(attempted_seeds)),
                        "max_generation_attempts": int(max_generation_attempts),
                        "contrast_validation_tolerance_byte": float(
                            args.contrast_validation_tolerance
                        ),
                        "rejected_candidates": rejected_candidates,
                        "ranking_top_n": int(top_n),
                        "results": top_ranked,
                    },
                    f,
                    indent=2,
                    allow_nan=False,
                )

            # Release temporary image tensors after each voxel.
            for item in image_records:
                item["tensor"] = None
            del ranked_indices
            del image_records
            gc.collect()
            torch.cuda.empty_cache()

            if len(top_ranked) > 0:
                print(
                    f"Voxel ranking complete for {region} voxel {voxel_id}: "
                    f"best_seed={top_ranked[0]['seed']}, "
                    f"best_score={top_ranked[0]['rank_score']:.6f}, "
                    f"kept_top_n={top_n}, "
                    f"saved={ranking_file}"
                )



if __name__ == "__main__":
    main()
