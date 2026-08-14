import sys
import os
import time
import json
import h5py

import torch
import numpy as np
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



class LayerSize:
    pass

setattr(LayerSize, "CLIP_RN50_layer_size", {
    'relu1': [32, 112, 112],
    'relu2': [32, 112, 112],
    'relu': [64, 112, 112],
    'avgpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
})

setattr(LayerSize, "OPEN_CLIP_RN50_layer_size", {
    'act1': [32, 112, 112],
    'act2': [32, 112, 112],
    'relu': [64, 112, 112],
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
    return (scaled_image-getattr(Normalize, f"{args.model_name}_MEAN"))/getattr(Normalize, f"{args.model_name}_STD")


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
            'relu3': 'relu3', #torch.Size([1, 64, 112, 112])
            'avgpool': 'avgpool', #torch.Size([1, 64, 56, 56])
            'layer1': 'layer1', #torch.Size([1, 256, 56, 56])
            'layer2': 'layer2', #torch.Size([1, 512, 28, 28])
            'layer3': 'layer3', #torch.Size([1, 1024, 14, 14])
            'layer4': 'layer4' #torch.Size([1, 2048, 7, 7])
        }
    
    elif model_name == "OPEN_CLIP_RN50":
        model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='yfcc15m')
        CNN_backbone = model.visual
        del model.transformer
        torch.cuda.empty_cache()

        # train_nodes, val_nodes = get_graph_node_names(CNN_backbone)
        
        return_nodes = {
            'act1': 'act1', #torch.Size([1, 32, 112, 112])
            'act2': 'act2', #torch.Size([1, 32, 112, 112])
            'act3': 'act3', #torch.Size([1, 64, 112, 112])
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


def rank_model_fitting(voxel_id, best_prf_idx,voxel_data, train_ids, val_ids, nest_ids, features_folder, args, device):
    """
    Return:
    - best_prf_idx: list of length of num_voxels, the index of the best pRF for each voxel
    - best_lambda: list of length of num_voxels, the best lambda for each voxel for the best pRF
    - best_loss: list of length of num_voxels, the best r2 for each voxel for the best pRF
    - best_weights: array of length of [num_voxels, num_features + 1], the best weights for each voxel for the best pRF, with the intercept
    """

    train_voxel = voxel_data[train_ids, voxel_id]
    nest_voxel = voxel_data[nest_ids, voxel_id]

    train_voxel = torch.from_numpy(train_voxel).float().to(device)
    nest_voxel = torch.from_numpy(nest_voxel).float().to(device)

    num_features = getattr(LayerSize, args.model_name + '_layer_size')[args.layer_name][0]

    features_path = os.path.join(features_folder, f'features_prf_{best_prf_idx}.npy')
    features = np.load(features_path)

    # split train and nest features, then z-score the features across images axis 0 using model_fitting_utils.split_normalize_feats
    # train_features, val_features, nest_features, features_s, features_m = model_fitting_utils.split_normalize_feats(features, train_ids, val_ids, nest_ids)
    train_features, val_features, nest_features, channel_kept, features_s, features_m = model_fitting_utils.split_normalize_feats_with_drop(features, train_ids, val_ids, nest_ids)
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
    parser.add_argument('--layer_name', type=str, default="layer2")
    parser.add_argument('--feature_root', type=str, default="/user_data/junruz/prf_features")
    parser.add_argument('--model_root', type=str, default="/user_data/junruz/prf_models/fixed/split_1_zscore")
    parser.add_argument('--roi', nargs='+', default=["V1"], type=str)
    parser.add_argument('--k', type=int, default=3, help="Top k voxels to keep in the region")
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--brain_guidance_scale', type=float, default=30)
    parser.add_argument('--max_rms_contrast', type=float, default=16.0,
                        help="Maximum Rec.709 RMS contrast in 0-255 pixel units")
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--ranking_top_n', type=int, default=5)
    parser.add_argument('--save_root', type=str, default="/user_data/junruz/BrainDiVE/pRF_zscore_drop_model_ranked")
    args = parser.parse_args()

    ss = args.subject_id

    n_pix_feat = getattr(LayerSize, f"{args.gen_model}_layer_size")[args.layer_name][-1]
    n_pix_img = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###### BRAIN ENCODER ###################################################
    print("Building brain encoder")
    brain_encoder = build_brain_encoder(args.gen_model, device)
    print("Building rank encoder")
    rank_encoder = build_brain_encoder(args.rank_model, device)


    gen_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=getattr(Normalize, f"{args.gen_model}_MEAN"),
            std=getattr(Normalize, f"{args.gen_model}_STD"),
        ),
    ])

    rank_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=getattr(Normalize, f"{args.rank_model}_MEAN"),
            std=getattr(Normalize, f"{args.rank_model}_STD"),
        ),
    ])


    def score_generated_images_for_voxel(image_records, channel_idx, prf_kernel, voxel_weights, voxel_intercept, features_s, features_m, batch_size=8, encoder=rank_encoder, preprocess=rank_preprocess):
        if len(image_records) == 0:
            return []

        channel_idx = torch.as_tensor(channel_idx, device=device, dtype=torch.long)
        scores = []
        with torch.no_grad():
            for batch_start in range(0, len(image_records), batch_size):
                batch_records = image_records[batch_start: batch_start + batch_size]
                batch_tensor = torch.stack(
                    [preprocess(item["pil"].convert("RGB")) for item in batch_records],
                    dim=0,
                ).to(device)
                image_features = encoder(batch_tensor)[args.layer_name].double()
                image_features = image_features[:, channel_idx, :, :]
                prf_features = torch.einsum('bchw,hw->bc', image_features, prf_kernel)
                prf_features = (prf_features - features_m) / features_s
                voxel_responses = torch.matmul(prf_features, voxel_weights.squeeze(0)) + voxel_intercept.squeeze(0)
                scores.extend(voxel_responses.detach().cpu().numpy().tolist())

        return scores



    ############################## Brain PRF Model #########################################

    model_path = os.path.join(args.model_root, f"S{args.subject_id}", args.gen_model, args.layer_name)
    weights_path = os.path.join(model_path, 'best_weights.pkl')
    channel_kept_path = os.path.join(model_path, 'channel_kept.pkl')
    prf_idx_path = os.path.join(model_path, 'best_prf_idx.npy')
    lambda_path = os.path.join(model_path, 'best_lambda.npy')
    r2_path = os.path.join(model_path, 'r2.pkl')
    features_s_path = os.path.join(model_path, 'best_features_s.npy')
    features_m_path = os.path.join(model_path, 'best_features_m.npy')


    ####################### Turn on if zscore is used #######################
    # channel_mean = np.load(os.path.join(feature_path, 'channel_mean.npy'))
    # channel_std = np.load(os.path.join(feature_path, 'channel_std.npy'))
    
    with open(weights_path, "rb") as f:
        best_weights = pickle.load(f)
    with open(channel_kept_path, "rb") as f:
        channel_kept = pickle.load(f)
    best_prf_idx = np.load(prf_idx_path)
    with open(r2_path, "rb") as f:
        r2 = pickle.load(f)
    best_features_s = np.load(features_s_path)
    best_features_m = np.load(features_m_path)

    best_features_s = torch.from_numpy(best_features_s).to(device)
    best_features_m = torch.from_numpy(best_features_m).to(device)
    # Load rois masks
    roi_masks, noise_ceiling = load_nsd_rois(ss, args)
    prf_stack = get_prf_stack(n_pix=n_pix_feat, which_prf_grid='default-log-polar')


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
        # region_mask = region_mask.unsqueeze(0).to(device)
        # Apply region_mask to best_weights
        # This will create a new dict of best_weights only for voxels where region_mask is True
        best_weights_region = {k: v for i, (k, v) in enumerate(best_weights.items()) if region_mask[i]}
        weights_region_keys = list(best_weights_region.keys())

        best_prf_idx_region = best_prf_idx[region_mask]

        channel_kept_region = {k: v for i, (k, v) in enumerate(channel_kept.items()) if region_mask[i]}
        channel_kept_region_keys = list(channel_kept_region.keys())
        
        r2_voxels = np.squeeze(r2[region]['r2_voxels'])
        # Get the top 5 values and their indices in r2_voxels
        topk_indices = np.argsort(r2_voxels)[-args.k:][::-1]
        topk_values = r2_voxels[topk_indices]
        print(f"Top 5 r2_voxels values for {region}: {topk_values}")
        print(f"Corresponding indices: {topk_indices}")

        
        for i, idx in enumerate(topk_indices):
            weights = best_weights_region[weights_region_keys[idx]][:-1]
            intercept = best_weights_region[weights_region_keys[idx]][-1]

            weights = torch.from_numpy(weights).unsqueeze(0).double().to(device)
            intercept = torch.tensor(intercept).unsqueeze(0).double().to(device)

            prf_idx = best_prf_idx_region[idx]
            prf_gaussian = prf_stack[:,:,prf_idx]
            prf_gaussian = torch.from_numpy(prf_gaussian).double().to(device)

            channel_used = channel_kept_region[channel_kept_region_keys[idx]]

            # Turn on if zscore is used
            # mean = channel_mean[prf_idx, channel_used]
            # std = channel_std[prf_idx, channel_used]
            # mean = torch.from_numpy(mean).double().to(device)
            # std = torch.from_numpy(std).double().to(device)

            def single_voxel_objective(image_input):
                # Aim: Maximize the predicted mean response for the region
                image_features = brain_encoder(image_input)
                image_features = image_features[args.layer_name].double().squeeze(0)
                image_features = image_features[channel_used,:,:]

                prf_features = torch.einsum('chw,hw->c', image_features, prf_gaussian)
                prf_features = (prf_features - best_features_m) / best_features_s

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
            current_folder = os.path.join(args.save_root, f"S{args.subject_id}_{args.model_name}_{args.layer_name}/test_step{num_steps}_scale{brain_guidance_scale}/{region}")
            # try:
            os.makedirs(current_folder, exist_ok=True)
            offset = 0
            pipe.brain_tweak = single_voxel_objective

            image_records = []

            seeds = [random.randint(0, 2**32 - 1) for _ in range(args.num_seeds)]
            for seed in seeds:
                offset += 1
                print("Starting {}".format(str(offset).zfill(5)))
                g = torch.Generator(device=device).manual_seed(int(seed))
                image = pipe("", sag_scale=0.75, guidance_scale=0.0, num_inference_steps=num_steps, generator=g, clip_guidance_scale=brain_guidance_scale, max_rms_contrast=args.max_rms_contrast / 255.0)
                image_records.append({
                    "seed": int(seed),
                    "pil": image.images[0].convert("RGB"),
                })
            
            ############## Train the ranking model based on voxel id #########################################
            split_dir = os.path.join(args.model_root, f'S{ss}')
            with open(os.path.join(split_dir, f'data_splits_S{ss}.pkl'), 'rb') as f:
                data_splits = pickle.load(f) # 0: ratio, 1:split 1, 2:split 2

            ids = data_splits[args.split_id]
            train_ids, val_ids, nest_ids = ids['train'], ids['val'], ids['nest']
            features_folder = os.path.join(args.feature_root, f"S{args.subject_id}", args.rank_model, args.layer_name)
            print(f"Loading features from: {features_folder}")

            voxel_data, good_values = load_nsd_data(data_folder, labels_folder, ss)

            rank_weights, rank_channel_kept, best_lambda_idx, rank_features_s, rank_features_m = rank_model_fitting(idx, prf_idx, voxel_data, train_ids, val_ids, nest_ids, features_folder, args, device)

            rank_intercept = rank_weights[-1]
            rank_weights = rank_weights[:-1]

            pred_scores = score_generated_images_for_voxel(
                image_records=image_records,
                channel_idx=rank_channel_kept,
                prf_kernel=prf_gaussian,
                voxel_weights=rank_weights,
                voxel_intercept=rank_intercept,
                features_s=rank_features_s,
                features_m=rank_features_m,
                batch_size=8,
            )
            ranked_indices = sorted(
                range(len(pred_scores)),
                key=lambda j: pred_scores[j],
                reverse=True,
            )

            top_n = max(1, int(args.ranking_top_n))
            top_n = min(top_n, len(ranked_indices))
            top_ranked = []
            for rank_idx, img_idx in enumerate(ranked_indices[:top_n], start=1):
                item = image_records[img_idx]
                score = float(pred_scores[img_idx])
                score_tag = f"{score:.6f}"
                ranked_image_name = (
                    f"{region}_top{i+1}_voxel{idx}_rank{rank_idx}_score{score_tag}_seed{str(item['seed']).zfill(4)}.png"
                )
                ranked_image_path = os.path.join(current_folder, ranked_image_name)
                item["pil"].save(ranked_image_path, format="PNG", compress_level=6)
                top_ranked.append({
                    "seed": item["seed"],
                    "rank": rank_idx,
                    "pred_score": score,
                    "image_path": ranked_image_path,
                })
            ranked_results = top_ranked

            ranking_file = os.path.join(current_folder, f"{region}_top{i+1}_voxel{idx}_ranking.json")
            with open(ranking_file, "w") as f:
                json.dump(
                    {
                        "region": region,
                        "subject": int(ss),
                        "voxel_index_in_region": int(idx),
                        "voxel_rank_in_topk": int(i + 1),
                        "ranking_top_n": int(top_n),
                        "results": ranked_results,
                        "top_ranked": top_ranked,
                    },
                    f,
                    indent=2,
                )

            # Release PIL images and temporary tensors after each voxel.
            for item in image_records:
                if "pil" in item:
                    item["pil"].close()
                    item["pil"] = None
            del pred_scores
            del ranked_indices
            del image_records
            gc.collect()
            torch.cuda.empty_cache()

            if len(top_ranked) > 0:
                print(
                    f"Voxel ranking complete for {region} voxel {idx}: "
                    f"best_seed={top_ranked[0]['seed']}, best_score={top_ranked[0]['pred_score']:.6f}, "
                    f"kept_top_n={top_n}, "
                    f"saved={ranking_file}"
                )



if __name__ == "__main__":
    main()
