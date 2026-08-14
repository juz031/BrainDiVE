import sys
import os
import time
import json

import torch
import numpy as np
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, LMSDiscreteScheduler
from brain_guide_pipeline import mypipelineSAG
import pickle
import gc

import nibabel as nib
import os
import clip
import argparse
from torchvision.models.feature_extraction import create_feature_extractor
from prf_utils import get_prf_stack
import random
from base64 import b64encode
random.seed(a=b64encode(os.urandom(5)).decode('utf-8'))
from PIL import Image
from torchvision import transforms


clip_rn50_layer_size = {
    'relu1': [32, 112, 112],
    'relu2': [32, 112, 112],
    'relu3': [64, 112, 112],
    'avgpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
}


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

def main():
    ### PATHS ############################################################
    nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
    rois_folder = os.path.join(nsd_path, 'rois')
    ### ARGUMENTS ############################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=int, default=1)
    parser.add_argument('--roi_path', type=str, default=os.path.join(rois_folder, 'S{}_voxel_roi_info.npy'))
    parser.add_argument('--model_name', type=str, default="RN50")
    parser.add_argument('--layer_name', type=str, default="layer2")
    parser.add_argument('--model_root', type=str, default="/user_data/junruz/prf_models/no_zscore_drop")
    parser.add_argument('--roi', nargs='+', default=["V1"], type=str)
    parser.add_argument('--k', type=int, default=3, help="Top k voxels to keep in the region")
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--brain_guidance_scale', type=float, default=30)
    parser.add_argument('--max_rms_contrast', type=float, default=16.0,
                        help="Maximum Rec.709 RMS contrast in 0-255 pixel units")
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--ranking_top_n', type=int, default=5)
    parser.add_argument('--save_root', type=str, default="/user_data/junruz/BrainDiVE/pRF_no_zscore_ranked")
    args = parser.parse_args()

    ss = args.subject_id

    n_pix_feat = clip_rn50_layer_size[args.layer_name][-1]
    n_pix_img = 224

    ### BRAIN ENCODER ############################################################
    print("Creating CLIP ResNet50")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "RN50"
    # visual_encoder, _ = clip.load("ViT-B/32", device=device)
    visual_encoder, _ = clip.load(model_name, device=device)
    CNN_backbone = visual_encoder.visual.to(device)

    del visual_encoder.transformer
    torch.cuda.empty_cache()
    CNN_backbone.eval()
    assert not visual_encoder.training
    for name, param in CNN_backbone.named_parameters():
        param.requires_grad = False

    return_nodes = {
        'relu1': 'conv1', #torch.Size([1, 32, 112, 112])
        'relu2': 'conv2', #torch.Size([1, 32, 112, 112])
        'relu3': 'conv3', #torch.Size([1, 64, 112, 112])
        'avgpool': 'avgpool', #torch.Size([1, 64, 56, 56])
        'layer1': 'layer1', #torch.Size([1, 256, 56, 56])
        'layer2': 'layer2', #torch.Size([1, 512, 28, 28])
        'layer3': 'layer3', #torch.Size([1, 1024, 14, 14])
        'layer4': 'layer4'#torch.Size([1, 2048, 7, 7])

    }


    feature_extractor = create_feature_extractor(CNN_backbone, return_nodes=return_nodes)
    # out = feature_extractor(torch.rand(1, 3, 224, 224).to(device))
    # print([(k, v.shape) for k, v in out.items()])
    print("Created CLIP ResNet50 and moved to GPU, feature extractor created")

    clip_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    def score_generated_images_for_voxel(image_records, channel_idx, prf_kernel, voxel_weights, voxel_intercept, batch_size=8):
        if len(image_records) == 0:
            return []

        channel_idx = torch.as_tensor(channel_idx, device=device, dtype=torch.long)
        scores = []
        with torch.no_grad():
            for batch_start in range(0, len(image_records), batch_size):
                batch_records = image_records[batch_start: batch_start + batch_size]
                batch_tensor = torch.stack(
                    [clip_preprocess(item["pil"].convert("RGB")) for item in batch_records],
                    dim=0,
                ).to(device)
                image_features = feature_extractor(batch_tensor)[args.layer_name].double()
                image_features = image_features[:, channel_idx, :, :]
                prf_features = torch.einsum('bchw,hw->bc', image_features, prf_kernel)
                voxel_responses = torch.matmul(prf_features, voxel_weights.squeeze(0)) + voxel_intercept.squeeze(0)
                scores.extend(voxel_responses.detach().cpu().numpy().tolist())
        return scores



    ############################## Brain PRF Model #########################################

    model_path = os.path.join(args.model_root, f"S{args.subject_id}", args.model_name, args.layer_name)
    weights_path = os.path.join(model_path, 'best_weights.pkl')
    channel_kept_path = os.path.join(model_path, 'channel_kept.pkl')
    prf_idx_path = os.path.join(model_path, 'best_prf_idx.npy')
    lambda_path = os.path.join(model_path, 'best_lambda.npy')
    r2_path = os.path.join(model_path, 'r2.pkl')

    # Turn on if zscore is used
    # feature_root = '/user_data/junruz/prf_features'
    # feature_path = os.path.join(feature_root, f"S{args.subject_id}", args.model_name, args.layer_name)
    # channel_mean = np.load(os.path.join(feature_path, 'channel_mean.npy'))
    # channel_std = np.load(os.path.join(feature_path, 'channel_std.npy'))
    
    with open(weights_path, "rb") as f:
        best_weights = pickle.load(f)
    with open(channel_kept_path, "rb") as f:
        channel_kept = pickle.load(f)
    best_prf_idx = np.load(prf_idx_path)
    best_lambda = np.load(lambda_path)
    with open(r2_path, "rb") as f:
        r2 = pickle.load(f)
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
                image_features = feature_extractor(image_input)
                image_features = image_features[args.layer_name].double().squeeze(0)
                image_features = image_features[channel_used,:,:]

                prf_features = torch.einsum('chw,hw->c', image_features, prf_gaussian)
                # prf_features = (prf_features - mean) / std


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
                # if offset % 20 == 0:
                #     print("S{}, {} region, {}/500".format(ss, region, offset))
                #     gc.collect()
                # if offset % 50 == 0:
                #     gc.collect()
                #     torch.cuda.empty_cache()
                #     gc.collect()
                g = torch.Generator(device=device).manual_seed(int(seed))
                image = pipe("", sag_scale=0.75, guidance_scale=0.0, num_inference_steps=num_steps, generator=g, clip_guidance_scale=brain_guidance_scale, max_rms_contrast=args.max_rms_contrast / 255.0)
                image_records.append({
                    "seed": int(seed),
                    "pil": image.images[0].convert("RGB"),
                })

            pred_scores = score_generated_images_for_voxel(
                image_records=image_records,
                channel_idx=channel_used,
                prf_kernel=prf_gaussian,
                voxel_weights=weights,
                voxel_intercept=intercept,
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
