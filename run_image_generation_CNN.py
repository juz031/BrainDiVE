import sys
import os
import time

begin_time = time.time()
import timm
import torch
import numpy as np
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, LMSDiscreteScheduler
from encoder_dataloader import neural_loader
from brain_guide_pipeline import mypipelineSAG
import pickle
import gc
import encoder_model_vit
import nibabel as nib
import os
import clip
import argparse
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from encoder_training import soft_quantizer_v3, neural_loader
end_time = time.time()
print(f"Time taken to import libraries: {end_time - begin_time} seconds")


def load_from_nii(nii_file):
    return nib.load(nii_file).get_fdata()

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

train_nodes, eval_nodes = get_graph_node_names(CNN_backbone)
return_nodes = {
    # 'relu1': 'conv1', #torch.Size([1, 32, 112, 112])
    # 'relu2': 'conv2', #torch.Size([1, 32, 112, 112])
    # 'relu3': 'conv3', #torch.Size([1, 64, 112, 112])
    # 'avgpool': 'avgpool', #torch.Size([1, 64, 56, 56])
    # 'layer1': 'layer1', #torch.Size([1, 256, 56, 56])
    'layer2': 'layer2', #torch.Size([1, 512, 28, 28])
    # 'layer3': 'layer3', #torch.Size([1, 1024, 14, 14])
    # 'layer4': 'layer4', #torch.Size([1, 2048, 7, 7])
    # 'attnpool': 'attnpool' #torch.Size([1, 1024])

}

#TODO Do PCA on intermediate features, and get projection matrix from PCA

feature_extractor = create_feature_extractor(CNN_backbone, return_nodes=return_nodes)
# out = feature_extractor(torch.rand(1, 3, 224, 224).to(device))
# print([(k, v.shape) for k, v in out.items()])
print("Created CLIP ResNet50 and moved to GPU, feature extractor created")

# TODO if this part is loading mask, write code to load mask in our way
subjects = [1]
roi_root = "/lab_data/hendersonlab/datasets/nsd_preproc/rois"
roi_path = os.path.join(roi_root, "S{}_voxel_roi_info.npy")

functional = []
functional_dict = {}
print("Loading mask from roi files")

for s in subjects:
    selected = [] # voxel selected, TODO select voxels based on R2 values
    roi_info = np.load(roi_path.format(s), allow_pickle=True).item()
    big_mask = roi_info['voxel_mask']
    roi_keys = ['roi_labels_face', 'roi_labels_place', 'roi_labels_retino'] #['roi_labels_retino', 'roi_labels_kastner', 'roi_labels_face', 'roi_labels_place', 'roi_labels_body']
    roi_names = ['floc_face_roi_names', 'floc_place_roi_names', 'ret_prf_roi_names'] #['ret_prf_roi_names', 'kastner_atlas_roi_names', 'floc_face_roi_names', 'floc_place_roi_names', 'floc_body_roi_names']
    for key, name in zip(roi_keys, roi_names):
        roi_labels = roi_info[key][big_mask]
        roi_name = roi_info[name]
        for name in roi_name.keys():
            functional_dict[name] = roi_labels==roi_name[name]
print(functional_dict.keys())


import random
from base64 import b64encode
random.seed(a=b64encode(os.urandom(5)).decode('utf-8'))

all_subjects = [1]
random.shuffle(all_subjects)
# We shuffle here so the slurm run tries to avoid collisions
# ["RSC", "PPA", "OPA", "FFA", "OFA"]
experiment_id = {"bodies":0, "faces":1, "places":2, "words":3, "food":4,
                 "RSC":5, "PPA":6, "OPA":7, "FFA":8, "OFA":9}

############################# Loop over subjects #########################################
for subject in all_subjects:
    # TODO create a new random seeds file for the new experiment, similar to the below:

    subject_seeds = [0,1,2,3,4,5,6,7,8,9]
    print("Starting subject {}".format(str(subject)))
    try:
        del dataset
    except:
        pass

    try:
        del brain_projector
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    
    ############################## Folder paths #########################################
    nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
    data_folder = os.path.join(nsd_path, 'data')
    labels_folder = os.path.join(nsd_path, 'labels')
    stim_folder = os.path.join(nsd_path, 'stimuli')
    rois_folder = os.path.join(nsd_path, 'rois')

    ############################## NSD data paths #########################################
    n_pix = 224
    args = argparse.Namespace()
    args.subject_id = [1]
    args.neural_activity_path = os.path.join(data_folder, 'S{}_betas_avg_bigmask_dict.hdf5')
    args.image_path = os.path.join(stim_folder, 'S{}_stimuli_%d_dict.hdf5'%n_pix)
    args.roi_path = os.path.join(rois_folder, 'S{}_voxel_roi_info.npy')
    args.stim_keys_path = os.path.join(stim_folder, "all_keys.pkl")


    dataset = neural_loader(args)
    ############################## Brain Projector #########################################
    brain_projector = soft_quantizer_v3(num_higher_output=1433, embed_dim=512*14*14)

    ckpt_path = f"/home/junruz/BrainDiVE/weights/S1_layer2_V2_RN50_bs128_e400_AdamW_lr0.0003_decay0.5_wd0.015.pth"
    state_dict = torch.load(ckpt_path, map_location='cpu')
    brain_projector.load_state_dict(state_dict, strict=True)
    
    # load checkpoint with r2 values
    # ckpt_path = f"/home/junruz/BrainDiVE/weights/weights_bs256_e400_AdamW_lr0.0003_decay0.5_wd0.015_r2.pth"
    # brain_projector = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    brain_projector.cuda()
    brain_projector.eval()



    for name, param in brain_projector.named_parameters():
        param.requires_grad = False
    assert not brain_projector.training

    ######## Where is the memory leak??????
    try:
        del pipe
    except:
        pass

    try:
        del pipe2
    except:
        pass

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    ########################## Stable Diffusion Pipeline #########################################
    repo_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = mypipelineSAG.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe2 = pipe.to("cuda")


    ############################## Loop over regions #########################################
    # regions = ["bodies", "faces", "places", "words", "food"]
    # regions = list(functional_dict.keys())#["FFA-1", "FFA-2", "PPA"]
    regions = ['V2']
    random.seed(a=b64encode(os.urandom(5)).decode('utf-8'))
    random.shuffle(regions) # TODO why shuffle?
    # shuffle here to avoid slurm collision
    for region in regions:
        print("Starting S{} {}".format(subject, region))
        region_seeds = [0,1,2,3,4,5,6,7,8,9]
        # random.seed(a=b64encode(os.urandom(5)).decode('utf-8'))
        # random.shuffle(region_seeds)
        try:
            del region_mask
        except:
            pass

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        region_mask = (torch.tensor(functional_dict[f'{region}v']) + torch.tensor(functional_dict[f'{region}d'])).unsqueeze(0).bool().to("cuda")
        noise_ceiling = (torch.tensor(dataset.noise_ceiling[str(args.subject_id[0])])/ 100).unsqueeze(0).to("cuda")
        region_mask = noise_ceiling[region_mask] > 0.1
        region_mask = region_mask.unsqueeze(0)
        print(f"Selected {region_mask.sum().item()} voxels")



        # print(f"size of mask: {np.shape(region_mask)} num of true: {np.sum(region_mask)}")
    
        # create r2 mask based on r2 threshold
        # r2_values = torch.tensor(brain_projector.r2.squeeze()).unsqueeze(0).to("cuda")
        # r2_mask = torch.tensor(r2_values[region_mask] >= 0.2).unsqueeze(0).bool().to("cuda")
        # TODO write the objective function for the gradient guidance, replace loss_function_higher in run_brain_guidance_category.py
        def guider_objective(image_input):
            # Aim: Maximize the predicted mean response for the region
            image_features = feature_extractor(image_input)
            image_features = image_features['layer2'].float()
            image_features = torch.nn.functional.avg_pool2d(image_features, kernel_size=2, stride=2)
            image_features = image_features.view(image_features.size(0), -1)
            image_features = image_features/(image_features.norm(dim=-1, keepdim=True)+1e-10)
            pred_response = brain_projector(image_features)
            pred_response = pred_response[region_mask]
            # pred_response = pred_response[r2_mask]
            return -torch.mean(pred_response)
        
        def activate_deactivate_objective(image_input):
            # Aim: Maximize the predicted mean response for the region
            area_activate = region
            area_deactivate = list(set(regions) - set([region]))[0]

            area_activate_mask = torch.tensor(functional_dict[area_activate]).unsqueeze(0).bool().to("cuda")
            area_deactivate_mask = torch.tensor(functional_dict[area_deactivate]).unsqueeze(0).bool().to("cuda")
            image_featrues = CNN_backbone.encode_image(image_input)
            image_featrues = image_featrues/image_featrues.norm(dim=-1, keepdim=True)
            pred_voxels = brain_projector(image_featrues.float())
            activate_response = pred_voxels[area_activate_mask]
            deactivate_response = pred_voxels[area_deactivate_mask]
            return -torch.mean(activate_response) + torch.mean(deactivate_response)

        num_steps = 100
        brain_guidance_scale = 150
        current_folder = f"/user_data/junruz/BrainDiVE/S1_600e_RN50_layer2/test_step{num_steps}_scale{brain_guidance_scale}/{f"S{subject}"}/{region}"
        # try:
        os.makedirs(current_folder, exist_ok=True)
        offset = 0
        pipe.brain_tweak = guider_objective
        for seed in region_seeds:
            offset += 1
            print("Starting {}".format(str(offset).zfill(5)))
            if offset % 20 == 0:
                print("S{}, {} region, {}/500".format(subject, region, offset))
                gc.collect()
            if offset % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
            image_name = os.path.join(current_folder, "{}_{}.png".format(region, str(seed).zfill(4)))
            if os.path.exists(image_name):
                print("image already exists, skipping")
                continue
            g = torch.Generator(device="cuda").manual_seed(int(seed))
            image = pipe("", sag_scale=0.75, guidance_scale=0.0, num_inference_steps=num_steps, generator=g, clip_guidance_scale=brain_guidance_scale)
            if os.path.exists(image_name):
                continue
            image.images[0].save(image_name, format="PNG", compress_level=6)