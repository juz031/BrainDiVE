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
from skimage.transform import resize

from encoder_training import soft_quantizer_v3, neural_loader
end_time = time.time()
print(f"Time taken to import libraries: {end_time - begin_time} seconds")


OPENAI_CLIP_MEAN = np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.single)[:, None, None]
OPENAI_CLIP_STD = np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.single)[:, None, None]

def load_from_nii(nii_file):
    return nib.load(nii_file).get_fdata()

def listdir(path):
    return [os.path.join(path, x) for x in os.listdir(path)]

def join(*paths):
    return os.path.join(*paths)

def check_between(start_count, end_count, check_idx):
    return (check_idx >= start_count) and (check_idx < end_count)

def normalize_image(input_ndarray):
    # print(input_ndarray.dtype, np.max(input_ndarray), np.min(input_ndarray), input_ndarray.shape)
    # exit()
    image_resized = resize(input_ndarray, (224, 224), preserve_range=True)
    scaled_image = image_resized.astype(np.single).transpose((2, 0, 1))*random.uniform(0.95, 1.05)/(255.0)
    # print(scaled_image.shape, OPENAI_CLIP_STD.shape, OPENAI_CLIP_MEAN.shape, "SHAPES")
    return (scaled_image-OPENAI_CLIP_MEAN)/OPENAI_CLIP_STD

def normalize_image_deterministic(input_ndarray):
    image_resized = resize(input_ndarray, (224, 224), preserve_range=True)
    scaled_image = image_resized.astype(np.single).transpose((2, 0, 1))/(255.0)
    return (scaled_image-OPENAI_CLIP_MEAN)/OPENAI_CLIP_STD

class generated_img(torch.utils.data.Dataset):
    """
    PyTorch Dataset for loading PNG images from a directory.
    Only loads .png files (case-insensitive).
    """
    def __init__(self, image_dir, transform=normalize_image):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths, self.image_names = self._gather_image_paths()

    def _gather_image_paths(self):
        image_paths = []
        image_names = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    image_paths.append(os.path.join(root, file))
                    image_names.append(file)
        return sorted(image_paths), sorted(image_names)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = self.image_names[idx]
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image)
        return torch.from_numpy(image), img_path, img_name



### BRAIN ENCODER ############################################################
print("Creating CLIP ViT")
model_name = "EVA02-CLIP-B-16"
pretrained = "eva_clip" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone, _ = clip.load("ViT-B/32", device=device)

del backbone.transformer
torch.cuda.empty_cache()
backbone.eval()
assert not backbone.training
for name, param in backbone.named_parameters():
    param.requires_grad = False
print("Created CLIP ViT and moved to GPU")

NUM_TO_GENERATE = 10


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
    roi_keys = ['roi_labels_face', 'roi_labels_place'] #['roi_labels_retino', 'roi_labels_kastner', 'roi_labels_face', 'roi_labels_place', 'roi_labels_body']
    roi_names = ['floc_face_roi_names', 'floc_place_roi_names'] #['ret_prf_roi_names', 'kastner_atlas_roi_names', 'floc_face_roi_names', 'floc_place_roi_names', 'floc_body_roi_names']
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

regions = ['PPA', 'FFA-1', 'FFA-2']
img_root = '/user_data/junruz/BrainDiVE/test_step75_scale10000.0/S{}'

############################# Loop over subjects #########################################
for subject in all_subjects:
    # TODO create a new random seeds file for the new experiment, similar to the below:
    img_root = img_root.format(subject)
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
    brain_projector = soft_quantizer_v3(num_higher_output=dataset.neural_sizes[str(subject)])
    ckpt_path = f"/home/junruz/BrainDiVE/weights/weights_bs256_e400_AdamW_lr0.0003_decay0.5_wd0.015.pth"
    state_dict = torch.load(ckpt_path, map_location='cpu')
    brain_projector.load_state_dict(state_dict, strict=True)
    brain_projector.cuda()
    brain_projector.eval()

    mean_response = dict()
    mean_response_names = dict()
        
    for region in regions:
        mean_response[region] = []
        mean_response_names[region] = []
        img_path = os.path.join(img_root, region)
        dataset = generated_img(img_path)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        region_mask = np.expand_dims(functional_dict[region], axis=0)
        for image, _, image_name in dataset_loader:
            image = image.to(device)
            with torch.no_grad():
                features = backbone.encode_image(image)
                response = brain_projector(features.float())
            response = response.detach().cpu().numpy()
            region_response = response[region_mask]
            mean_response[region].append(float(region_response.mean()))
            mean_response_names[region].append(image_name)
        print(f"Mean response for {region}: \n{mean_response[region]}, \n{mean_response_names[region]}, \nlength: {len(mean_response[region])}")


        


    

