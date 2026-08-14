import sys
import os
import time

begin_time = time.time()
import timm
import torch
import numpy as np
import pickle
import gc
import encoder_model_vit
import nibabel as nib
import os
import clip
import argparse
from tqdm import tqdm
sys.path.insert(0,'/lab_data/hendersonlab/code/utils/')
import stats_utils
import matplotlib.pyplot as plt
from encoder_training import soft_quantizer_v3, neural_loader
end_time = time.time()
print(f"Time taken to import libraries: {end_time - begin_time} seconds")

### BRAIN ENCODER ############################################################
print("Creating CLIP ViT")
model_name = "EVA02-CLIP-B-16"
pretrained = "eva_clip" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone, _ = clip.load("ViT-B/32", device=device)

del backbone.transformer
if device == 'cuda':
    torch.cuda.empty_cache()
backbone.eval()
assert not backbone.training
for name, param in backbone.named_parameters():
    param.requires_grad = False
print(f"Created CLIP ViT and moved to {device}")

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
subject = 1
brain_projector = soft_quantizer_v3(num_higher_output=dataset.neural_sizes[str(subject)])
ckpt_path = f"/home/junruz/BrainDiVE/weights/S1_bs256_e600_AdamW_lr0.0003_decay0.5_wd0.015_r2.pth"
brain_projector = torch.load(ckpt_path, map_location='cpu', weights_only=False)
brain_projector.load_state_dict(brain_projector.state_dict(), strict=True)

# state_dict = torch.load(ckpt_path, map_location='cpu')
# brain_projector.load_state_dict(state_dict, strict=True)
brain_projector.to(device)
brain_projector.eval()



test_neural_data = [] # shape: (test_set_length (907), n_voxels)

test_features = []

test_preds = []

# Get the test set length
test_set_length = len(dataset.testing_set)
print(f"Extracting {test_set_length} test samples...")

for idx in tqdm(range(test_set_length)):
    test_item = dataset.get_item_test(idx)
    
    test_neural_data.append(test_item['neural_data'].detach().to('cpu'))
    test_image_data = test_item['image_data']
    
    # get CLIP embedding one image at a time.
    feat = backbone.encode_image(test_image_data.to(device)).float()
    # print(feat.shape)
    feat = feat/feat.norm(dim=-1, keepdim=True)
    # make prediction for held-out data here
    # print(feat.shape, w_best.shape, b_best.shape)
    # print(feat.shape, w_best.T.shape, b_best[None,:].shape)

    # run forward model: predict voxel response
    y_pred = brain_projector(feat.float())
    y_pred = y_pred.detach().cpu()
    # print(y_pred.shape)

    test_preds.append(y_pred)
    # test_features.append(feature_extractor.encode_image(test_image_data.to(device)))

    del feat, test_item, y_pred

    if device == 'cuda':
        torch.cuda.empty_cache()  # Clear GPU cache
    

# Concatenate all items
test_neural_data = torch.stack(test_neural_data, dim=0)  # Shape: [test_length, neural_features]
test_preds = torch.stack(test_preds, dim=0)[:,0,:]

# get R2 per voxel
n_voxels = dataset.neural_sizes[str(subject)]
r2_voxels = np.zeros((n_voxels,1))
for vv in range(n_voxels):    
    r2_voxels[vv] = stats_utils.get_r2(test_neural_data[:,vv].cpu().numpy(), \
                                    test_preds[:,vv].cpu().numpy())

mask = dataset.functional_dict[str(args.subject_id[0])]['PPA'] #+ dataset.functional_dict[str(args.subject_id[0])]['FFA-2']
r2_voxels = r2_voxels[mask, :]
noise_ceiling = dataset.noise_ceiling[str(args.subject_id[0])][mask] / 100
selected_voxels = noise_ceiling > 0.1
print(f"Selected {np.sum(selected_voxels)} voxels")
r2_voxels = r2_voxels[selected_voxels, :]

print("mean r2: ", np.mean(r2_voxels))
plt.hist(r2_voxels)
plt.axvline(np.median(r2_voxels), color='k')
plt.title(f'R2 histogram')
plt.xlabel('R2')
plt.ylabel('Voxel count')
plt.savefig(f'/home/junruz/BrainDiVE/figure/hist_S1_PPA_ViT_thresh0.1_bs128_e400_AdamW_lr0.0003_decay0.5_wd0.015_r2.png', dpi=300)
plt.close()


a, b = np.polyfit(noise_ceiling[selected_voxels], r2_voxels, 1)

plt.figure()
plt.scatter(noise_ceiling[selected_voxels], r2_voxels, s=5, )
plt.plot(noise_ceiling[selected_voxels], a*noise_ceiling[selected_voxels] + b, color='r', linestyle='--')
plt.xlabel('Noise Ceiling')
plt.ylabel('R2 per voxel')
plt.title('R2 vs Noise Ceiling per voxel')
plt.axline((0, 0), slope=1, color='k', linestyle='--')
plt.savefig(f'/home/junruz/BrainDiVE/figure/scatter_S1_PPA_ViT_thresh0.1_bs128_e400_AdamW_lr0.0003_decay0.5_wd0.015_r2.png', dpi=300)
plt.close()
