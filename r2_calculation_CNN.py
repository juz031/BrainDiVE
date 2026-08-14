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
from encoder_training_CNN import soft_quantizer_v3, neural_loader
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

end_time = time.time()
print(f"Time taken to import libraries: {end_time - begin_time} seconds")

### BRAIN ENCODER ############################################################
print("Creating CLIP ResNet50")
model_name = "RN50"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
visual_encoder, _ = clip.load(model_name, device=device)

CNN_backbone = visual_encoder.visual.to(device)

del visual_encoder.transformer
if device == 'cuda':
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
    # 'layer2': 'layer2', #torch.Size([1, 512, 28, 28])
    # 'layer3': 'layer3', #torch.Size([1, 1024, 14, 14])
    'layer4': 'layer4', #torch.Size([1, 2048, 7, 7])
    # 'attnpool': 'attnpool' #torch.Size([1, 1024])

}

feature_extractor = create_feature_extractor(CNN_backbone, return_nodes=return_nodes)
print(f"Created CLIP ResNet50 and moved to {device}")

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
brain_projector = soft_quantizer_v3(num_higher_output=687, embed_dim=2048*7*7)
ckpt_path = f"/home/junruz/BrainDiVE/weights/S1_layer4_hV4_RN50_bs128_e400_AdamW_lr0.0003_decay0.5_wd0.015.pth"
state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
brain_projector.load_state_dict(state_dict, strict=True)

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
    feat = feature_extractor(test_image_data.to(device))['layer4'].float()
    # print(feat.shape)
    # feat = torch.nn.functional.avg_pool2d(feat, kernel_size=2, stride=2)
    feat = feat.view(feat.size(0), -1)
    feat = feat/(feat.norm(dim=-1, keepdim=True)+1e-10)
    # make prediction for held-out data here
    # if idx == 0:
    #     print(feat.shape, w_best.shape, b_best.shape)
    #     print(feat.shape, w_best.T.shape, b_best[None,:].shape)

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
r2_voxels = np.zeros((687,1)) #s1: [v1:1350, V2:1433, V3:1187, hV4:687]
for vv in range(687):    
    r2_voxels[vv] = stats_utils.get_r2(test_neural_data[:,vv].cpu().numpy(), \
                                    test_preds[:,vv].cpu().numpy())

# mask = dataset.functional_dict[str(args.subject_id[0])]['V1v'] + dataset.functional_dict[str(args.subject_id[0])]['V1d']
mask = dataset.functional_dict[str(args.subject_id[0])]['hV4']
noise_ceiling = dataset.noise_ceiling[str(args.subject_id[0])][mask] / 100
selected_voxels = noise_ceiling > 0.1
print(f"Selected {np.sum(selected_voxels)} voxels")
r2_voxels = r2_voxels[selected_voxels, :]

plt.figure()
plt.hist(r2_voxels)
plt.axvline(np.median(r2_voxels), color='k')
print("mean r2: ", np.mean(r2_voxels))
print("median r2: ", np.median(r2_voxels))
plt.title(f'R2 histogram')
plt.xlabel('R2')
plt.ylabel('Voxel count')
plt.savefig(f'/home/junruz/BrainDiVE/figure/hist_S1_layer4_V4_RN50_thresh0.1_bs128_e400_AdamW_lr0.0003_decay0.5_wd0.015_r2.png', dpi=300)
plt.close()



a, b = np.polyfit(noise_ceiling[selected_voxels], r2_voxels, 1)

plt.figure()
plt.scatter(noise_ceiling[selected_voxels], r2_voxels, s=5, )
plt.plot(noise_ceiling[selected_voxels], a*noise_ceiling[selected_voxels] + b, color='r', linestyle='--')
plt.xlabel('Noise Ceiling')
plt.ylabel('R2 per voxel')
plt.title('R2 vs Noise Ceiling per voxel')
plt.axline((0, 0), slope=1, color='k', linestyle='--')
plt.savefig(f'/home/junruz/BrainDiVE/figure/scatter_S1_layer4_V4_RN50_thresh0.1_bs128_e400_AdamW_lr0.0003_decay0.5_wd0.015_r2.png', dpi=300)
plt.close()
# brain_projector.r2 = r2_voxels
# torch.save(brain_projector, f"/home/junruz/BrainDiVE/weights/S1_weights_bs256_e600_AdamW_lr0.0003_decay0.5_wd0.015_r2.pth")
pass
