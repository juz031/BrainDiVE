# import basic modules
import sys
import os
import time
import numpy as np
import h5py
import pandas as pd
import scipy
import torch
import matplotlib
import matplotlib.pyplot as plt
import warnings
import pickle
import argparse, gc
from skimage.transform import resize
import random
import time
from tqdm import tqdm

import torch.multiprocessing as mp
import os
import socket
from torch.cuda.amp import GradScaler
from contextlib import closing
import torch.distributed as dist
# import model
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import math
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

sys.path.insert(0,'/lab_data/hendersonlab/code/utils/')
import stats_utils
import prf_utils

# from options import Options
import functools
import random
from torch import autocast
# from eva_clip import create_model_and_transforms, get_tokenizer
import clip
# import timm
torch.backends.cudnn.benchmark=True

# Stuff from data_loader_224
area = 'hV4'
n_voxel = 1187 # s1: [v1;1350, V2:1433, V3:1187, hV4:687]
layer = 'layer3'

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

def create_pRF(angle, eccentricity, size, exponent, n_pix):
    # Converting them into [x, y] coordinates, from polar
    # The units here are degrees visual angle
    x_mapping, y_mapping = prf_utils.pol_to_cart(angle, eccentricity)
    x_mapping = np.minimum(np.maximum(x_mapping, -7), 7)
    # S = pRF size, sigma of the Gaussian function.
    y_mapping = np.minimum(np.maximum(y_mapping, -7), 7)
    s_mapping = np.minimum(size, 8.4)
    x = x_mapping[vv] / 8.4
    y = y_mapping[vv] / 8.4
    sigma = s_mapping[vv] / 8.4
    
    prf_2d = prf_utils.gauss_2d(center=[x,y], sd=sigma, patch_size=n_pix)

    return prf_2d


class neural_loader_with_pRF(torch.utils.data.Dataset):
    def __init__(self, arg_stuff):

        # NOTE this is really meant to be used for one subject.
        # It looks like it would accept a list of subs, but will break for > 1.
        
        self.subject_id = arg_stuff.subject_id
        assert(len(self.subject_id)==1)
        if isinstance(self.subject_id, int):
            self.subject_id = list([self.subject_id])
            
        self.neural_activity_path = arg_stuff.neural_activity_path
        self.image_path = arg_stuff.image_path
        # self.image_data = None

        self.voxel_id = arg_stuff.voxel_id
        
        self.roi_path = arg_stuff.roi_path
        self.pRF_path = arg_stuff.pRF_path
        
        self.transform = normalize_image

        self.all_keys = dict() # Maps subject id to valid COCO_ids
        self.num_stimulus = dict() # Maps subject id to number of stimulus
        self.neural_sizes = dict() # Maps subject id to number of voxels
        self.roi_info = dict()
        self.pRF_info = dict()
        self.noise_ceiling = dict()
        self.double_mask = dict()
        self.functional_dict = dict()

        
        self.stim_keys_path = arg_stuff.stim_keys_path

        # Load my dictionary: this has a list of saved keys for all our subjects.
        # Key = COCO ID
        with open(self.stim_keys_path, "rb") as dict_saver:
            all_keys = pickle.load(dict_saver)

        # Define testing set
        # These are coco ids that overlap across all 8 subjects. There are 907 of these.
        testing_set = set.intersection(*[set(_) for _ in list(all_keys.values())]) 
        self.testing_set = sorted(testing_set)
        
        self.complete_keys = all_keys

        # Now figuring out some things specific to each subject...
        for subject in self.subject_id:
            str_subject = str(subject)
            neural_data = h5py.File(self.neural_activity_path.format(str_subject), 'r')

            # all_keys: this is just the TRAINING set, not testing.
            # For the subjects with all trials (1,2,5,7), this will have: 10000 - 907 = 9093 elements
            self.all_keys[str_subject] = [i for i in list(neural_data.keys()) if ((not "mask" == i) and (not i in testing_set))] # What is "mask": no mask in this data

            # Number of training stimuli
            self.num_stimulus[str_subject] = len(self.all_keys[str_subject])

            # This is a dict of the info about ROI defs for this subject
            # This can be used to get masks for your desired areas.
            self.roi_info[str_subject] = np.load(self.roi_path.format(str_subject), allow_pickle=True).item()
            self.noise_ceiling[str_subject] = self.roi_info[str_subject]['noise_ceiling_avgreps']

            pRF_info = np.load(self.pRF_path.format(str_subject), allow_pickle=True).item()
            self.pRF_info[str_subject] = [pRF_info[key][self.voxel_id] for key in pRF_info.keys()]

            # Number of voxels we have here
            self.neural_sizes[str_subject] = np.sum(self.roi_info[str_subject]['voxel_mask']) # voxel mask: choose all valid brain voxels we wanted (~700000 to 19738) 
            self.double_mask[str_subject] = np.ones(self.neural_sizes[str_subject],dtype=bool) # What is double mask?
            
            roi_info = self.roi_info[str_subject]
            big_mask = roi_info['voxel_mask']
            roi_keys = ['roi_labels_retino'] #['roi_labels_retino', 'roi_labels_kastner', 'roi_labels_face', 'roi_labels_place', 'roi_labels_body']
            roi_names = ['ret_prf_roi_names'] #['ret_prf_roi_names', 'kastner_atlas_roi_names', 'floc_face_roi_names', 'floc_place_roi_names', 'floc_body_roi_names']
            self.functional_dict[str_subject] = dict()
            for key, name in zip(roi_keys, roi_names):
                roi_labels = roi_info[key][big_mask]
                roi_name = roi_info[name]
                for name in roi_name.keys():
                    self.functional_dict[str_subject][name] = roi_labels==roi_name[name]
            # apply voxel mask to neural data and roi mask to match voxel space

            neural_data.close()
            neural_data = None
            gc.collect()
            # Pytorch will fail if you try to use multiprocessing with an open h5py
            # Zero it out
            
            setattr(self, "subj_{}_neural_data".format(str_subject), None) # Why set to None?
            setattr(self, "subj_{}_image_data".format(str_subject), None)
            
        self.all_subjects = sorted(list(self.all_keys.keys()))

    def __len__(self):
        
        # This is TRAINING set length
        if len(self.all_subjects)==1:
            return list(self.num_stimulus.values())[0]
        print("multi subject case", max(list(self.num_stimulus.values())))
        return max(list(self.num_stimulus.values()))

    
    def __getitem__(self, idx):
        # return a dictionary with keys: 'subject_id', 'neural_data', 'image_data'
        # 'neural data': neural data for single subject, shape: (n_voxels, )
   
        # this is how we get TRAINING set
        
        all_images = []
        all_neural = []
        for subject_idx in self.all_subjects:
            
            # mask = self.double_mask[subject_idx]
            if area != 'hV4':
                mask = self.functional_dict[subject_idx][f'{area}v'] +self.functional_dict[subject_idx][f'{area}d']
            else:
                mask = self.functional_dict[subject_idx][f'{area}']
            # mask = self.functional_dict[subject_idx][f'{area}']

            # if neural and image data already loaded, use that variable
            subject_neural_h5py = getattr(self, "subj_{}_neural_data".format(subject_idx))
            subject_image_h5py = getattr(self, "subj_{}_image_data".format(subject_idx))

            # otherwise load them here...
            if subject_neural_h5py is None:
                # print('Loading neural data: %s'%self.neural_activity_path.format(subject_idx))
                subject_neural_h5py = h5py.File(self.neural_activity_path.format(subject_idx), 'r')
            else:
                pass

            if subject_image_h5py is None:
                # print('Loading image data: %s'%self.image_path.format(subject_idx))
                subject_image_h5py = h5py.File(self.image_path.format(subject_idx), 'r') # only open the file, notnloading the file to the memory
            else:
                pass
                
            # print(len(subject_neural_h5py), subject_idx)

            # Choosing which index to load 
            # print(idx, self.num_stimulus[subject_idx])
            assert idx <= (self.num_stimulus[subject_idx]-1)
            curidx = idx

            
            # this bottom part is if you had more subs with different numbers of trials.
            # if idx > (self.num_stimulus[subject_idx]-1):
            #     curidx = random.randint(0, self.num_stimulus[subject_idx]-1)
            # else:
            #     curidx = idx

            # Get the COCO ID for this image. It is a key into my h5py file
            neural_key = self.all_keys[subject_idx][curidx]

            # print([idx, neural_key])
            
            # apply ROI mask here. can be nothing if mask is all ones
            selected_neural = subject_neural_h5py[neural_key][:][mask] # get real data and load to memory
            
            selected_image = subject_image_h5py[neural_key][:]

            # print(f'Selected neural: {selected_neural.shape}')
            # print(f'Selected image: {selected_image.shape}')
            
            if not (self.transform is None):
                selected_image = self.transform(selected_image)
            else:
                assert False
                
            all_images.append(np.copy(selected_image))
            all_neural.append(np.copy(selected_neural))

        all_neural = np.concatenate(all_neural)
        
        return_subjects = np.array([int(x) for x in self.all_subjects])

        # print(f'All neural: {all_neural.shape}')
        # print('returning')
        
        return {"subject_id":torch.from_numpy(return_subjects), \
                "neural_data": torch.from_numpy(all_neural), \
                "image_data": torch.from_numpy(np.array(all_images))}

    def get_item_test(self, idx):

        # This is how we get TESTING set
        
        all_images = []
        all_neural = []
        for subject_idx in self.all_subjects:
            # mask = self.double_mask[subject_idx]
            if area != 'hV4':
                mask = self.functional_dict[subject_idx][f'{area}v'] +self.functional_dict[subject_idx][f'{area}d']
            else:
                mask = self.functional_dict[subject_idx][f'{area}']

            subject_neural_h5py = getattr(self, "subj_{}_neural_data".format(subject_idx))
            subject_image_h5py = getattr(self, "subj_{}_image_data".format(subject_idx))

            if subject_neural_h5py is None:
                subject_neural_h5py = h5py.File(self.neural_activity_path.format(subject_idx), 'r')
            else:
                pass

            if subject_image_h5py is None:
                subject_image_h5py = h5py.File(self.image_path.format(subject_idx), 'r')
            else:
                pass
                
            assert idx <= (len(self.testing_set) - 1)
            curidx = idx
            neural_key = self.testing_set[curidx]

            # print([idx, neural_key])
            
            selected_neural = subject_neural_h5py[neural_key][:][mask]
            
            selected_image = subject_image_h5py[neural_key][:]
            selected_image = normalize_image_deterministic(selected_image)
            
            all_images.append(np.copy(selected_image))
            all_neural.append(np.copy(selected_neural))
            
        all_neural = np.concatenate(all_neural)
        
        return_subjects = np.array([int(x) for x in self.all_subjects])
        
        return {"subject_id": torch.from_numpy(return_subjects), \
                "neural_data": torch.from_numpy(all_neural), \
                "image_data": torch.from_numpy(np.array(all_images))}


# from BrainSCUBA / model.py
# this is just the linear projection layer. CLIP to voxel space.
class soft_quantizer_v3(torch.nn.Module):
    def __init__(self, num_higher_output=1000, num_prototypes=4, embed_dim=512, clip_offset=5.0, max_norm=10.0):
        super().__init__()
        self.linear = torch.nn.Linear(embed_dim, num_higher_output)
        self.r2 = np.zeros((num_higher_output,1))
    def forward(self, image_vectors):
        # image_vectors should be pre_normalized
        return self.linear(image_vectors)


def shuffle_shift(input_image, extent=4):
    offset_x = random.randint(-extent, extent)
    offset_y = random.randint(-extent, extent)
    orig_shape = input_image.shape
    temp = input_image[:,:, max(0,offset_x):min(orig_shape[2], orig_shape[2]+offset_x), max(0,offset_y):min(orig_shape[3], orig_shape[3]+offset_y)]
#     temp = torch.nn.functional.pad(temp, (max(0,offset_y), max(0, -offset_y), max(0,offset_x), max(0, -offset_x)), mode='replicate')
    temp = torch.nn.functional.pad(temp, (max(0, -offset_y),max(0,offset_y), max(0, -offset_x), max(0,offset_x)), mode='replicate')
    return temp


if __name__ == "__main__":
    # Set some paths: where the preprocessed NSD files live
    nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
    data_folder = os.path.join(nsd_path, 'data')
    labels_folder = os.path.join(nsd_path, 'labels')
    stim_folder = os.path.join(nsd_path, 'stimuli')
    rois_folder = os.path.join(nsd_path, 'rois')

    n_pix = 224

    # Create namespace and set attributes
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', nargs='+', default=[1], type=int)
    parser.add_argument('--voxel_id', type=int, default=0)
    parser.add_argument('--neural_activity_path', type=str, default=os.path.join(data_folder, 'S{}_betas_avg_bigmask_dict.hdf5'))
    parser.add_argument('--image_path', type=str, default=os.path.join(stim_folder, 'S{}_stimuli_%d_dict.hdf5'%n_pix))
    parser.add_argument('--roi_path', type=str, default=os.path.join(rois_folder, 'S{}_voxel_roi_info.npy'))
    parser.add_argument('--pRF_path', type=str, default=os.path.join(rois_folder, 'S{}_prf_params.npy'))
    parser.add_argument('--stim_keys_path', type=str, default=os.path.join(stim_folder, "all_keys.pkl"))
    parser.add_argument('--optimizer_name', type=str, default="AdamW")
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr_init', type=float, default=3e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-1)
    parser.add_argument('--weight_decay', type=float, default=1.5e-2)
    args = parser.parse_args()
    # args = parser.parse_args()
    # args = argparse.Namespace()
    # args.subject_id = [1]
    # args.neural_activity_path = os.path.join(data_folder, 'S{}_betas_avg_bigmask_dict.hdf5')
    # args.image_path = os.path.join(stim_folder, 'S{}_stimuli_%d_dict.hdf5'%n_pix)
    # args.roi_path = os.path.join(rois_folder, 'S{}_voxel_roi_info.npy')
    # args.stim_keys_path = os.path.join(stim_folder, "all_keys.pkl")
    
    # args.optimizer_name = "Adam"
    # args.epochs = 400
    # args.batch_size = 256
    # args.lr_init = 3e-4
    # args.lr_decay = 5e-1
    # args.weight_decay=1.5e-2

    loader = neural_loader_with_pRF(args)
    # n_voxels = loader.neural_sizes[str(args.subject_id[0])]

    # make dataloader
    neural_train_loader = torch.utils.data.DataLoader(loader, \
                                                batch_size=args.batch_size, \
                                                shuffle=False, \
                                                num_workers=0, \
                                                drop_last=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

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

    
    projector = soft_quantizer_v3(num_higher_output=n_voxel, embed_dim=1024*7*7)
    projector = projector.to(device)
    projector.train()
    criterion = torch.nn.MSELoss()

    # SGD
    if args.optimizer_name == "SGD":
        optimizer = torch.optim.SGD(projector.parameters(), \
                                    lr=args.lr_init, \
                                    weight_decay=args.weight_decay) 
    elif args.optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(projector.parameters(), \
                                    lr=args.lr_init, \
                                    weight_decay=args.weight_decay) 
    elif args.optimizer_name == "Adam":
        optimizer = torch.optim.Adam(projector.parameters(), \
                                    lr=args.lr_init, \
                                    weight_decay=args.weight_decay) 

    # AdamW
    # optimizer = torch.optim.AdamW(projector.parameters(), \
    #                             lr=args.lr_init, \
    #                             weight_decay=args.weight_decay) 
    # weight decay here: this implements L2 penalty, like ridge

    # loop over epochs: in each epoch, we're passing over whole training set.
    # i am only doing one for testing this...
    start_epoch = 0
    max_steps = len(neural_train_loader) 
    # max_steps = 5; # for debugging, stop early
    
    print(f"Training for {args.epochs} epochs...")
    for epoch in tqdm(range(start_epoch, args.epochs)):

        # adjust my learning rate, according to decay function
        decay_rate = args.lr_decay
        new_lrate = args.lr_init * (decay_rate ** (epoch / args.epochs))

        # setting new learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # keeping track of losses, throughout my epoch
        total_losses = 0
        cur_iter = 0

        # train_sampler.set_epoch(epoch)
        # doing steps over dataloader now

        for step, data_stuff in enumerate(neural_train_loader):

            st = time.time()
            
            # if np.mod(step, 50)==0:
            #     # print(step, max_steps)
            if step > max_steps:
                # print(step)
                break
                
            neural_data = data_stuff["neural_data"].to(device)
            image_data = data_stuff["image_data"][:,0,:,:,:].to(device) # the 1st dim is zero here

            # print(neural_data.shape)
            # print(image_data.shape)
            
            # neural_data = data_stuff["neural_data"].to(output_device, non_blocking=True) # Flat tensor already
            # image_data = data_stuff["image_data"][:,0].to(output_device, non_blocking=True) # collapse along batch
            
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                # features = visual_encoder.encode_image(image_data)
                features = feature_extractor(image_data)
                features = features[layer].float()
                features = torch.nn.functional.avg_pool2d(features, kernel_size=2, stride=2)
                features = features.view(features.size(0), -1)
                features = features/(features.norm(dim=-1, keepdim=True)+1e-10)
            predicted = projector(features.float())
            
            loss = criterion(predicted, neural_data)
                
            loss.backward()
            optimizer.step()

            elapsed = time.time() - st
            # if np.mod(step, 50)==0:
            #     print('step took: %.5f s'%elapsed)
            
        print(f"Loss: {loss.item()}")
        # Gather the trained model parameters
        w_best = projector.linear.weight
        b_best = projector.linear.bias

        torch.save(projector.state_dict(), f'weights/S{args.subject_id[0]}_{layer}_{area}_{model_name}_bs{args.batch_size}_e{args.epochs}_{args.optimizer_name}_lr{args.lr_init}_decay{args.lr_decay}_wd{args.weight_decay}.pth')

        # Gather the test set / features
    # I'm doing predictions with the linear layer in this loop.
    # could be a better way...

    test_neural_data = [] # shape: (test_set_length (907), n_voxels)
    # test_image_data = []
    test_features = []

    test_preds = []

    # Get the test set length
    test_set_length = len(loader.testing_set)
    print(f"Extracting {test_set_length} test samples...")

    for idx in tqdm(range(test_set_length)):
        test_item = loader.get_item_test(idx)
        
        test_neural_data.append(test_item['neural_data'].detach().to('cpu'))
        test_image_data = test_item['image_data']
        
        # get CLIP embedding one image at a time.
        # feat = visual_encoder.encode_image(test_image_data.to(device)).float()
        feat = feature_extractor(test_image_data.to(device))
        feat = feat[layer].float()
        feat = torch.nn.functional.avg_pool2d(feat, kernel_size=2, stride=2)
        feat = feat.view(feat.size(0), -1)
        feat = feat/feat.norm(dim=-1, keepdim=True)
        # make prediction for held-out data here
        # print(feat.shape, w_best.shape, b_best.shape)
        # print(feat.shape, w_best.T.shape, b_best[None,:].shape)

        # run forward model: predict voxel response
        y_pred = feat @ w_best.T + b_best[None,:]
        y_pred = y_pred.detach().cpu()
        # print(y_pred.shape)

        test_preds.append(y_pred)
        # test_features.append(visual_encoder.encode_image(test_image_data.to(device)))

        del feat, test_item, y_pred
        torch.cuda.empty_cache()  # Clear GPU cache
        

    # Concatenate all items
    test_neural_data = torch.stack(test_neural_data, dim=0).to(device)  # Shape: [test_length, neural_features]
    # test_image_data = torch.stack(test_image_data, dim=0)    # Shape: [test_length, num_subjects, channels, height, width]
    # test_image_data = test_image_data[:,0,:,:,:].to(device)
    test_preds = torch.stack(test_preds, dim=0)[:,0,:].to(device)

    print(f"Test set shapes:")
    print(test_neural_data.shape)
    # print(test_image_data.shape)
    print(test_preds.shape)

    # get R2 per voxel
    r2_voxels = np.zeros((n_voxel,1))
    for vv in range(n_voxel):    
        r2_voxels[vv] = stats_utils.get_r2(test_neural_data[:,vv].cpu().numpy(), \
                                        test_preds[:,vv].cpu().numpy())
    
    plt.figure()
    plt.hist(r2_voxels)
    plt.axvline(np.median(r2_voxels), color='k')
    plt.title(f'R2 histogram')
    plt.xlabel('R2')
    plt.ylabel('Voxel count')
    plt.savefig(f'/home/junruz/BrainDiVE/figure/hist_S1_{layer}_{area}_RN50_bs128_e400_AdamW_lr0.0003_decay0.5_wd0.015_r2.png', dpi=300)
    plt.close()

    if area != 'hV4':
        mask = loader.functional_dict[str(args.subject_id[0])][f'{area}v'] + loader.functional_dict[str(args.subject_id[0])][f'{area}d']
    else:
        mask = loader.functional_dict[str(args.subject_id[0])][f'{area}']
    noise_ceiling = loader.noise_ceiling[str(args.subject_id[0])][mask] / 100

    a, b = np.polyfit(noise_ceiling, r2_voxels, 1)

    plt.figure()
    plt.scatter(noise_ceiling, r2_voxels, s=5, )
    plt.plot(noise_ceiling, a*noise_ceiling + b, color='r', linestyle='--')
    plt.xlabel('Noise Ceiling')
    plt.ylabel('R2 per voxel')
    plt.title('R2 vs Noise Ceiling per voxel')
    plt.axline((0, 0), slope=1, color='k', linestyle='--')
    plt.savefig(f'/home/junruz/BrainDiVE/figure/scatter_S1_{layer}_{area}_RN50_bs128_e400_AdamW_lr0.0003_decay0.5_wd0.015_r2.png', dpi=300)
    plt.close()