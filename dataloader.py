import os
import numpy as np
from torch.utils.data import Dataset
import torch
import glob
import pdb
from PIL import Image
from mpdb import mpdb
from train_utils import calculate_parameter_error
import random
import matplotlib.pyplot as plt

def filter_invalid_index(data_list, param_array, invalid_idx):
    if invalid_idx.shape[0] == 0:
        return data_list, param_array
    all_idx = np.array([int((data_path.split('/')[-1].split('.')[0]).split('_')[-1]) for data_path in data_list])
    # all_idx = np.array([int(data_path.split('/')[-1].split('.')[0]) for data_path in data_list])
    invalid_mask = (all_idx[:,None] == invalid_idx[None,:]).sum(axis = -1) > 0
    data_list = np.array(data_list)[~invalid_mask].tolist()
    print(param_array[invalid_mask])
    param_array = param_array[~invalid_mask]
    return data_list, param_array

def crop(traj, Tsidx, Tlen):
    assert Tsidx.shape[0] == traj.shape[0]
    traj_list = [traj[i, Tsidx[i]:Tsidx[i] + Tlen] for i in range(Tsidx.shape[0])]
    return torch.stack(traj_list)

@torch.no_grad()
def random_cropping(traj, crop_T):
    T = traj.shape[1]
    assert T >= crop_T
    Tsidx = torch.randint(T - crop_T, (traj.shape[0],1))[:,0]
    train_crop = crop(traj, Tsidx, crop_T)
    return train_crop

crop_T = 500
K = 36
train_size = 6000

class ValidationData(Dataset):
    def __init__(self, train_size = 500, validation_size = 100):
        self.data_path = '{}/uniform_large'.format(self.folder_path)
        self.data_list = glob.glob('{}/one_pos_traj_symm_{}_{}_*_stricter.tiff'.format(self.data_path, train_size, dist_index))
        self.data_list = self.data_list[:train_size]
        self.data_list = self.data_list[:validation_size]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_1d_recon = np.array(Image.open(self.data_list[idx])).reshape(-1)
        anchor_param, anchor_t, pos_param, pos_t, pos_dist = data_1d_recon[:4], data_1d_recon[4: 4 + 1000 * 396].reshape(1000, 396), \
                                                                data_1d_recon[4 + 1000 * 396: 4 + 1000 * 396 + 4], \
                                                                data_1d_recon[4 + 1000 * 396 + 4 : 4 + 1000 * 396 + 4 + 1000 * 396].reshape(1000, 396), \
                                                                data_1d_recon[-1:]

        return anchor_param, anchor_t


class TrainingData(Dataset):

    def __init__(self, crop_T, args, num_of_pos = 1, dist_index = 1, \
                filter = False, train_size = 10000):
        self.crop_T = crop_T
        self.augpos_scale = args.augpos_scale
        self.augpos_prob = args.augpos_prob
        self.augpos_threshold = args.augpos_threshold
        self.num_of_pos = num_of_pos

        self.folder_path = '/net/scratch/roxie62/emulator'
        self.data_path = '{}/uniform_training_data'.format(self.folder_path)

        self.data_list = glob.glob('{}/new_{}_{}_*_stricter.tiff'.format(self.data_path, train_size, dist_index))
        self.data_list.sort()
        self.params_array = torch.load('{}/training_params.pth'.format(self.data_path))

        nan_idx_array = torch.load('{}/nan_idx_stricter_5e05.pth'.format(self.data_path))
        self.params_array = self.params_array[:len(self.data_list)]
        self.data_list, self.params_array = filter_invalid_index(self.data_list, self.params_array, nan_idx_array)

        self.data_list = self.data_list[:train_size]
        self.params_array = self.params_array[:train_size]
        self.dist_index = dist_index
        print('min', self.params_array.min(axis = 0))
        print('max', self.params_array.max(axis = 0))
        print('the size of the training data is:', len(self.data_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_1d_recon = np.array(Image.open(self.data_list[idx])).reshape(-1)
        anchor_param, anchor_t, pos_param, pos_t, pos_dist = data_1d_recon[:4], data_1d_recon[4: 4 + 1000 * 396].reshape(1000, 396), \
                                                                data_1d_recon[4 + 1000 * 396: 4 + 1000 * 396 + 4], \
                                                                data_1d_recon[4 + 1000 * 396 + 4 : 4 + 1000 * 396 + 4 + 1000 * 396].reshape(1000, 396), \
                                                                data_1d_recon[-1:]
        filter = pos_dist > self.augpos_threshold
        anchor_param = torch.from_numpy(anchor_param)
        pos_param = torch.from_numpy(pos_param)
        if filter:
            filter_params_mask = (torch.rand(anchor_param.shape) >= self.augpos_prob).float()
            random_noise = torch.randn(anchor_param.shape).clamp(min = -3, max = 3)
            pos_param = anchor_param + abs(anchor_param) * self.augpos_scale * filter_params_mask * random_noise
            pos_t = anchor_t
        return anchor_param, random_cropping(torch.from_numpy(anchor_t[None, :, :]), self.crop_T)[0], pos_param, random_cropping(torch.from_numpy(pos_t[None, :, :]), self.crop_T)[0], idx, filter
