from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, reuse_dis = sample["heatmap"], sample["reuse_dis"]
        return {
            "heatmap": torch.unsqueeze(torch.from_numpy(image), dim=0).float(),
            "reuse_dis": torch.unsqueeze(torch.as_tensor(reuse_dis, dtype=torch.float32), dim=0),
        }


def read_reuse_dis(reuse_dis_file):
    reuse_dis_list = []
    label_list = []
    curr_tuple = []
    curr_cache_line_tuple = []
    with open(reuse_dis_file, "r") as f:
        curr_benchmark = None
        for line in f.readlines():
            if line.strip().find('log')>0:
                label_list.append(line.strip().split('.')[0]+'.npy')

                if len(curr_cache_line_tuple)>0:
                    curr_tuple.append(curr_cache_line_tuple)
                curr_cache_line_tuple = []
                if len(curr_tuple)>0:
                    reuse_dis_list.append(curr_tuple)
                curr_tuple = []
            elif line.find('line')>0:
                if len(curr_cache_line_tuple)>0:
                    curr_tuple.append(curr_cache_line_tuple)
                curr_cache_line_tuple = []
            else:
                curr_cache_line_tuple.append(int(line.split()[-1]))
    return reuse_dis_list, label_list


class HeatmapDataset(Dataset):
    def __init__(self, reuse_dis_file, heatmap_dir, transform=None):
        """
        Args:
            reuse_dis_file (string): Path to the reuse distance log.
            heatmap_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.reuse_dis_list, self.label_list = read_reuse_dis(reuse_dis_file)
        self.heatmap_dir = heatmap_dir
        self.transform = transform

    def __len__(self):
        return len(self.reuse_dis_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.heatmap_dir, self.label_list[idx])
        image = np.load(img_name)
        sample = {"heatmap": image, "reuse_dis": self.reuse_dis_list[idx]}

        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    heatmap_dataset = HeatmapDataset(reuse_dis_file="../train_data/label.log", \
                                    heatmap_dir="../train_data/npy_heatmap",\
                                    transform=transforms.Compose([ToTensor()]))

    fig = plt.figure()

    for i in range(len(heatmap_dataset)):
        sample = heatmap_dataset[i]
        print(i, sample["reuse_dis"].shape)
        if i == 5:
            break