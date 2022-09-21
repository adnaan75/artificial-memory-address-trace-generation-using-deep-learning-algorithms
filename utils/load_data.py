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
        mem_trace, reuse_dis = sample["mem_trace"], sample["reuse_dis"]
        return {
            # divide mem address by 16, to convert to cache line address
            "mem_trace": (torch.unsqueeze(torch.as_tensor(mem_trace, dtype=torch.int64), dim=0)).long(),
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
                label_list.append(line.strip().split('/')[-1])

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
            elif len(line)==1:
                continue
            else:
                curr_cache_line_tuple.append(int(line.split()[-1]))
    return reuse_dis_list, label_list


class MemTraceDataset(Dataset):
    def __init__(self, reuse_dis_file, mem_trace_dir, transform=None):
        """
        Args:
            reuse_dis_file (string): Path to the reuse distance log.
            mem_trace_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.reuse_dis_list, self.label_list = read_reuse_dis(reuse_dis_file)
        self.mem_trace_dir = mem_trace_dir
        self.transform = transform

    def __len__(self):
        return len(self.reuse_dis_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace_path = os.path.join(self.mem_trace_dir, self.label_list[idx])
        mem_trace = []
        with open(trace_path,'r') as f:
            for line in f.readlines():
                # convert mem address to cache line address
                mem_trace.append(int(line, 16)/16)

        sample = {"mem_trace": mem_trace, "reuse_dis": self.reuse_dis_list[idx]}

        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    memtrace_dataset = MemTraceDataset(reuse_dis_file="data_generator/train_data_label.log", \
                                    mem_trace_dir="data_generator/train_data",\
                                    transform=transforms.Compose([ToTensor()]))

    fig = plt.figure()

    for i in range(len(memtrace_dataset)):
        sample = memtrace_dataset[i]
        print(i, sample["reuse_dis"])
        if i == 5:
            break