import torch
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from reuse_predictor import ReuseDisPredictor
from autoencoder import AutoEncoder
import numpy as np
import random

heatmap_dir = "../../train_data/npy_heatmap"
def init_sampling(heatmap):
    for x_axis in range(heatmap.shape[0]):
        sum = torch.sum(heatmap[x_axis])
        heatmap[x_axis]/=sum

def sampling(heatmap, x_axis):
    r = random.uniform(0, 1)
    c = 0
    for y_axis in range(heatmap.shape[1]):
        c+=heatmap[x_axis][y_axis]
        if c>r:
            return y_axis

def gen_trace(heatmap, output_file):
    # generate dummy trace
    heatmap = torch.transpose(torch.squeeze(heatmap),0,1)
    init_sampling(heatmap)
    with open(output_file,'w') as f:
        for idx in range(5000):
            f.write(hex(sampling(heatmap, idx%heatmap.shape[0])*4))
            f.write('\n')

def main():
    torch.manual_seed(42)
    # load pre-trained model
    encoder_decoder = AutoEncoder(n_channels=1).cuda().float()
    encoder_decoder.load_state_dict(torch.load('./ckpt/AE_epoch_10.pth'),strict=False)
    # open label log
    with open("../../train_data/label.log",'r') as f:
        for line in f.readlines():
            if line.find('log')>0:
                file_name = line.strip().replace("log", "npy")
                img_name = os.path.join(heatmap_dir, file_name)
                image = np.load(img_name)
                real_heatmap = torch.unsqueeze(torch.from_numpy(image), dim=0).float().cuda()
                real_heatmap = torch.unsqueeze(real_heatmap, dim=0)
                # generate dummy heatmap
                dummy_heatmap = encoder_decoder(real_heatmap)
                gen_trace(dummy_heatmap,'dummy_trace/{}'.format(file_name.replace('npy','log')))
                # exit(1)

if __name__ == "__main__":
    main()