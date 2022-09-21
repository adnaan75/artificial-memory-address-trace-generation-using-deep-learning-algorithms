import torch
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from reuse_predictor import ReuseDisPredictor
from autoencoder import AutoEncoder
import numpy as np

heatmap_dir = "../../train_data/npy_heatmap"


def main():
    torch.manual_seed(42)
    # load pre-trained model
    encoder_decoder = AutoEncoder(n_channels=1).cuda().float()
    encoder_decoder.load_state_dict(torch.load(
        './ckpt/AE_epoch_10.pth'), strict=False)
    # open label log
    with open("../../train_data/4096_label.log", 'r') as f:
        for line in f.readlines():
            if line.find('log') > 0:
                file_name = line.strip().replace("log", "npy")
                img_name = os.path.join(heatmap_dir, file_name)
                image = np.load(img_name)
                real_heatmap = torch.unsqueeze(
                    torch.from_numpy(image), dim=0).float().cuda()
                real_heatmap = torch.unsqueeze(real_heatmap, dim=0)
                # generate dummy heatmap
                dummy_heatmap = encoder_decoder(real_heatmap)
                # visualize real heatmap
                image = (image/image.max())*255
                image = 255-image
                image = image.astype(np.int8)
                heatmap_figure = Image.fromarray(np.asarray(image))
                heatmap_figure = heatmap_figure.convert("L")
                heatmap_name = file_name.replace('npy', 'png')
                heatmap_figure.save(
                    "new_visualization/{}".format(heatmap_name))
                # visualize dummy heatmap
                dummy_heatmap = torch.squeeze(
                    dummy_heatmap).detach().cpu().numpy()
                dummy_heatmap = (dummy_heatmap/dummy_heatmap.max())*255
                dummy_heatmap = 255-dummy_heatmap
                dummy_heatmap = dummy_heatmap.astype(np.int8)
                heatmap_figure = Image.fromarray(np.asarray(dummy_heatmap))
                heatmap_figure = heatmap_figure.convert("L")
                heatmap_name = file_name.replace('.npy', '_gen.png')
                heatmap_figure.save(
                    "new_visualization/{}".format(heatmap_name))


if __name__ == "__main__":
    main()
