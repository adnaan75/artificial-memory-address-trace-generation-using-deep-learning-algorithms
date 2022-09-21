import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from load_data import ToTensor, HeatmapDataset
from reuse_predictor import ReuseDisPredictor
from autoencoder import AutoEncoder
from dist_util import setup, cleanup, run_demo
from torch.nn.parallel import DistributedDataParallel as DDP


def train(encoder_decoder, predictor, train_loader, optimizer, epoch, loss_func):
    encoder_decoder.train()
    predictor.eval()
    cnt = 0

    diff_sum = 0
    diff_cnt = 0

    for batch_idx, sample_batched in enumerate(train_loader):
        data = sample_batched["heatmap"]
        target_reuse_dis = sample_batched["reuse_dis"]
        real_heatmap = data.cuda()
        target_reuse_dis = target_reuse_dis.cuda()
        optimizer.zero_grad()
        predictor.zero_grad()

        proxy_heatmap = encoder_decoder(real_heatmap)

        # get predictor miss rate of proxy heatmap
        proxy_reuse_dis = predictor(proxy_heatmap)

        with torch.no_grad():
            predict_real_reuse_dis = predictor(real_heatmap)

        loss = loss_func(proxy_reuse_dis, predict_real_reuse_dis)
        # not only predicted miss rate, but proxy benchmark should also has close memory reference
        # real # of memory reference
        real_memory_reference = real_heatmap.sum()
        proxy_memory_reference = proxy_heatmap.sum()
        loss += (torch.abs(real_memory_reference -
                 proxy_memory_reference)/real_memory_reference)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            encoder_decoder.parameters(), max_norm=2.0, norm_type=2)

        optimizer.step()

        if True:
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    predict_real_reuse_dis = predictor(real_heatmap)
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Real Error rate: {:.6f} Predict Error rate: {:.6f} Diff: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                        torch.mean(
                            torch.abs((proxy_reuse_dis-target_reuse_dis)/(target_reuse_dis+1))),
                        torch.mean(
                            torch.abs((predict_real_reuse_dis-target_reuse_dis)/(target_reuse_dis+1))),
                        torch.mean(
                            torch.abs((predict_real_reuse_dis-proxy_reuse_dis)/(predict_real_reuse_dis+1))),
                    )
                )
                diff_sum += torch.mean(
                    torch.abs((predict_real_reuse_dis-proxy_reuse_dis)/(predict_real_reuse_dis+1)))
                diff_cnt += 1
    print("batch diff: {}".format(diff_sum/diff_cnt))


def main():
    torch.manual_seed(42)

    train_kwargs = {"batch_size": 64}

    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)

    train_heatmap_dataset = HeatmapDataset(
        reuse_dis_file="../../train_data/4096_label.log",
        heatmap_dir="../../train_data/npy_heatmap",
        transform=transforms.Compose([ToTensor()]),
    )
    train_loader = torch.utils.data.DataLoader(
        train_heatmap_dataset, **train_kwargs)

    encoder_decoder = AutoEncoder(n_channels=1).cuda().float()
    reuse_dis_predictor = ReuseDisPredictor().cuda().float()
    reuse_dis_predictor.load_state_dict(torch.load(
        './ckpt/predictor_epoch_14.pth'), strict=False)

    loss_func = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(encoder_decoder.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, 11):
        train(encoder_decoder, reuse_dis_predictor,
              train_loader, optimizer, epoch, loss_func)
        torch.save(encoder_decoder.state_dict(),
                   './ckpt/AE_epoch_{}.pth'.format(epoch))
        scheduler.step()


if __name__ == "__main__":
    main()
