from __future__ import print_function
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from load_data import ToTensor, HeatmapDataset
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from reuse_predictor import ReuseDisPredictor
from dist_util import setup, cleanup, run_demo


def train(args, model, train_loader, loss_func, optimizer, epoch):
    model.train()
    cnt = 0
    for batch_idx, sample_batched in enumerate(train_loader):
        data = sample_batched["heatmap"]
        target_reuse_dis = sample_batched["reuse_dis"]
        import pdb
        pdb.set_trace()
        data = data.cuda()
        target_reuse_dis = target_reuse_dis.cuda()
        optimizer.zero_grad()

        output = model(data)
        loss = loss_func(output, target_reuse_dis)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=2.0, norm_type=2)

        optimizer.step()
        if True:
            if batch_idx % args.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Error rate: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                        torch.mean(
                            torch.abs((output-target_reuse_dis)/(target_reuse_dis+1))),
                    )
                )
                if args.dry_run:
                    break


def test(model, rank, test_loader):
    model.eval()
    error_rate = 0
    cnt = 0
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(test_loader):
            cnt += 1
            data = sample_batched["heatmap"]
            target_reuse_dis = sample_batched["reuse_dis"]
            data = data.to(rank)
            target_reuse_dis = target_reuse_dis.to(rank)
            output = model(data)
            import pdb
            pdb.set_trace()
            error_rate += torch.mean(torch.abs((output -
                                     target_reuse_dis)/target_reuse_dis+0.0001))

    error_rate /= cnt

    print("Test set: LLC Error-rate: {:.4f}".format(error_rate))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=16, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=14, metavar="N",
                        help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0,
                        metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7,
                        metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true",
                        default=False, help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true",
                        default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1,
                        metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true",
                        default=False, help="For Saving the current Model")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}

    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_heatmap_dataset = HeatmapDataset(
        reuse_dis_file="../../train_data/4096_label.log",
        heatmap_dir="../../train_data/npy_heatmap",
        transform=transforms.Compose([ToTensor()]),
    )
    train_loader = torch.utils.data.DataLoader(
        train_heatmap_dataset, **train_kwargs)

    model = ReuseDisPredictor().float().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    loss_func = nn.MSELoss().cuda()

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, loss_func, optimizer, epoch)
        torch.save(model.state_dict(),
                   './ckpt/predictor_epoch_{}.pth'.format(epoch))
        scheduler.step()


if __name__ == "__main__":
    main()
