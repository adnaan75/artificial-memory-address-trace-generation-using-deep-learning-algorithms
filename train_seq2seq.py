import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import random
import math
import time
from models.seq2seq import Encoder, Decoder, Seq2Seq
from utils.load_data import ToTensor,  MemTraceDataset
from torchvision import transforms

WRAPPED_ADDRESS = 256


def train(model, train_loader, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, sample_batched in enumerate(train_loader):
        mem_trace = sample_batched["mem_trace"].cuda() % WRAPPED_ADDRESS
        target_reuse_dis = sample_batched["reuse_dis"].cuda()

        optimizer.zero_grad()

        output = model(mem_trace)
        output = output.permute(1, 0, 2)
        output = torch.reshape(
            output, (output.shape[0] * output.shape[1], WRAPPED_ADDRESS))

        # we do not need to predict the first address
        mem_trace = mem_trace[:, :, 1:].flatten()

        loss = criterion(output, mem_trace)

        loss.backward()

        pred = output.argmax(dim=1)
        acc = 100*pred.eq(mem_trace).sum().item()/mem_trace.shape[0]
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        print("loss: {} Acc: {}".format(loss, acc))

    return epoch_loss / len(train_loader)


if __name__ == "__main__":
    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    OUTPUT_DIM = WRAPPED_ADDRESS
    ENC_EMB_DIM = WRAPPED_ADDRESS
    DEC_EMB_DIM = WRAPPED_ADDRESS
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss()

    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float('inf')

    train_kwargs = {"batch_size": 32}

    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)

    train_memtrace_dataset = MemTraceDataset(reuse_dis_file="data_generator/train_data_label.log",
                                             mem_trace_dir="data_generator/train_data",
                                             transform=transforms.Compose([ToTensor()]))
    train_loader = torch.utils.data.DataLoader(
        train_memtrace_dataset, **train_kwargs)

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        # print("train loss: {}".format(train_loss))

