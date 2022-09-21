import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
cache_line_size_cnt = 5
reuse_dis_hist = 4

class ReuseDisPredictor(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(n_layers * hid_dim, 1024)
        self.fc2 = nn.Linear(1024, cache_line_size_cnt*reuse_dis_hist)
        
    def forward(self, src):
        src = src.squeeze()
        # convert cache line address in src, from [0, MEM_SIZE) to [0, emb_dim-1)
        src=src%self.emb_dim

        embedded = F.one_hot(src, num_classes=self.hid_dim).permute(1,0,2).float()
        #embedded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        hidden = hidden.permute(1,0,2)
        hidden = torch.flatten(hidden,1)

        x = self.fc1(hidden)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.relu(x)

        output = output.view(-1,1,cache_line_size_cnt,reuse_dis_hist)
        return output
