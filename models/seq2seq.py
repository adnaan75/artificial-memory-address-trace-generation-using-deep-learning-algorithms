import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

import random
import math
import time


class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(src)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size, wrapped cache line]
        embedded = input.unsqueeze(0)
        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell

# Train an AutoEncoder model


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, teacher_forcing_ratio=0.5):
        wrapped_line_cnt = 256
        # src = [batch size, 1, mem trace length]
        batch_size = src.shape[0]
        trg_len = src.shape[2]
        trg_vocab_size = wrapped_line_cnt

        src = src.squeeze()  # src = [batch size, mem trace length]
        # convert cache line address in src, from [0, MEM_SIZE) to [0, emb_dim-1)
        src = src % wrapped_line_cnt
        embedded = F.one_hot(
            src, num_classes=wrapped_line_cnt).permute(1, 0, 2).float()
        # embedded: [mem trace length, batch size, wrapped_size]
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(embedded)

        # first input to the decoder is the <sos> tokens
        # as we train an AE model, the target is the input
        trg = embedded
        input = trg[0]

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len-1, batch_size,
                              trg_vocab_size).to(self.device)

        # input should be: 217, 6, 2
        for t in range(0, trg_len-1):
            # input: first round: 217, 242,  80, 171,  81, 228, 169, 119, 110...  32
            #        second round: 6, 245, 145, 145,  83, 229,   6, 119...
            #        third round: 2, 170,  13, 145,  92, 228, 169, 122, 245
            # input = trg[t]
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = F.one_hot(output.argmax(
                1), num_classes=wrapped_line_cnt).float()
            # #if teacher forcing, use actual next token as next input
            # #if not, use predicted token
            # input = trg[t+1] if teacher_force and t+1<trg_len else top1
            input = trg[t+1]

        return outputs
