import math

import torch

import emphases


###############################################################################
# Transformer stack
###############################################################################


class Transformer(torch.nn.Module):

    def __init__(self, num_layers=emphases.LAYERS, channels=emphases.CHANNELS):
        super().__init__()
        self.position = PositionalEncoding(channels, .1)
        self.model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                channels,
                2,
                dim_feedforward=emphases.CHANNELS),
            num_layers)

    def forward(self, x, lengths):
        mask = emphases.model.mask_from_lengths(lengths)
        return self.model(
            self.position(x.permute(2, 0, 1)),
            src_key_padding_mask=~mask.squeeze(1)
        ).permute(1, 2, 0)


###############################################################################
# Utilities
###############################################################################


class PositionalEncoding(torch.nn.Module):

    def __init__(self, channels, dropout=.1, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        index = torch.arange(max_len).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, channels, 2) * (-math.log(10000.0) / channels))
        encoding = torch.zeros(max_len, 1, channels)
        encoding[:, 0, 0::2] = torch.sin(index * frequency)
        encoding[:, 0, 1::2] = torch.cos(index * frequency)
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        return self.dropout(x + self.encoding[:x.size(0)])
