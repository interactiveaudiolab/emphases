import functools

import torch

import emphases


###############################################################################
# Framewise model
###############################################################################


class Framewise(torch.nn.Sequential):

    def __init__(
        self,
        input_channels=emphases.NUM_FEATURES,
        output_channels=1,
        hidden_channels=128,
        kernel_size=5):
        conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=kernel_size,
            padding='same')
        super().__init__(
            conv_fn(input_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, output_channels))

    def forward(self, features, *_):
        return super().forward(features)
