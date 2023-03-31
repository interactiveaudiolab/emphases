import functools

import torch

import emphases


###############################################################################
# Convolution model
###############################################################################


class Convolution(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # Bind common parameters
        conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=emphases.KERNEL_SIZE,
            padding='same')

        # Hidden layers
        layers = []
        channels = emphases.CHANNELS
        for _ in range(emphases.LAYERS - 1):
            layers.extend((conv_fn(channels, channels), torch.nn.ReLU()))

        # Output layer
        layers.append(conv_fn(channels, 1))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, lengths):
        mask = emphases.model.mask_from_lengths(lengths)
        return self.layers(x) * mask, mask
