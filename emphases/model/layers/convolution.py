import functools

import torch

import emphases


###############################################################################
# Convolution model
###############################################################################


class Convolution(torch.nn.Sequential):
    # TODO - add masking

    def __init__(self):
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

        super.__init__(*layers)
