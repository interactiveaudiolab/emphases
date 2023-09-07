import functools

import torch

import emphases


###############################################################################
# Convolution model
###############################################################################


class Convolution(torch.nn.Sequential):

    def __init__(self, kernel_size=emphases.ENCODER_KERNEL_SIZE):
        # Bind common parameters
        conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=kernel_size,
            padding='same')

        # Layers
        layers = []
        channels = emphases.CHANNELS
        for _ in range(emphases.LAYERS):
            layers.extend((
                conv_fn(channels, channels),
                emphases.ACTIVATION_FUNCTION()))
            if emphases.DROPOUT is not None:
                layers.append(torch.nn.Dropout(emphases.DROPOUT))

        # Register to Module
        super().__init__(*layers)

    # Ignore sequence length parameter needed for Transformer model
    def forward(self, x, _):
        return super().forward(x)
