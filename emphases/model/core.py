import functools

import torch

import emphases


###############################################################################
# Model definition
###############################################################################


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Bind common parameters
        conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=emphases.KERNEL_SIZE,
            padding='same')

        # Input projection
        self.input_layer = conv_fn(emphases.NUM_FEATURES, emphases.CHANNELS)

        # Frame encoder
        self.frame_encoder = emphases.model.Layers()

        # If we are resampling within the model, initialize word decoder
        if emphases.DOWNSAMPLE_LOCATION == 'intermediate':
            self.word_decoder = emphases.model.Layers()

        # Output projection
        self.output_layer = conv_fn(emphases.CHANNELS, 1)

    def forward(self, features, frame_lengths, word_bounds, word_lengths):
        # Embed frames
        frame_embeddings = self.frame_encoder(
            self.input_layer(features),
            frame_lengths)

        if emphases.DOWNSAMPLE_LOCATION == 'intermediate':

            # Downsample activations to word resolution
            word_embeddings = emphases.downsample(
                frame_embeddings,
                word_bounds,
                word_lengths)

            # Infer emphasis scores from word embeddings
            word_embeddings = self.word_decoder(word_embeddings, word_lengths)

        elif emphases.DOWNSAMPLE_LOCATION == 'loss':

            # Downsample activations to word resolution
            word_embeddings = emphases.downsample(
                frame_embeddings,
                word_bounds,
                word_lengths)

        elif emphases.DOWNSAMPLE_LOCATION == 'inference':

            if self.training:

                # Return frame resolution prominence for framewise loss
                return self.output_layer(frame_embeddings)

            else:

                # Downsample activations to word resolution
                word_embeddings = emphases.downsample(
                    frame_embeddings,
                    word_bounds,
                    word_lengths)

        else:
            raise ValueError(
                f'Downsample location {emphases.DOWNSAMPLE_LOCATION} ' +
                'not recognized')

        # Project to scalar
        return self.output_layer(word_embeddings)


###############################################################################
# Utilities
###############################################################################


def mask_from_lengths(lengths):
    """Create boolean mask from sequence lengths"""
    x = torch.arange(lengths.max(), dtype=lengths.dtype, device=lengths.device)
    return (x.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1)
