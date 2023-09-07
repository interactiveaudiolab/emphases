import torch

import emphases


###############################################################################
# Model definition
###############################################################################


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # Input projection
        self.input_layer = torch.nn.Conv1d(
            emphases.NUM_FEATURES,
            emphases.CHANNELS,
            kernel_size=emphases.ENCODER_KERNEL_SIZE,
            padding='same')

        # Frame encoder
        self.frame_encoder = emphases.model.Layers(
            kernel_size=emphases.ENCODER_KERNEL_SIZE)

        # If we are resampling within the model, initialize word decoder
        if emphases.DOWNSAMPLE_LOCATION in ['input', 'intermediate']:
            self.word_decoder = emphases.model.Layers(
            kernel_size=emphases.DECODER_KERNEL_SIZE)

        # Output projection
        self.output_layer = torch.nn.Conv1d(
            emphases.CHANNELS,
            1,
            kernel_size=emphases.DECODER_KERNEL_SIZE,
            padding='same')

    def forward(self, features, frame_lengths, word_bounds, word_lengths):

        if emphases.DOWNSAMPLE_LOCATION == 'input':

            # Segment acoustic features into word segments
            segments, bounds, lengths = emphases.segment(
                features,
                word_bounds,
                word_lengths)

            # Embed frames
            frame_embeddings = self.frame_encoder(
                self.input_layer(segments),
                lengths)

            # Downsample
            if emphases.DOWNSAMPLE_METHOD == 'average':
                word_embeddings = frame_embeddings.mean(dim=2, keepdim=True)
            elif emphases.DOWNSAMPLE_METHOD == 'max':
                word_embeddings = frame_embeddings.max(
                    dim=2,
                    keepdim=True
                ).values
            elif emphases.DOWNSAMPLE_METHOD == 'sum':
                word_embeddings = frame_embeddings.sum(dim=2, keepdim=True)
            elif emphases.DOWNSAMPLE_METHOD == 'center':
                word_embeddings = emphases.downsample(
                    frame_embeddings,
                    bounds,
                    torch.ones(
                        (len(lengths),),
                        dtype=torch.long,
                        device=lengths.device))
            else:
                raise ValueError(
                    f'Interpolation method {emphases.DOWNSAMPLE_METHOD} is not defined')

            # Stitch together word segment embeddings
            mask = mask_from_lengths(word_lengths)
            word_embeddings = word_embeddings.squeeze(2).transpose(0, 1).reshape(
                word_embeddings.shape[1],
                word_bounds.shape[0],
                word_bounds.shape[2]
            ).permute(1, 0, 2) * mask

            # Decode
            word_embeddings = self.word_decoder(
                word_embeddings,
                word_lengths)

        else:

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
                word_embeddings = self.word_decoder(
                    word_embeddings,
                    word_lengths)

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
