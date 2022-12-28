import functools

import torch

import emphases


###############################################################################
# Model selection
###############################################################################


def Model():
    """Create a model"""
    if emphases.METHOD == 'framewise':
        return Framewise()
    elif emphases.METHOD == 'wordwise':
        return Wordwise()
    else:
        raise ValueError(f'Model {emphases.METHOD} is not defined')


###############################################################################
# Model definitions
###############################################################################


class Framewise(torch.nn.Sequential):

    def __init__(
        self,
        input_channels=emphases.NUM_MELS,
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


class Wordwise(torch.nn.Module):

    def __init__(
            self,
            input_channels=emphases.NUM_MELS,
            output_channels=1,
            hidden_channels=128,
            kernel_size=5):
        super().__init__()

        # Setup frame encoder
        conv1d = functools.partial(
            torch.nn.Conv1d,
            kernel_size=kernel_size,
            padding='same')
        self.frame_encoder = torch.nn.Sequential(
            conv1d(input_channels, hidden_channels),
            torch.nn.ReLU(),
            conv1d(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv1d(hidden_channels, hidden_channels))

        # Setup word decoder
        conv2d = functools.partial(
            torch.nn.Conv2d,
            kernel_size=kernel_size,
            padding='same')
        self.word_decoder = torch.nn.Sequential(
            conv2d(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv2d(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(emphases.MAX_WORD_DURATION, 1),
            torch.nn.ReLU(),
            conv2d(hidden_channels, output_channels))

    def forward(self, features, word_bounds, word_lengths):
        # Embed frames
        frame_embedding = self.frame_encoder(features)

        # Slice embeddings into words
        # TODO - separate MAX_NUM_OF_WORDS, MAX_WORD_DURATION in forward pass
        feats, feat_lens = [], []
        for idx, (embedding, bounds, length) in enumerate(zip(frame_embedding, word_bounds, word_lengths)):
            feat, feat_length = self.slice(embedding, bounds, length)
            feats.append(feat)
            feat_lens.append(feat_length)

        # Place in one tensor
        # BATCH * HIDDEN_CHANNEL * MAX_NUM_OF_WORDS * MAX_WORD_DURATION
        padded_features_2 = torch.zeros(
            (
                emphases.BATCH_SIZE,
                frame_embedding.shape[1],
                emphases.MAX_NUM_OF_WORDS,
                emphases.MAX_WORD_DURATION
            ),
            device=features.device)
        for idx, (f_len, f_item) in enumerate(zip(feat_lens, feats)):
            padded_features_2[idx, :, :f_item.shape[1],
                              :f_item.shape[-1]] = f_item[:]

        # Infer emphasis scores from word embeddings
        # BATCH * 1 * MAX_NUM_OF_WORDS * 1
        return self.word_decoder(padded_features_2)

    def slice(self, features, word_bounds, length):
        """
        Generate framewise slices as per word bounds
        return a padded tensor for given input_features

        """
        # TODO
        # Get durations in frames
        durations = word_bounds[1, :length] - word_bounds[0, :length]

        extra_noise = False
        if sum(duration_slices) != features.shape[-1]:
            extra_noise = True
            duration_slices.append(features.shape[-1] - sum(duration_slices))

        padded_features = torch.zeros(
            (len(features), emphases.MAX_NUM_OF_WORDS, emphases.MAX_WORD_DURATION))

        for channel_idx, input_features in enumerate(features):

            slices = torch.split(input_features, duration_slices)

            if extra_noise:
                slices = slices[:-1]

            for idx, sl in enumerate(slices[:emphases.MAX_NUM_OF_WORDS]):

                if len(sl) <= emphases.MAX_WORD_DURATION:
                    padded_features[channel_idx, idx, :len(
                        sl)] = sl[:emphases.MAX_WORD_DURATION]

                else:
                    padded_features[channel_idx, idx,
                                    :emphases.MAX_WORD_DURATION] = sl[:emphases.MAX_WORD_DURATION]

        return padded_features, padded_features.shape[-1]
