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

    def forward(self, features, word_bounds):
        # generate input slices for every item in batch,
        # then form a padded tensor from all the slice tensors, and further pass down the network

        # Embed frames
        frame_embedding = self.frame_encoder(features)

        # Slice embeddings into words
        # TODO - separate MAX_NUM_OF_WORDS, MAX_WORD_DURATION in forward pass
        feats, feat_lens = [], []
        for idx, (input_features, wb) in enumerate(zip(frame_embedding, word_bounds)):
            feat, feat_length = self.get_slices_spectro_channels(input_features, wb)
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
            padded_features_2[idx, :, :f_item.shape[1], :f_item.shape[-1]] = f_item[:]

        # Infer emphasis scores from word embeddings
        # BATCH * 1 * MAX_NUM_OF_WORDS * 1
        return self.word_decoder(padded_features_2)

    def get_slices_spectro_channels(self, input_features_channels, wb):
        """
        Generate framewise slices as per word bounds
        return a padded tensor for given input_features

        """
        duration_slices = []

        for bound in wb:
            dur = (bound[1] - bound[0])
            duration_slices.append(dur)

        extra_noise = False

        if sum(duration_slices)!=input_features_channels.shape[-1]:
            extra_noise = True
            duration_slices.append(input_features_channels.shape[-1] - sum(duration_slices))

        # if extra_noise:
            # padded_features = torch.zeros(
            #         (len(input_features_channels), len(duration_slices[:-1]), max(duration_slices[:-1])))
        # else:
            # padded_features = torch.zeros(
            #         (len(input_features_channels), len(duration_slices), max(duration_slices)))

        padded_features = torch.zeros(
                (len(input_features_channels), emphases.MAX_NUM_OF_WORDS, emphases.MAX_WORD_DURATION))

        for channel_idx, input_features in enumerate(input_features_channels):

            slices = torch.split(input_features, duration_slices)

            if extra_noise:
                slices = slices[:-1]

            for idx, sl in enumerate(slices[:emphases.MAX_NUM_OF_WORDS]):

                if len(sl)<=emphases.MAX_WORD_DURATION:
                    padded_features[channel_idx, idx, :len(sl)] = sl[:emphases.MAX_WORD_DURATION]

                else:
                    padded_features[channel_idx, idx, :emphases.MAX_WORD_DURATION] = sl[:emphases.MAX_WORD_DURATION]

        return padded_features, padded_features.shape[-1]


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
