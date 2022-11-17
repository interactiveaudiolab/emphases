"""model.py - model definition"""


import torch
import emphases
import functools

###############################################################################
# Model
###############################################################################



class BaselineModel(torch.nn.Sequential):
    # Question: let's say assuming framewise training 
    # why not output channel == no. of words (no. of prom values) for a given frame
    # TODO - Use MEL features - change the network accordingly


    def __init__(
        self,
        input_channels=emphases.NUM_MELS, #TODO have this filled in dynamically
        output_channels=1,
        hidden_channels=128,
        kernel_size=5):

        conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=kernel_size,
            padding='same')

        conv2d_fn = functools.partial(
            torch.nn.Conv2d,
            kernel_size=kernel_size,
            padding='same')

        super().__init__()
        self.layers = torch.nn.Sequential(
            conv_fn(input_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, hidden_channels))

        self.layers2 = torch.nn.Sequential(
            conv2d_fn(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv2d_fn(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv2d_fn(hidden_channels, output_channels))

        # TODO - add layer to convert final feature set into BATCH_SIZE * 1 * MAX_NUM_WORD
        # maybe linear layer - but it needs input dim, which might change

    def forward(self, features):
        # TODO: generate input slices for every item in batch, 
        # then form a padded tensor from all the slice tensors, and further pass down the network
        
        padded_mel_spectrogram, word_bounds, padded_prominence = features
        intermid_output = self.layers(padded_mel_spectrogram)
        
        feat_lens = []
        feats = []
        for idx, (input_features, wb) in enumerate(zip(intermid_output, word_bounds)):
            feat, feat_length = self.get_slices_spectro_channels(input_features, wb)
            feats.append(feat)
            feat_lens.append(feat_length)

        max_flen = max(feat_lens) # max_slice_duration_len sampled from the sliced samples of a batch
        # BATCH * HIDDEN_CHANNEL * MAX_NUM_OF_WORDS * max_slice_duration_len
        padded_features_2 = torch.zeros((emphases.BATCH_SIZE, intermid_output.shape[1], padded_prominence.shape[-1], max_flen))

        for idx, (f_len, f_item) in enumerate(zip(feat_lens, feats)):
            padded_features_2[idx, :, :f_item.shape[1], :f_item.shape[-1]] = f_item[:]

        return self.layers2(padded_features_2)

    def get_slices_spectro_channels(self, input_features_channels, wb):
        """
        Generate framewise slices as per word bounds
        return a padded tensor for given input_features

        """
        duration_slices = []

        for bound in wb:
            # dur = (bound[1] - bound[0])*emphases.HOPSIZE # for audio inputs
            dur = (bound[1] - bound[0]) # for mel spectro inputs
            duration_slices.append(dur)

        extra_noise = False

        if sum(duration_slices)!=input_features_channels.shape[-1]:
            extra_noise = True
            duration_slices.append(input_features_channels.shape[-1] - sum(duration_slices))

        if extra_noise:
            padded_features = torch.zeros(
                    (len(input_features_channels), len(duration_slices[:-1]), max(duration_slices[:-1])))
        else:
            padded_features = torch.zeros(
                    (len(input_features_channels), len(duration_slices), max(duration_slices)))

        for channel_idx, input_features in enumerate(input_features_channels):

            slices = torch.split(input_features, duration_slices)

            if extra_noise:
                slices = slices[:-1]

            for idx, sl in enumerate(slices):
                padded_features[channel_idx, idx, :len(sl)] = sl

        return padded_features, padded_features.shape[-1]

    def get_slices(self, input_features, wb):
        """
        Generate framewise slices as per word bounds
        return a padded tensor for given input_features

        """
        duration_slices = []
        for bound in wb:
            # dur = (bound[1] - bound[0])*emphases.HOPSIZE # for audio inputs
            dur = (bound[1] - bound[0]) # for mel spectro inputs
            duration_slices.append(dur)
        
        extra_noise = False
        
        if sum(duration_slices)!=input_features.shape[-1]:
            extra_noise = True
            duration_slices.append(input_features.shape[-1] - sum(duration_slices))

        slices = torch.split(input_features, duration_slices)

        if extra_noise:
            # get rid of the extra noise duration
            duration_slices = duration_slices[:-1]
            slices = slices[:-1]

        # forming a padded tensor to stack all the duration slices, 
        # using maximum slice-duration size for padding dimension
        padded_features = torch.zeros(
                (1, len(duration_slices), max(duration_slices)))

        for idx, sl in enumerate(slices):
            padded_features[:, idx, :len(sl)] = sl

        feat_len = padded_features.shape[-1]

        return padded_features, feat_len

# class Model(torch.nn.Module):
#     """Model definition"""

#     # TODO - add hyperparameters as input args
#     def __init__(self):
#         super().__init__()

#         # TODO - define model
#         raise NotImplementedError

#     ###########################################################################
#     # Forward pass
#     ###########################################################################

#     def forward(self):
#         """Perform model inference"""
#         # TODO - define model arguments and implement forward pass
#         raise NotImplementedError
