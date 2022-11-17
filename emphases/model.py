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
        super().__init__()
        self.layers = torch.nn.Sequential(
            conv_fn(input_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, output_channels))

    def forward(self, features):
        # TODO: generate input slices for every item in batch, 
        # then form a padded tensor from all the slice tensors, and further pass down the network
        
        padded_mel_spectrogram, word_bounds, padded_prominence = features
        intermid_output = self.layers(padded_mel_spectrogram)
        
        feat_lens = []
        feats = []
        for idx, (input_features, wb) in enumerate(zip(intermid_output, word_bounds)):
            feat, feat_length = self.get_slices(input_features.reshape(-1), wb)
            feats.append(feat)
            feat_lens.append(feat_length)

        max_flen = max(feat_lens)
        padded_features_2 = torch.zeros((emphases.BATCH_SIZE, padded_prominence.shape[-1], max_flen))
        for idx, (f_len, f_item) in enumerate(zip(feat_lens, feats)):
            padded_features_2[idx, :f_item.shape[1], :f_item.shape[-1]] = f_item[:]

        lin_f1 = padded_features_2.shape[-1]
        self.layers2 = torch.nn.Sequential(
            torch.nn.ReLU(), 
            torch.nn.Linear(lin_f1, 1)
            )

        return self.layers2(padded_features_2)

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
