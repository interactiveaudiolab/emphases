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

    def __init__(
        self,
        # input_channels=emphases.NUM_MELS, #TODO have this filled in dynamically
        input_channels=1,
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
        
        return self.layers(features)


    def get_slices(input_features, wb):
        """
        Generate framewise slices as per word bounds
        return a padded tensor for given input_features

        """
        duration_slices = []
        for bound in wb:
            dur = (bound[1] - bound[0])*emphases.HOPSIZE
            duration_slices.append(dur)

        if sum(duration_slices)!=input_features.shape[-1]:
            extra_noise = True
            duration_slices.append(input_features.shape[-1] - sum(duration_slices))

        slices = torch.split(input_features, duration_slices)

        if extra_noise:
            # get rid of the extra noise duration
            duration_slices = duration_slices[::-1]
            slices = slices[::-1]
            
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
