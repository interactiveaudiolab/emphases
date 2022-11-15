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
        return self.layers(features)

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
