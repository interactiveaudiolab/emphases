import torch

MODULE = 'emphases'

# Configuration name
CONFIG = 'mse'

# Loss function
LOSS = torch.nn.functional.mse_loss
