MODULE = 'emphases'

# Configuration name
CONFIG = 'framewise-linear-small'

# Batch size (per gpu)
BATCH_SIZE = 2

# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 200  # steps

# Interpolation method for framewise training
INTERPOLATION = 'linear'

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 50  # steps

# Method to use for inference
METHOD = 'framewise'

# Number of training steps
NUM_STEPS = 1000
