MODULE = 'emphases'

# Configuration name
CONFIG = 'wordwise-small'

# Batch size (per gpu)
BATCH_SIZE = 2

# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 200  # steps

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 50  # steps

# Method to use for evaluation
METHOD = 'wordwise'

# Number of training steps
NUM_STEPS = 1000
