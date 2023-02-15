MODULE = 'emphases'

# Configuration name
CONFIG = 'framewise-linear-bceLoss-mel-prom-loud-pitch-period-small'

# Batch size (per gpu)
BATCH_SIZE = 2

# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 200  # steps

# Interpolation method for framewise training
INTERPOLATION = 'linear'

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 50  # steps

# Number of evaluation steps
EVALUATION_INTERVAL = 50 # steps

# Method to use for inference
METHOD = 'framewise'

# Number of training steps
NUM_STEPS = 1000

# Whether to BCELogitloss function
USE_BCE_LOGITS_LOSS = True

# Whether to use pitch features
PITCH_FEATURE = True

# Whether to use pitch features
PERIODICITY_FEATURE = True

# Whether to use loudness features
LOUDNESS_FEATURE = True

# Whether to use prominence features
PROMINENCE_FEATURE = True
