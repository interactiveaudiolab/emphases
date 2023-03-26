# Task - search architectures

MODULE = 'emphases'

# Configuration name
CONFIG = 'avg-posthoc-wordwise'

# Search variables

# Method to use for inference
METHOD = 'framewise'

# Convert from frames to words on model evaluation (i.e. loss is evaluated wordwise)
MODEL_TO_WORDS = True

# Resampling mode for framewise models (from frames to words): 'max' or 'avg' or 'center'
FRAMES_TO_WORDS_RESAMPLE = 'avg'

# Whether to use pitch features
PITCH_FEATURE = True

# Whether to use pitch features
PERIODICITY_FEATURE = True

# Whether to use loudness features
LOUDNESS_FEATURE = True

# Whether to use prominence features
PROMINENCE_FEATURE = False

# Dataset
DATASETS = ['annotate']

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

# Whether to use BCELogitloss function
USE_BCE_LOGITS_LOSS = True

# Number of training steps
NUM_STEPS = 1000