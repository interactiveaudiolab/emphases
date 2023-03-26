# Task - search features
# Model - framewise
# Word resampling - True, max
# Features: mels, pitch
# Loss - BCE

MODULE = 'emphases'

# Configuration name
CONFIG = 'feat-mels-period-loud-intermediate-wordwise-resampling-max'

# Search variables

# Whether to use pitch features
PITCH_FEATURE = False

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

# Method to use for inference
METHOD = 'wordwise'

# Whether to use BCELogitloss function
USE_BCE_LOGITS_LOSS = True

# Resampling mode for framewise models (from frames to words): 'max' or 'avg' or 'center'
FRAMES_TO_WORDS_RESAMPLE = 'max'

# Number of training steps
NUM_STEPS = 1000