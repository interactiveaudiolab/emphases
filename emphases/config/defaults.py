import os
from pathlib import Path

import torch
import GPUtil


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG = 'emphases'


###############################################################################
# Directories
###############################################################################


# Location to save assets to be bundled with pip release
ASSETS_DIR = Path(__file__).parent.parent / 'assets'

# Location of preprocessed features
CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache'

# Location of datasets on disk
DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'datasets'

# Location to save evaluation artifacts
EVAL_DIR = Path(__file__).parent.parent.parent / 'eval'

# Location to save training and adaptation artifacts
RUNS_DIR = Path(__file__).parent.parent.parent / 'runs'

# Location of compressed datasets on disk
SOURCE_DIR = Path(__file__).parent.parent.parent / 'data' / 'sources'


###############################################################################
# Audio parameters
###############################################################################


# The maximum representable frequency
FMAX = 550.

# The minumum representable frequency
FMIN = 40.

# The number of samples between frames
HOPSIZE = 160

# Minimum decibel level
MIN_DB = -100.

# Number of linear frequency channels
NUM_FFT =  1024

# Number of mel channels
NUM_MELS = 80

# Voiced/unvoiced threshold for pitch estimation
VOICED_THRESHOLD = .1625

# Reference decibel level
REF_DB = 20.

# The audio samling rate
SAMPLE_RATE = 16000

# The size of the audio analysis window
WINDOW_SIZE = 1024


###############################################################################
# Data parameters
###############################################################################


# List of all datasets
DATASETS = ['libritts']

# Datasets to use for evaluation
EVALUATION_DATASETS = ['libritts']

# Whether to use mel features
MEL_FEATURE = True

# Whether to use loudness features
LOUDNESS_FEATURE = False

# Maximum number of allowed annotations
MAX_ANNOTATIONS = None

# Maximum number of training utterances
MAX_TRAINING_UTTERANCES = None

# Minimum number of allowed annotations
MIN_ANNOTATIONS = None

# Normalize input representations
NORMALIZE = False

# Whether to use the specified one-eighth dataset for scaling law experiments
ONE_EIGHTH_UTTERANCES = False

# Whether to use pitch features
PITCH_FEATURE = False

# Whether to use periodicity features
PERIODICITY_FEATURE = False

# Seed for all random number generators
RANDOM_SEED = 0

# Size of each partition. Must add to 1.
SPLIT_SIZE_TEST = .1
SPLIT_SIZE_TRAIN = .8
SPLIT_SIZE_VALID = .1

# Dataset to use for training
TRAINING_DATASET = 'libritts'

# Dataset to use for validation
VALIDATION_DATASET = 'libritts'


###############################################################################
# Evaluation parameters
###############################################################################


# Number of steps between logging to Tensorboard
LOG_INTERVAL = 100  # steps

# Number of steps to perform for tensorboard logging
LOG_STEPS = 32

# Number of examples to plot to Tensorboard during training
PLOT_EXAMPLES = 2


###############################################################################
# Wavelet baseline parameters
###############################################################################


# Line of maximum amplitude bounds
LOMA_BOUNDARY_START = -2  # octaves
LOMA_BOUNDARY_END = 1  # octaves
LOMA_PROMINENCE_START = -3  # octaves
LOMA_PROMINENCE_END = 0  # octaves

# Weight applied to the duration
PROMINENCE_DURATION_WEIGHT = .5

# Maximum frequency in energy calculation
PROMINENCE_ENERGY_MAX = 5000.

# Minimum frequency in energy calculation
PROMINENCE_ENERGY_MIN = 200.

# Weight applied to the energy
PROMINENCE_ENERGY_WEIGHT = 1.

# Weight applied to the pitch
PROMINENCE_PITCH_WEIGHT = 1.

# Voiced/unvoiced threshold from 0 (all voiced) to 100 (all unvoiced)
VOICED_THRESHOLD = 50


###############################################################################
# Model parameters
###############################################################################


# Activation function to use in convolution model
ACTIVATION_FUNCTION = torch.nn.ReLU

# Model architecture. One of ['convolution', 'transformer'].
ARCHITECTURE = 'convolution'

# Model width
CHANNELS = 80

# Decoder convolution kernel size
DECODER_KERNEL_SIZE = 3

# Dropout probability (or None to not use dropout)
DROPOUT = None

# Location to perform resampling from frame resolution to word resolution.
# One of ['inference', 'input', 'intermediate', 'loss'].
DOWNSAMPLE_LOCATION = 'intermediate'

# Method to use for resampling from frame resolution to word resolution.
# One of ['average', 'center', 'max', 'sum'].
DOWNSAMPLE_METHOD = 'sum'

# Encoder convolution kernel size
ENCODER_KERNEL_SIZE = 3

# Number of network layers
LAYERS = 6

# Method to use for inference. One of
# ['neural', 'pitch-variance', 'duration-variance', 'prominence].
METHOD = 'neural'

# Method to use for resampling from word resolution to frame resolution.
# One of ['linear', 'nearest'].
UPSAMPLE_METHOD = 'linear'


###############################################################################
# Training parameters
###############################################################################


# Number of buckets of data lengths used by the sampler
BUCKETS = 2

# Loss function. One of ['bce', 'mse']
LOSS = 'bce'

# Maximum number of frames in one batch
MAX_TRAINING_FRAMES = 75000

# Number of training steps
NUM_STEPS = 6000

# Number of data loading worker threads
try:
    NUM_WORKERS = int(os.cpu_count() / max(1, len(GPUtil.getGPUs())))
except ValueError:
    NUM_WORKERS = os.cpu_count()
