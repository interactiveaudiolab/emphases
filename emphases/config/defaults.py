from pathlib import Path


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


# The number of samples between frames
HOPSIZE = 160

# The audio samling rate
SAMPLE_RATE = 16000


###############################################################################
# Data parameters
###############################################################################


# Interpolation method for framewise training
INTERPOLATION = "linear"

MAX_NUM_OF_WORDS = 56 # set limit for number of words to be considered from in a given sample (truncating after this point for now)
MAX_WORD_DURATION = 96 # duration limit for a word based frame

# NUM_FFT is the length of the transformed axis of the output/ length of the windowed signal after padding with zeros
# https://stackoverflow.com/questions/62584184/understanding-the-shape-of-spectrograms-and-n-mels
NUM_FFT =  1024
NUM_MELS = 80

# Seed for all random number generators
RANDOM_SEED = 1234

# Size of each partition. Must add to 1.
SPLIT_SIZE_TEST = .1
SPLIT_SIZE_TRAIN = .8
SPLIT_SIZE_VALID = .1


###############################################################################
# Evaluation parameters
###############################################################################


# List of all datasets
DATASETS = ['buckeye', 'libritts']

# Maximum number of frames to perform inference on at once
MAX_FRAMES_PER_BATCH = 1024

# Number of steps between evaluation
EVALUATION_INTERVAL = 2500  # steps

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 1000  # steps

# Method to use for evaluation
METHOD = 'framewise'


###############################################################################
# Training parameters
###############################################################################


# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 25000  # steps

# Batch size (per gpu)
BATCH_SIZE = 64

# Number of training steps
NUM_STEPS = 300000

# Number of data loading worker threads
NUM_WORKERS = 2
