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
DATASETS = ['annotate', 'buckeye', 'libritts']

# Datasets to use for evaluation
EVALUATION_DATASETS = ['annotate', 'buckeye']

# Whether to use mel features
MEL_FEATURE = True

# Whether to use loudness features
LOUDNESS_FEATURE = True

# Maximum number of allowed annotations
MAX_ANNOTATIONS = None

# Minimum number of allowed annotations
MIN_ANNOTATIONS = None

# Whether to use the specified one-eighth dataset for scaling law experiments
ONE_EIGHTH_ANNOTATIONS = False

# Whether to use pitch features
PITCH_FEATURE = True

# Whether to use periodicity features
PERIODICITY_FEATURE = True

# Seed for all random number generators
RANDOM_SEED = 1234

# Size of each partition. Must add to 1.
SPLIT_SIZE_TEST = .1
SPLIT_SIZE_TRAIN = .8
SPLIT_SIZE_VALID = .1


###############################################################################
# Evaluation parameters
###############################################################################


# Number of steps between logging to Tensorboard
# TEMPORARY
# LOG_INTERVAL = 1000  # steps
LOG_INTERVAL = 100  # steps

# Number of steps to perform for tensorboard logging
LOG_STEPS = 32

# Number of examples to plot to Tensorboard during training
PLOT_EXAMPLES = 2


###############################################################################
# Prominence baseline parameters
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


# Model architecture. One of ['convolution', 'transformer'].
# TEMPORARY
# ARCHITECTURE = 'transformer'
ARCHITECTURE = 'convolution'

# Model width
CHANNELS = 512

# Location to perform resampling from frame resolution to word resolution.
# One of ['inference', 'intermediate', 'loss'].
DOWNSAMPLE_LOCATION = 'intermediate'

# Method to use for resampling from frame resolution to word resolution.
# One of ['average', 'center', 'max'].
DOWNSAMPLE_METHOD = 'max'

# Convolution kernel size
KERNEL_SIZE = 3

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
BUCKETS = 8

# Number of steps between saving checkpoints
# TEMPORARY
# CHECKPOINT_INTERVAL = 25000  # steps
CHECKPOINT_INTERVAL = 500  # steps

# Loss function. One of ['bce', 'mse']
# TEMPORARY
# LOSS = 'bce'
LOSS = 'mse'

# Maximum number of frames in one batch
MAX_FRAMES = 50000

# Number of training steps
# TEMPORARY
# NUM_STEPS = 300000
NUM_STEPS = 4000

# Number of data loading worker threads
NUM_WORKERS = 4

# Number of seconds of data to limit training to
TRAIN_DATA_LIMIT = None
