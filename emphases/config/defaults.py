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

# The audio samling rate
SAMPLE_RATE = 16000

# The size of the audio analysis window
WINDOW_SIZE = 1024


###############################################################################
# Data parameters
###############################################################################


# List of all datasets
DATASETS = ['buckeye', 'libritts']

# Interpolation method for framewise training
INTERPOLATION = 'linear'

# Number of linear frequency channels
NUM_FFT =  1024

# Number of mel channels
NUM_MELS = 80

# Whether to use pitch features
PITCH_FEATURE = False

# Whether to use pitch features
PERIODICITY_FEATURE = False

# Whether to use loudness features
LOUDNESS_FEATURE = False

# Whether to use prominence features
PROMINENCE_FEATURE = False

# Seed for all random number generators
RANDOM_SEED = 1234

# Size of each partition. Must add to 1.
SPLIT_SIZE_TEST = .1
SPLIT_SIZE_TRAIN = .8
SPLIT_SIZE_VALID = .1


###############################################################################
# Evaluation parameters
###############################################################################


# Maximum number of frames to perform inference on at once
MAX_FRAMES_PER_BATCH = 2560

# Number of steps between evaluation
EVALUATION_INTERVAL = 100  # steps

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 1000  # steps

# Method to use for inference
METHOD = 'framewise'

# Convert from frames to words on model evaluation (i.e. loss is evaluated wordwise)
MODEL_TO_WORDS = True

# Either 'conv' or 'transformer', type of encoding stack to use
ENCODING_STACK = 'conv'

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
# Variance baseline parameters
###############################################################################

# Variance resampling mode (from phonemes to words): 'max' or 'avg'
VARIANCE_RESAMPLE = 'max'

# Used for interp_unvoiced_at for penn
PENN_VOICED_THRESHOLD = .065

###############################################################################
# Model component parameters
###############################################################################

HIDDEN_CHANNELS = 128

N_HEADS = 2

N_LAYERS = 2

ATTN_ENC_KERNEL_SIZE = 3

CONV_KERNEL_SIZE = 5

FFN_KERNEL_SIZE = 3

NUM_CONVS = 4

###############################################################################
# Training parameters
###############################################################################


# Number of buckets of data lengths used by the sampler
BUCKETS = 8

# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 25000  # steps

# Number of training steps
NUM_STEPS = 300000

# Number of data loading worker threads
NUM_WORKERS = 2

# Number of seconds of data to limit training to
TRAIN_DATA_LIMIT = None


# Resampling mode for framewise models (from frames to words): 'max' or 'avg' or 'center'
FRAMES_TO_WORDS_RESAMPLE = None

# Maximum number of frames in one batch
MAX_FRAMES = 50000

# Whether to use BCELogitloss function
USE_BCE_LOGITS_LOSS = False
