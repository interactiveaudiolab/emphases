import torch

import emphases


###############################################################################
# Files and directories
###############################################################################


# Directory to save annotation artifacts
ANNOTATION_DIR = emphases.SOURCE_DIR / 'crowdsource'

# Default configuration file for emphasis annotation
DEFAULT_ANNOTATION_CONFIG = emphases.ASSETS_DIR / 'configs' / 'annotate.yaml'

# Location to save dataset partitions
PARTITION_DIR = emphases.ASSETS_DIR / 'partitions'


###############################################################################
# Audio parameters
###############################################################################


# The hopsize in seconds
HOPSIZE_SECONDS = emphases.HOPSIZE / emphases.SAMPLE_RATE

# The maximum representable frequency in log-hz
LOGFMAX = torch.log2(torch.tensor(emphases.FMAX))

# The minumum representable frequency in log-hz
LOGFMIN = torch.log2(torch.tensor(emphases.FMIN))


###############################################################################
# Model parameters
###############################################################################


# Number of input features to the model
NUM_FEATURES = (
    emphases.MEL_FEATURE * emphases.NUM_MELS +
    int(emphases.PITCH_FEATURE) +
    int(emphases.PERIODICITY_FEATURE) +
    int(emphases.LOUDNESS_FEATURE))
