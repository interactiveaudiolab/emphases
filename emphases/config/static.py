"""Config parameters whose values depend on other config parameters"""
import emphases


###############################################################################
# Files and directories
###############################################################################


# Directory to save annotation artifacts
ANNOTATION_DIR = emphases.EVAL_DIR / 'annotation'

# Default configuration file for emphasis annotation
DEFAULT_ANNOTATION_CONFIG = emphases.ASSETS_DIR / 'configs' / 'absolute.yaml'

# Default checkpoint for generation
DEFAULT_CHECKPOINT = emphases.ASSETS_DIR / 'checkpoints'

# Default configuration file
DEFAULT_CONFIGURATION = emphases.ASSETS_DIR / 'configs' / 'emphases.py'

# Location to save dataset partitions
PARTITION_DIR = emphases.ASSETS_DIR / 'partitions'


###############################################################################
# Audio parameters
###############################################################################


# The hopsize in seconds
HOPSIZE_SECONDS = emphases.HOPSIZE / emphases.SAMPLE_RATE


###############################################################################
# Model parameters
###############################################################################


# Number of input features to the model
NUM_FEATURES = emphases.NUM_MELS + \
                int(emphases.PITCH_FEATURE) + \
                int(emphases.PERIODICITY_FEATURE) + \
                int(emphases.LOUDNESS_FEATURE) + \
                int(emphases.PROMINENCE_FEATURE)
