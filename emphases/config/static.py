"""Config parameters whose values depend on other config parameters"""
import emphases


###############################################################################
# Directories
###############################################################################


# Location to save dataset partitions
PARTITION_DIR = emphases.ASSETS_DIR / 'partitions'

# Default checkpoint for generation
DEFAULT_CHECKPOINT = emphases.ASSETS_DIR / 'checkpoints'

# Default configuration file
DEFAULT_CONFIGURATION = emphases.ASSETS_DIR / 'configs' / 'emphases.py'


###############################################################################
# Audio parameters
###############################################################################


# The hopsize in seconds
HOPSIZE_SECONDS = emphases.HOPSIZE / emphases.SAMPLE_RATE
