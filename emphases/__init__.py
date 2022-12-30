# Development
# - Upload buckeye tarball****
# - Buckeye score files***
# - Baseline prominence method**


###############################################################################
# Configuration
###############################################################################


# Default configuration parameters to be modified
from .config import defaults

# Modify configuration
import yapecs
yapecs.configure('emphases', defaults)

# Import configuration parameters
from .config.defaults import *
from .config.static import *


###############################################################################
# Module imports
###############################################################################


from .core import *
from . import baselines
from . import checkpoint
from . import data
from . import evaluate
from . import load
from . import partition
from . import preprocess
from . import train
from . import write
from .interpolate import interpolate
from .model import Model
