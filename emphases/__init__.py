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
from .model import Model
from . train import loss, train
from . import annotate
from . import baselines
from . import convert
from . import data
from . import evaluate
from . import load
from . import partition
from . import plot
