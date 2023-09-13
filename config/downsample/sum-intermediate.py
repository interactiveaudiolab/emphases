MODULE = 'emphases'

# Configuration name
CONFIG = 'sum-intermediate'

# Location to perform resampling from frame resolution to word resolution.
# One of ['inference', 'input', 'intermediate', 'loss'].
DOWNSAMPLE_LOCATION = 'intermediate'

# Method to use for resampling from frame resolution to word resolution.
# One of ['average', 'center', 'max', 'sum'].
DOWNSAMPLE_METHOD = 'sum'
