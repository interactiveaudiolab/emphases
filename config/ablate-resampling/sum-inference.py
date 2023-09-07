MODULE = 'emphases'

# Configuration name
CONFIG = 'sum-inference'

# Location to perform resampling from frame resolution to word resolution.
# One of ['inference', 'input', 'intermediate', 'loss'].
DOWNSAMPLE_LOCATION = 'inference'

# Method to use for resampling from frame resolution to word resolution.
# One of ['average', 'center', 'max', 'sum'].
DOWNSAMPLE_METHOD = 'sum'
