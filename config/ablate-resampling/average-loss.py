MODULE = 'emphases'

# Configuration name
CONFIG = 'average-loss'

# Location to perform resampling from frame resolution to word resolution.
# One of ['inference', 'input', 'intermediate', 'loss'].
DOWNSAMPLE_LOCATION = 'loss'

# Method to use for resampling from frame resolution to word resolution.
# One of ['average', 'center', 'max'].
DOWNSAMPLE_METHOD = 'average'
