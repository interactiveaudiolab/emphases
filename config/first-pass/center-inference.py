MODULE = 'emphases'

# Configuration name
CONFIG = 'center-inference'

# Location to perform resampling from frame resolution to word resolution.
# One of ['inference', 'intermediate', 'loss'].
DOWNSAMPLE_LOCATION = 'inference'

# Method to use for resampling from frame resolution to word resolution.
# One of ['average', 'center', 'max'].
DOWNSAMPLE_METHOD = 'center'
