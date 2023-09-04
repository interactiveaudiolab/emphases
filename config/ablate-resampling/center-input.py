MODULE = 'emphases'

# Configuration name
CONFIG = 'center-input'

# Location to perform resampling from frame resolution to word resolution.
# One of ['inference', 'input', 'intermediate', 'loss'].
DOWNSAMPLE_LOCATION = 'input'

# Method to use for resampling from frame resolution to word resolution.
# One of ['average', 'center', 'max'].
DOWNSAMPLE_METHOD = 'center'
