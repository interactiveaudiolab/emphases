###############################################################################
# Audio parameters
###############################################################################


# Maximum frequency in energy calculation
ENERGY_MAX = 5000.

# Minimum frequency in energy calculation
ENERGY_MIN = 200.

# Maximum allowable frequency
FMAX = 400.

# Minimum allowable frequency
FMIN = 50.

# Voiced/unvoiced threshold from 0 (all voiced) to 100 (all unvoiced)
VOICED_THRESHOLD = 50


###############################################################################
# Feature combination weights
###############################################################################


# Weight applied to the duration
DURATION_WEIGHT = .5

# Weight applied to the energy
ENERGY_WEIGHT = 1.

# Weight applied to the pitch
PITCH_WEIGHT = 1.


###############################################################################
# Line of Maximum Amplitude (LOMA) parameters
###############################################################################


# Boundary bounds
LOMA_BOUNDARY_START = -2  # octaves
LOMA_BOUNDARY_END = 1  # octaves

# Prominence bounds
LOMA_BOUNDARY_START = -3  # octaves
LOMA_BOUNDARY_END = 0  # octaves
