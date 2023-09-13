import torchaudio

import emphases


###############################################################################
# Loading utilities
###############################################################################


def audio(file):
    """Load audio and maybe resample"""
    # Load
    audio, sample_rate = torchaudio.load(file)

    # Maybe resample
    return emphases.resample(audio, sample_rate)
