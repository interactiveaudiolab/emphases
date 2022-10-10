import json

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
    if sample_rate != emphases.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            sample_rate,
            emphases.SAMPLE_RATE)
        audio = resampler(audio)

    return audio


def partition(dataset):
    """Load partitions for dataset"""
    with open(emphases.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)
