import json

import torch
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


def partition(dataset):
    """Load partitions for dataset"""
    with open(emphases.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)


def prominence(file):
    """Load prominence annotations"""
    # Read file
    with open(file) as file:
        data = file.read()

    # Skip header on first line
    lines = [x.split('\t') for x in data.split('\n')[1:]]

    # Get prominence values
    return torch.tensor([float(x[4]) for x in lines[:-1]])
