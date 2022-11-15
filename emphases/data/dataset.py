"""dataset.py - data loading"""


import os

import pypar
import torch

import emphases
import numpy as np


###############################################################################
# Dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset

    Arguments
        name - string
            The name of the dataset
        partition - string
            The name of the data partition
    """

    def __init__(self, name, partition):
        # Get list of stems
        self.cache = emphases.CACHE_DIR / name
        self.stems = emphases.load.partition(name)[partition]

        # TODO - Get the length corresponding each stem so the sampler can
        #        use it. Note: you should not load all of the dataset to
        #        determine the lengths. Instead, you can use the file size.
        #        Here is an example that assumes 16-bit audio. It might work
        #        for your purposes.
        audio_files = [
            self.cache / 'wavs' / f'{stem}.wav' for stem in self.stems]
        self.lengths = [
            os.path.getsize(audio_file) // 2 for audio_file in audio_files]

        self.spectrogram_lengths = []
        for stem in self.stems:
            mel_spectrogram = torch.tensor(np.load(self.cache / 'mels' / f'{stem}.npy'))
            self.spectrogram_lengths.append(mel_spectrogram.shape[-1])

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        stem = self.stems[index]

        # Load audio
        audio = emphases.load.audio(self.cache / 'wavs' / f'{stem}.wav')

        # mel spectrogram
        mel_spectrogram = torch.tensor(np.load(self.cache / 'mels' / f'{stem}.npy'))

        # Load alignment
        alignment = pypar.Alignment(
            self.cache / 'alignment' / f'{stem}.json')

        # Load per-word ground truth prominence values
        prominence = emphases.load.load_prominence(self.cache / 'annotation' / f'{stem}.prom')

        # Get word start and end indices
        word_bounds = alignment.word_bounds(
            emphases.SAMPLE_RATE,
            emphases.HOPSIZE)

        assert (len(word_bounds) == prominence.shape[0]), 'array length mismatch b/w alignment and ground truth'

        return audio, mel_spectrogram, prominence, word_bounds

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)
