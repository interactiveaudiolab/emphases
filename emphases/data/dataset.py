"""dataset.py - data loading"""


import os

import pypar
import torch

import emphases
import numpy as np
from emphases.data.utils import constant, grid_sample, interpolate_numpy

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

        # updating word bounds with padding on silent time stamps
        wb_prom_pairs = []
        audio_len = audio.shape[-1]

        if word_bounds[0][0]!=0:
            wb_prom_pairs.append([(0, word_bounds[0][0]), 0])

        for idx in range(len(word_bounds)):
            wb_prom_pairs.append([word_bounds[idx], prominence[idx].item()])
            if idx+1<len(word_bounds):
                if word_bounds[idx][-1]!=word_bounds[idx+1][0]:
                    start = word_bounds[idx][-1]
                    end = word_bounds[idx+1][0]
                    wb_prom_pairs.append([(start, end), 0])

        # generating interpolated prom vector
        prom_extended = []
        for wb in wb_prom_pairs:
            start, end = wb[0][0], wb[0][1]
            prom_extended.extend([wb[-1]]*(end-start)*emphases.HOPSIZE)
            
        if word_bounds[-1][-1]!=audio_len//emphases.HOPSIZE:
            pad_len = audio_len - len(prom_extended)
            prom_extended.extend([0]*pad_len)

        prom_extended = torch.tensor(prom_extended)

        # frame based prominence values
        grid = constant(audio, emphases.HOPSIZE)
        # interpolated_prom_values = grid_sample(prominence, grid)
        interpolated_prom_values = interpolate_numpy(prom_extended, grid)

        return audio, mel_spectrogram, prominence, word_bounds, interpolated_prom_values

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)
