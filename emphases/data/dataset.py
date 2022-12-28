import os

import pypar
import torch

import emphases


###############################################################################
# Dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):

    def __init__(self, name, partition):
        # Get list of stems
        self.cache = emphases.CACHE_DIR / name
        self.stems = emphases.load.partition(name)[partition]

        # Store spectrogram lengths for bucketing
        audio_files = list([self.cache / f'{stem}.wav' for stem in self.stems])
        self.spectrogram_lengths = [
            os.path.getsize(audio_file) // (2 * emphases.HOPSIZE)
            for audio_file in audio_files]

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        stem = self.stems[index]

        # Load alignment
        alignment = pypar.Alignment(
            self.cache / 'alignment' / f'{stem}.TextGrid')

        # Compute word bounds and lengths
        bounds = alignment.word_bounds(emphases.SAMPLE_RATE, emphases.HOPSIZE)
        word_bounds = torch.cat(
            torch.tensor(bound for bound in bounds)[None],
            dim=1)

        # Load audio
        audio = emphases.load.audio(self.cache / 'audio' / f'{stem}.wav')

        # Load mels
        mels = torch.tensor(self.cache / 'mels' / f'{stem}.pt')

        # Load per-word ground truth emphasis scores
        scores = torch.load(self.cache / 'scores' / f'{stem}.pt')

        # Maybe interpolate scores for framewise model
        if emphases.METHOD == 'framewise':

            # Get center time of each word in frames
            word_centers = (word_bounds[1] - word_bounds[0]) / 2.

            # Get frame centers
            frame_centers = .5 + torch.arange(mels.shape[-1])

            # Interpolate
            scores = emphases.interpolate(scores, word_centers, frame_centers)

        return mels, scores, word_bounds, alignment, audio, stem

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)
