import os

import numpy as np
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

        # Store lengths for bucketing
        audio_files = list([
            self.cache / 'audio' / f'{stem}.wav' for stem in self.stems])
        self.lengths = [
            os.path.getsize(audio_file) // (2 * emphases.HOPSIZE)
            for audio_file in audio_files]

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        stem = self.stems[index]

        # Load alignment
        alignment = pypar.Alignment(
            self.cache / 'alignment' / f'{stem}.TextGrid')

        # Compute word bounds
        bounds = alignment.word_bounds(
            emphases.SAMPLE_RATE,
            emphases.HOPSIZE,
            silences=True)
        word_bounds = torch.cat(
            [torch.tensor(bound)[None] for bound in bounds]).T

        # Load audio
        audio = emphases.load.audio(self.cache / 'audio' / f'{stem}.wav')

        # Load mels
        features = torch.load(self.cache / 'mels' / f'{stem}.pt')
        
        if emphases.PITCH_FEATURE:
            # Load pitch
            pitch = torch.load(self.cache / 'pitch' / f'{stem}-pitch.pt')
            features = torch.cat((features, torch.log2(pitch)[None, :]), dim=1)

        if emphases.PERIODICITY_FEATURE:
            # Load periodicity
            periodicity = torch.load(self.cache / 'pitch' / f'{stem}-periodicity.pt')
            features = torch.cat((features, periodicity[None, :]), dim=1)

        # Load per-word ground truth emphasis scores
        scores = torch.load(self.cache / 'scores' / f'{stem}.pt')[None]

        # Maybe interpolate scores for framewise model
        if emphases.METHOD == 'framewise':

            # Get center time of each word in frames
            word_centers = \
                word_bounds[0] + (word_bounds[1] - word_bounds[0]) / 2.

            # Get frame centers
            frame_centers = .5 + torch.arange(features.shape[-1])

            # Interpolate
            scores = emphases.interpolate(
                frame_centers[None],
                word_centers[None],
                scores)

        return features, scores, word_bounds, alignment, audio, stem

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)

    def buckets(self):
        """Partition indices into buckets based on length for sampling"""
        # Get the size of a bucket
        size = len(self) // emphases.BUCKETS

        # Get indices in order of length
        indices = np.argsort(self.lengths)

        # Split into buckets based on length
        return [indices[i:i + size] for i in range(0, len(self), size)]
