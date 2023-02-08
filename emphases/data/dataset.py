import os

import numpy as np
import pypar
import torch
import torchaudio

import emphases


###############################################################################
# Dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):

    def __init__(self, name, partition, train_limit=None):
        # Get list of stems
        self.cache = emphases.CACHE_DIR / name
        all_stems = emphases.load.partition(name)[partition]

        # Store lengths for bucketing
        audio_files = list([
            self.cache / 'audio' / f'{stem}.wav' for stem in all_stems])
        all_lengths = [
            emphases.convert.samples_to_frames(torchaudio.info(audio_file).num_frames)
            for audio_file in audio_files]
        limit_frames = emphases.convert.seconds_to_frames(train_limit) if train_limit is not None else None
        if (limit_frames is not None) and limit_frames < sum(all_lengths):
            total_frames = 0
            self.stems = []
            self.lengths = []
            while total_frames < limit_frames:
                stem_frames = all_lengths.pop(0)
                if total_frames + stem_frames <= limit_frames:
                    self.lengths.append(stem_frames)
                    self.stems.append(all_stems.pop(0))
                    total_frames += stem_frames
                else:
                    total_frames = limit_frames
        else:
            self.stems = all_stems
            self.lengths = all_lengths
        

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
        mels = torch.load(self.cache / 'mels' / f'{stem}.pt')

        # Load per-word ground truth emphasis scores
        scores = torch.load(self.cache / 'scores' / f'{stem}.pt')[None]

        # Maybe interpolate scores for framewise model
        if emphases.METHOD in ['framewise', 'attention']:

            # Get center time of each word in frames
            word_centers = \
                word_bounds[0] + (word_bounds[1] - word_bounds[0]) / 2.

            # Get frame centers
            frame_centers = .5 + torch.arange(mels.shape[-1])

            # Interpolate
            scores = emphases.interpolate(
                frame_centers[None],
                word_centers[None],
                scores)

        return mels, scores, word_bounds, alignment, audio, stem

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
