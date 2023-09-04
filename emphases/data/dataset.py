import numpy as np
import pypar
import torch
import torchaudio

import emphases


###############################################################################
# Dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):

    def __init__(self, name, partition):
        self.cache = emphases.CACHE_DIR / name

        # Get list of stems
        self.stems = emphases.load.partition(name)[partition]

        # Store lengths for bucketing
        audio_files = [
            self.cache / 'audio' / f'{stem}.wav' for stem in self.stems]
        self.lengths = [
            emphases.convert.samples_to_frames(
                torchaudio.info(audio_file).num_frames)
            for audio_file in audio_files]

        # Total number of frames
        self.frames = sum(self.lengths)

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

        features = []

        # Load mels
        if emphases.MEL_FEATURE:
            features.append(torch.load(self.cache / 'mels' / f'{stem}.pt'))

        # Load pitch
        if emphases.PITCH_FEATURE:
            pitch = torch.load(self.cache / 'pitch' / f'{stem}-pitch.pt')
            if emphases.NORMALIZE:
                features.append(
                    (torch.log2(pitch) - emphases.LOGFMIN) /
                    (emphases.LOGFMAX - emphases.LOGFMIN))
            else:
                features.append(torch.log2(pitch))

        # Load periodicity
        if emphases.PERIODICITY_FEATURE:
            periodicity = torch.load(
                self.cache / 'pitch' / f'{stem}-periodicity.pt')
            features.append(periodicity)

        # Load loudness
        if emphases.LOUDNESS_FEATURE:
            loudness = torch.load(self.cache / 'loudness' / f'{stem}.pt')
            features.append(loudness)

        # Concatenate
        features = features[0] if len(features) == 1 else torch.cat(features)

        # Load per-word ground truth emphasis scores
        scores = torch.load(self.cache / 'scores' / f'{stem}.pt')[None]

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
        buckets = [indices[i:i + size] for i in range(0, len(self), size)]

        # Add max length of each bucket
        return [(self.lengths[bucket[-1]], bucket) for bucket in buckets]
