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
        features = torch.load(self.cache / 'mels' / f'{stem}.pt')

        # Load pitch
        if emphases.PITCH_FEATURE:
            pitch = torch.load(self.cache / 'pitch' / f'{stem}-pitch.pt')
            features = torch.cat((features, torch.log2(pitch)[None, :]), dim=1)

        # Load periodicity
        if emphases.PERIODICITY_FEATURE:
            periodicity = torch.load(self.cache / 'pitch' / f'{stem}-periodicity.pt')
            features = torch.cat((features, periodicity[None, :]), dim=1)

        # Load loudness
        if emphases.LOUDNESS_FEATURE:
            loudness = torch.load(self.cache / 'loudness' / f'{stem}.pt')
            features = torch.cat((features, loudness[None, :]), dim=1)

        # Get center time of each word in frames
        word_centers = \
            word_bounds[0] + (word_bounds[1] - word_bounds[0]) / 2.

        # Get frame centers
        frame_centers = .5 + torch.arange(features.shape[-1])

        # Load prominence
        if emphases.PROMINENCE_FEATURE:
            prominence = torch.load(self.cache / 'prominence' / f'{stem}.pt')
            # Interpolate the prominence values
            prominence = emphases.interpolate(
                frame_centers[None],
                word_centers[None],
                prominence)
            features = torch.cat((features, prominence[None, :]), dim=1)

        # Load per-word ground truth emphasis scores
        scores = torch.load(self.cache / 'scores' / f'{stem}.pt')[None]

        # Maybe interpolate scores for framewise model
        if emphases.METHOD in ['framewise'] and not emphases.MODEL_TO_WORDS:

            # Interpolate
            scores = emphases.interpolate(
                frame_centers[None],
                word_centers[None],
                scores)
            scores = torch.clamp(scores, min=0)

        return features, scores, word_bounds, alignment, audio, stem

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)

    def buckets(self):
        """Partition indices into buckets based on length for sampling"""
        if len(self) < emphases.BUCKETS:
            #Just give each individually if not enough to get the right number of buckets
            buckets = [np.array([i]) for i in range(0, len(self))]
        else:
            # Get the size of a bucket
            size = len(self) // emphases.BUCKETS

            # Get indices in order of length
            indices = np.argsort(self.lengths)

            buckets = [indices[i:i + size] for i in range(0, len(self), size)]
        buckets = [(self.lengths[bucket[-1]], bucket) for bucket in buckets]

        # Split into buckets based on length
        return buckets