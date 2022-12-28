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

        # Load audio
        audio = emphases.load.audio(self.cache / 'audio' / f'{stem}.wav')

        # Load mels
        mels = torch.tensor(self.cache / 'mels' / f'{stem}.pt')

        # Load alignment
        alignment = pypar.Alignment(
            self.cache / 'alignment' / f'{stem}.TextGrid')

        # Load per-word ground truth emphasis scores
        scores = torch.load(self.cache / 'scores' / f'{stem}.pt')

        # Maybe interpolate scores for framewise model
        if emphases.METHOD == 'framewise':

            # TODO - interpolate
            pass

        # # REFACTOR
        # # updating word bounds with padding on silent time stamps
        # wb_prom_pairs = []
        # audio_len = audio.shape[-1]

        # if word_bounds[0][0]!=0:
        #     wb_prom_pairs.append([(0, word_bounds[0][0]), 0])

        # for idx in range(len(word_bounds)):
        #     wb_prom_pairs.append([word_bounds[idx], scores[idx].item()])
        #     if idx+1<len(word_bounds):
        #         if word_bounds[idx][-1]!=word_bounds[idx+1][0]:
        #             start = word_bounds[idx][-1]
        #             end = word_bounds[idx+1][0]
        #             wb_prom_pairs.append([(start, end), 0])

        # # generating interpolated prom vector
        # prom_extended = []
        # for wb in wb_prom_pairs:
        #     start, end = wb[0][0], wb[0][1]
        #     prom_extended.extend([wb[-1]]*(end-start)*emphases.HOPSIZE)

        # if word_bounds[-1][-1]!=audio_len//emphases.HOPSIZE:
        #     pad_len = audio_len - len(prom_extended)
        #     prom_extended.extend([0]*pad_len)

        # prom_extended = torch.tensor(prom_extended)

        # # frame based scores values
        # if emphases.INTERPOLATION=='nearest_neighbour':
        #     interpolated_prom_values = nearest_neighbour_interpolation(audio, word_bounds, scores)
        # elif emphases.INTERPOLATION=='linear':
        #     interpolated_prom_values = linear_interpolation(audio, word_bounds, scores)

        return alignment, audio, mels, scores, stem

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)
