"""dataset.py - data loading"""


import os

import torch

import emphases


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
        # self.stems = emphases.data.partitions(name)[partition]
        self.cache = emphases.CACHE_DIR / name
        self.stems = emphases.load.partition(name)[partition]
        print(self.cache)
        # TODO - Get the length corresponding each stem so the sampler can
        #        use it. Note: you should not load all of the dataset to
        #        determine the lengths. Instead, you can use the file size.
        #        Here is an example that assumes 16-bit audio. It might work
        #        for your purposes.

        audio_files = list([self.cache / 'wavs' / f'{stem}.wav' for stem in self.stems])
        self.lengths = [
            os.path.getsize(audio_file) // 2 for audio_file in audio_files]

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        stem = self.stems[index]

        # TODO - Load from stem
        audio = emphases.load.audio(self.cache / 'wavs' / f'{stem}.wav')
        annotation = emphases.load.read_textgrid(self.cache / 'annotation' / f'{stem}.TextGrid')

        return (audio, annotation)
        # raise NotImplementedError

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)
