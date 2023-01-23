import penn

import torch

import emphases

###############################################################################
# Pitch variance method
###############################################################################


def infer(alignment):
    """Compute per-word emphasis scores using duration variance method"""
    total_phonemes = 0
    total_length = 0
    word_average = torch.zeros(len(alignment)) #Average per-word phoneme duration
    for i in range(len(alignment)):
        #Iterate through words
        word = alignment[i]
        word_sum = 0
        for phoneme in word: #Get duration of each phoneme, add to word and total length
            word_sum += phoneme.duration()
            total_length += phoneme.duration()
        word_average[i] = word_sum / len(word)
        total_phonemes += len(word)
    avg_length = total_length / total_phonemes #Gives sentence average per-phoneme duration
    result = torch.abs(word_average - avg_length)
    return torch.reshape(result, (1, -1))
