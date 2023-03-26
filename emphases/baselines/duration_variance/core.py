import torch


###############################################################################
# Duration variance baseline method
###############################################################################


def infer(alignment):
    """Compute per-word emphasis scores using duration variance method"""
    # Average duration of phonemes in the sentence
    average_phoneme_duration = alignment.duration() / len(alignment.phonemes())

    # Average duration of phonemes in each word
    average_duration_per_word = torch.tensor([
        word.duration() / len(word) for word in alignment])

    # Zero-center
    # TODO - scale using statistics computed over training dataset
    return average_duration_per_word - average_phoneme_duration
