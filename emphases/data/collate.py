import torch

import emphases


###############################################################################
# Batch collation
###############################################################################


def collate(batch):
    """Batch collation"""
    # Unpack
    audio, prominence, word_bounds = zip(*batch)

    # Get maximum length prominence
    lengths = torch.tensor(
        [a.shape[-1] // emphases.HOPSIZE for a in audio],
        dtype=torch.long)
    max_length = lengths.max().values.item()

    # Pad audio and prominence to maximum length
    batch_size = audio.shape[0]
    padded_audio = torch.empty(
        (batch_size, 1, max_length * emphases.HOPSIZE),
        dtype=torch.float)
    padded_prominence = torch.empty(
        (batch_size, 1, max_length))
    for i, (a, p, l) in enumerate(zip(audio, prominence, lengths)):
        padded_audio[i, :, :l * emphases.HOPSIZE] = a[i]
        padded_prominence[i, :, :l] = p[i]

    return padded_audio, padded_prominence, word_bounds, lengths
