import torch

import emphases


###############################################################################
# Batch collation
###############################################################################


def collate(batch):
    """Batch collation"""
    # Unpack
    audio, prominence, word_bounds = zip(*batch)

    # Get word lengths
    word_lengths = torch.tensor(
        [p.shape[-1] for p in prominence],
        dtype=torch.long)
    max_word_length = word_lengths.max().values.item()

    # Get frame lengths
    frame_lengths = torch.tensor(
        [a.shape[-1] // emphases.HOPSIZE for a in audio],
        dtype=torch.long)
    max_frame_length = frame_lengths.max().values.item()

    # Allocate padded tensors
    batch_size = audio.shape[0]
    padded_audio = torch.empty(
        (batch_size, 1, max_frame_length * emphases.HOPSIZE),
        dtype=torch.float)
    padded_prominence = torch.empty(
        (batch_size, 1, max_word_length))

    # Place batch in padded tensors
    iterator = enumerate(zip(audio, prominence, frame_lengths, word_lengths))
    for i, (a, p, fl, wl) in iterator:
        padded_audio[i, :, :fl * emphases.HOPSIZE] = a[i]
        padded_prominence[i, :, :wl] = p[i]

    return (
        padded_audio,
        padded_prominence,
        word_bounds,
        word_lengths,
        frame_lengths)
