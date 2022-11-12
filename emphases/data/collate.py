import torch

import emphases

import tqdm

###############################################################################
# Batch collation
###############################################################################


def collate(batch):
    """Batch collation"""
    # Unpack
    audio, mel_spectrogram, prominence, word_bounds = zip(*batch)

    # Get word lengths
    word_lengths = torch.tensor(
        [p.shape[-1] for p in prominence],
        dtype=torch.long)
    max_word_length = word_lengths.max().item()

    # Get frame lengths
    frame_lengths = torch.tensor(
        [a.shape[-1] // emphases.HOPSIZE for a in audio],
        dtype=torch.long)
    max_frame_length = frame_lengths.max().item()

    # Get max time axis mel
    mel_lengths = torch.tensor(
        [mel.shape[-1] for mel in mel_spectrogram], 
        dtype=torch.long)
    max_mel_length = mel_lengths.max().item()

    # Allocate padded tensors
    batch_size = len(audio)
    padded_audio = torch.empty(
        (batch_size, 1, max_frame_length * emphases.HOPSIZE),
        dtype=torch.float)
    padded_prominence = torch.empty(
        (batch_size, 1, max_word_length))
    padded_mel_spectrogram = torch.empty((batch_size, 1, emphases.NUM_MELS, max_mel_length))

    # Place batch in padded tensors
    iterator = enumerate(zip(audio, prominence, mel_spectrogram, frame_lengths, word_lengths, mel_lengths))
    for i, (a, p, mel, fl, wl, ml) in tqdm.tqdm(iterator):
        padded_audio[i, :, :fl * emphases.HOPSIZE] = a[:, :fl * emphases.HOPSIZE]
        padded_prominence[i, :, :wl] = p
        padded_mel_spectrogram[i, :, :, :ml] = mel

    return (
        padded_audio,
        padded_mel_spectrogram,
        padded_prominence,
        word_bounds,
        word_lengths,
        frame_lengths)
