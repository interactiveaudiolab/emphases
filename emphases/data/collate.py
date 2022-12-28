import torch

import emphases


###############################################################################
# Batch collation
###############################################################################


def collate(batch):
    """Batch collation"""
    # Unpack
    alignments, word_bounds, audios, mels, scores, stems = zip(*batch)

    # Get word lengths
    word_lengths = torch.tensor(
        [bounds.shape[-1] for bounds in word_bounds],
        dtype=torch.long)
    max_word_length = word_lengths.max().item()

    # Get frame lengths
    frame_lengths = torch.tensor(
        [mel.shape[-1] for mel in mels],
        dtype=torch.long)
    max_frame_length = frame_lengths.max().item()

    # Allocate padded tensors
    padded_bounds = torch.zeros((len(word_bounds), 2, max_word_length))
    padded_audio = torch.zeros(
        (len(audios), 1, max_frame_length * emphases.HOPSIZE))
    padded_scores = torch.zeros((len(scores), 1, max_word_length))
    padded_mels = torch.zeros((len(mels), emphases.NUM_MELS, max_frame_length))

    # Place batch in padded tensors
    iterator = enumerate(
        zip(word_bounds, audios, mels, scores, frame_lengths, word_lengths))
    for i, (bounds, audio, mel, score, frame_length, word_length) in iterator:

        # Pad word bounds
        padded_bounds[i, :, :word_length] = bounds[:, word_length]

        # Pad audio
        end_sample = frame_length * emphases.HOPSIZE
        padded_audio[i, :, :end_sample] = audio[:, :end_sample]

        # Pad scores
        padded_scores[i, :, :word_length] = score[:, :word_length]

        # Pad mels
        padded_mels[i, :, :frame_length] = mel

    # Network output lengths
    if emphases.METHOD == 'framewise':
        output_lengths = frame_lengths
    elif emphases.METHOD == 'wordwise':
        output_lengths = word_lengths
    else:
        raise ValueError(f'Inference method {emphases.METHOD} is not defined')

    # Create mask
    mask = torch.zeros((len(scores), 1, output_lengths))
    for i, length in enumerate(output_lengths):
        mask[:, :, :length] = 1.

    return (
        padded_mels,
        padded_scores,
        padded_bounds,
        word_lengths,
        mask,
        alignments,
        padded_audio,
        stems)
