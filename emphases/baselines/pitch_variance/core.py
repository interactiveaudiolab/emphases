import penn
import torch

import emphases


###############################################################################
# Pitch variance method
###############################################################################


def infer(alignment, audio, sample_rate, gpu=None):
    """Compute per-word emphasis scores using pitch variance method"""
    # Infer pitch and periodicity
    pitch, _ = penn.from_audio(
        audio,
        sample_rate,
        hopsize=emphases.HOPSIZE_SECONDS,
        fmin=emphases.FMIN,
        fmax=emphases.FMAX,
        pad=True,
        interp_unvoiced_at=emphases.VOICED_THRESHOLD,
        gpu=gpu)

    # Compute pitch statistics in base-two log-space
    pitch = torch.log2(pitch)

    # Compute utterance statistics
    utterance_spread = spread(pitch)

    # Compute word statistics
    word_spreads = []
    for word in alignment:
        start = int(emphases.convert.seconds_to_frames(word.start()))
        end = int(emphases.convert.seconds_to_frames(word.end()))
        word_spreads.append(spread(pitch[0, start:end]))
    word_spreads = torch.tensor(
        word_spreads,
        dtype=pitch.dtype,
        device=pitch.device)[None]

    # Zero-center
    return word_spreads - utterance_spread


###############################################################################
# Utilities
###############################################################################


def spread(pitch):
    """Compute pitch spread"""
    return torch.quantile(pitch, .95) - torch.quantile(pitch, .05)
