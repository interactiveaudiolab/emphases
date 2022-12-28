import torch

import emphases


###############################################################################
# Interpolation
###############################################################################


def interpolate(frame_times, word_times, scores):
    """Interpolate emphasis scores to the frame rate"""
    method = emphases.INTERPOLATION_METHOD
    if method == 'linear':
        return linear(frame_times, word_times, scores)
    elif method == 'nearest':
        return nearest(frame_times, word_times, scores)
    else:
        raise ValueError(f'Interpolation method {method} is not defined')


def linear(frame_times, word_times, scores):
    """Linear interpolation"""
    # Compute slope and intercept at original times
    slope = (
        (scores[:, 1:] - scores[:, :-1]) /
        (word_times[:, 1:] - word_times[:, :-1]))
    intercept = scores[:, :-1] - (slope.mul(word_times[:, :-1]))

    # Compute indices at which we evaluate points
    indices = torch.sum(
        torch.ge(frame_times[:, :, None], word_times[:, None, :]), -1) - 1
    indices = torch.clamp(indices, 0, slope.shape[-1] - 1)

    # Compute index into parameters
    line_idx = torch.arange(indices.shape[0], device=indices.device)
    line_idx = line_idx.expand(indices.shape)

    # Interpolate
    return (
        slope[line_idx, indices].mul(frame_times) +
        intercept[line_idx, indices])


def nearest(frame_times, _, scores):
    """Nearest neighbors interpolation"""
    return scores[:, torch.round(frame_times).to(torch.long)]
