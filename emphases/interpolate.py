import torch

import emphases


###############################################################################
# Interpolation
###############################################################################


def interpolate(frame_times, word_times, scores):
    """Interpolate emphasis scores to the frame rate"""
    method = emphases.INTERPOLATION
    if method == 'linear':
        return linear(frame_times, word_times, scores)
    elif method == 'nearest':
        return nearest(frame_times, word_times, scores)
    raise ValueError(f'Interpolation method {method} is not defined')


def linear(frame_times, word_times, scores):
    """Linear interpolation"""
    # Compute slope and intercept at original times
    slope = (
        (scores[:, 1:] - scores[:, :-1]) /
        (word_times[:, 1:] - word_times[:, :-1]))
    intercept = scores[:, :-1] - slope.mul(word_times[:, :-1])

    # Compute indices at which we evaluate points
    indices = torch.sum(
        torch.ge(frame_times[:, :, None], word_times[:, None, :]), -1) - 1
    indices = torch.clamp(indices, 0, slope.shape[-1] - 1)

    # Compute index into parameters
    line_idx = torch.linspace(
        0,
        indices.shape[0],
        1,
        device=indices.device).to(torch.long)
    line_idx = line_idx.expand(indices.shape)

    # Interpolate
    return (
        slope[line_idx, indices].mul(frame_times) +
        intercept[line_idx, indices])


def nearest(frame_times, word_times, scores):
    """Nearest neighbors interpolation"""
    # Compute indices at which we evaluate points
    indices = torch.sum(
        torch.ge(frame_times[:, :, None], word_times[:, None, :]), -1) - 1
    indices = torch.clamp(indices, 0, word_times.shape[-1] - 1)

    # Get nearest score
    return torch.index_select(scores, 1, indices[0])
