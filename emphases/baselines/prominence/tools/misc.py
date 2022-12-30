import numpy as np


def get_peaks(params, threshold=-10):
    """Find the peaks based on the given prosodic parameters.

    Parameters
    ----------
    params: ?
        Prosodic parameters
    threshold: int
        description

    Returns
    -------
    peaks: arraylike
        array of peak values and peak indices
    """
    indices = (np.diff(np.sign(np.diff(params))) < 0).nonzero()[0] + 1
    peaks = params[indices]
    return np.array([peaks[peaks > threshold], indices[peaks > threshold]])


def get_best_scale(scales, labels):
    """Find the scale whose width is closest to the average label length

    Parameters
    ----------
    scales: 1D arraylike
        The scale indices
    labels: list of tuple (float, float, string)
        List of labels which are lists of 3 elements [start, end, description]

    Returns
    -------
    int
        The index of the best scale
    """
    mean_length = 0
    for l in labels:
        mean_length += l[1] - l[0]
    mean_length /= len(labels)
    dist = scales - mean_length
    return np.argmin(np.abs(dist))
