import numpy as np


def match_length(sig_list):
    """Reduce length of all signals to a the minimum one.

    Parameters
    ----------
    sig_list: list
        List of signals which are 1D array of samples.

    """
    length = min(map(len, sig_list))
    for i in range(0, len(sig_list)):
        sig_list[i] = sig_list[i][:int(length)]
    return sig_list


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
    """Find the scale whose width is the closes to the average unit length represented in the labels

    Parameters
    ----------
    scales: 1D arraylike
        The scale indices
    labels: list of tuple (float, float, string)
        List of labels which are lists of 3 elements [start, end, description]

    Returns
    -------
    int
        the index of the best scale
    """
    mean_length = 0
    for l in labels:
        mean_length += l[1] - l[0]
    mean_length /= len(labels)
    dist = scales - mean_length
    return np.argmin(np.abs(dist))


def normalize_minmax(params, epsilon=0.1):
    """Normalize parameters into a 0,1 scale

    Parameters
    ----------
    params: arraylike
        The parameters to normalize.
    epsilon: float
        The epsilon to deal with numerical stability

    Returns
    ------
    arraylike
        the normalized parameters
    """
    return (params - min(params) + epsilon) / (max(params) - min(params))


def normalize_std(params, std=0.):
    """Normalize parameters using a z-score paradigm

    Parameters
    ----------
    params: arraylike
        The parameters to normalize.
    std: float
        A given standard deviation. If 0, the standard deviation is computed on the params. (Default: 0)

    Returns
    ------
    arraylike
        the normalized parameters
    """
    if std == 0.:
        std = np.nanstd(params)
    if std < 0.00001:
        return np.zeros(len(params))
    return (params - np.nanmean(params)) / std
