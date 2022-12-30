import numpy as np
from scipy import interpolate


def remove_bias(params, win_len=300):
    return params - smooth(params, win_len)


def interpolate_zeros(params, method='pchip', min_val=0):
    """
    Interpolate 0 values
    :param params: 1D data vector
    :param method:
    :param factor: factor for interpolation (must be integer)
    :return: interpolated 1D vector by a given factor
    """
    voiced = np.array(params, float)
    for i in range(0, len(voiced)):
        if voiced[i] == min_val:
            voiced[i] = np.nan

    if np.isnan(voiced[-1]):
        voiced[-1] = np.nanmin(voiced)
    if np.isnan(voiced[0]):
        voiced[0] = np.nanmean(voiced)

    not_nan = np.logical_not(np.isnan(voiced))

    indices = np.arange(len(voiced))
    if method == 'spline':
        interp = interpolate.UnivariateSpline(
            indices[not_nan],
            voiced[not_nan],
            k=2,
            s=0)
        # return voiced parts intact
        smoothed = interp(indices)
        for i in range(0, len(smoothed)):
            if not np.isnan(voiced[i]):
                smoothed[i] = params[i]
        return smoothed

    elif method == 'pchip':
        interp = interpolate.pchip(indices[not_nan], voiced[not_nan])
    else:
        interp = interpolate.interp1d(
            indices[not_nan],
            voiced[not_nan],
            method)
    return interp(indices)


def smooth(params, win, type='HAMMING'):
    """gaussian type smoothing, convolution with hamming window"""
    win = int(win + .5)
    if win >= len(params) - 1:
        win = len(params) - 1

    if win % 2 == 0:
        win += 1

    s = np.r_[params[win - 1:0:-1], params, params[-1:-win:-1]]

    if type == 'HAMMING':
        w = np.hamming(win)
    else:
        w = np.ones(win)

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(win / 2):-int(win / 2)]


def peak_smooth(params, max_iter, win, min_win=2, voicing=[]):
    """Iterative smoothing while preserving peaks, 'true envelope' -style"""
    smoothed = np.array(params)
    win_reduce = np.exp(np.linspace(np.log(win), np.log(min_win), max_iter))

    for i in range(0, max_iter):

        smoothed = np.maximum(params, smoothed)

        if len(voicing) > 0:
            smoothed = smooth(smoothed, int(win + .5))
            smoothed[voicing > 0] = params[voicing > 0]
        else:
            smoothed = smooth(smoothed, int(win + .5), type='rectangle')

        win = win_reduce[i]

    return smoothed
