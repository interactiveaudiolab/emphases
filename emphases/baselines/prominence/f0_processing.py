import numpy as np

import emphases


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def _cut_boundary_vals(params, num_vals):
    cutted = np.array(params)
    for i in range(num_vals, len(params) - num_vals):
        if params[i] <= 0 and params[i + 1] > 0:
            for j in range(i, i + num_vals):
                cutted[j] = 0.

        if params[i] > 0 and params[i + 1] <= 0:
            for j in range(i - num_vals, i + 1):
                cutted[j] = 0.

    return cutted


def _remove_outliers(log_pitch):
    fixed = np.array(log_pitch)

    # Remove outlier f0 values from voicing boundaries
    boundary_cut = emphases.baselines.prominence.smooth_and_interp.interpolate_zeros(
        _cut_boundary_vals(fixed, 3),
        'linear')
    interp = emphases.baselines.prominence.smooth_and_interp.interpolate_zeros(fixed, 'linear')
    fixed[abs(interp - boundary_cut) > .1] = 0
    interp = emphases.baselines.prominence.smooth_and_interp.interpolate_zeros(fixed, 'linear')

    # iterative outlier removal
    # 1. compare current contour estimate to a smoothed contour and remove deviates larger than threshold
    # 2. smooth current estimate with shorter window, thighten threshold
    # 3. goto 1.

    # In practice, first handles large scale octave jump type errors,
    # finally small scale 'errors' like consonant perturbation effects and
    # other irregularities in voicing boundaries
    #
    # if this appears to remove too many correct values, increase thresholds
    num_iter = 30
    max_win_len = 100
    min_win_len = 10
    max_threshold = 3.  # threshold with broad window
    min_threshold = .5  # threshold with shorted window

    _std = np.std(interp)
    # do not tie fixing to liveliness of the original
    _std = .3

    win_len = np.exp(
        np.linspace(np.log(max_win_len), np.log(min_win_len), num_iter + 1))
    outlier_threshold = np.linspace(
        _std * max_threshold,
        _std * min_threshold,
        num_iter + 1)
    for i in range(0, num_iter):
        smooth_contour = emphases.baselines.prominence.smooth_and_interp.smooth(interp, win_len[i])
        low_limit = smooth_contour - outlier_threshold[i]
        # bit more careful upwards, not to cut emphases
        hi_limit = smooth_contour + outlier_threshold[i] * 1.5

        # octave jump down fix, more harm than good?
        fixed[interp > hi_limit] = 0
        fixed[interp < low_limit] = 0
        interp = emphases.baselines.prominence.smooth_and_interp.interpolate_zeros(fixed, 'linear')

    return fixed


def _interpolate(f0):
    interp = emphases.baselines.prominence.smooth_and_interp.interpolate_zeros(f0)
    _std = np.std(interp)
    _min = np.min(interp)
    low_limit = emphases.baselines.prominence.smooth_and_interp.smooth(interp, 200) - 1.5 * _std
    low_limit[low_limit < _min] = _min
    hi_limit = emphases.baselines.prominence.smooth_and_interp.smooth(interp, 100) + 2. * _std
    voicing = np.array(f0)
    constrained = np.array(f0)
    constrained = np.maximum(f0, low_limit)
    constrained = np.minimum(constrained, hi_limit)
    interp = emphases.baselines.prominence.smooth_and_interp.peak_smooth(
        constrained,
        100,
        20,
        voicing=voicing)
    # smooth voiced parts a bit too
    return emphases.baselines.prominence.smooth_and_interp.peak_smooth(interp, 3, 2)


def process(f0):
    log_pitch = np.array(f0)
    log_scaled = True
    if np.mean(f0[f0 > 0]) > 20:
        log_scaled = False
        log_pitch[f0 > 0] = np.log(f0[f0 > 0])
        log_pitch[f0 <= 0] = 0

    log_pitch = _remove_outliers(log_pitch)
    log_pitch = _interpolate(log_pitch)

    if not log_scaled:
        return np.exp(log_pitch)
    else:
        return log_pitch
