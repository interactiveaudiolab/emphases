from . import smooth_and_interp, misc
import numpy as np


###############################################################################
# Constants
###############################################################################


SIL_SYMBOLS = [
    '#',
    '!pau',
    'sp',
    '<s>',
    'pau',
    '!sil',
    'sil',
    '',
    ' ',
    '<p>',
    '<p:>',
    '.',
    ',',
    '?']


###############################################################################
# Duration
###############################################################################


def _get_dur_stats(labels, linear=False, sil_symbols=[]):
    durations = []
    for i in range(len(labels)):
        (st,en, unit) = labels[i]
        if unit.lower() not in sil_symbols:
            dur = en-st
            if not linear:
                dur = np.log(dur + 1.)
            durations.append(dur)
    durations = np.array(durations)
    return np.min(durations), np.max(durations), np.mean(durations)


def get_rate(params, hp=10, lp=150):
    """
    estimation of speech rate as a center of gravity of wavelet spectrum
    similar to method described in "Boundary Detection using Continuous Wavelet Analysis" (2016)
    """
    from . import cwt_utils

    params = smooth_and_interp.smooth(params, hp)
    params -= smooth_and_interp.smooth(params, lp)

    wavelet_matrix, *_  = cwt_utils.cwt_analysis(
        params,
        mother_name="Morlet",
        num_scales=80,
        scale_distance=0.1,
        apply_coi=True,
        period=2)
    wavelet_matrix = abs(wavelet_matrix)

    rate = np.zeros(len(params))

    for i in range(0,wavelet_matrix.shape[1]):
        frame_en = np.sum(wavelet_matrix[:, i])
        # center of gravity
        rate[i] = np.nonzero(
            wavelet_matrix[:, i].cumsum() >= frame_en * .5)[0].min()

    return smooth_and_interp.smooth(rate, 30)


def duration(
    labels,
    rate=200,
    linear=False,
    bump=False,
    sil_symbols=SIL_SYMBOLS):
    """Construct duration signal from labels"""
    dur = np.zeros(len(labels))
    params = np.zeros(int(labels[-1][1] * rate))
    prev_end = 0
    min_dur, *_ = _get_dur_stats(labels, linear, sil_symbols)

    for i in range(0, len(labels)):
        st, en, unit = labels[i]
        st *= rate
        en *= rate
        dur[i] = en-st
        if not linear:
            dur[i] = np.log(dur[i] + 1.)

        if unit.lower() in sil_symbols:
            dur[i] = min_dur

        # skip very short units, likely labelling errors
        if en <= st + .01:
            continue

        # unit duration -> height of the duration contour in the middle of the unit
        params[int(st + (en - st) / 2.)] = dur[i]

        # "bump" -> emphasize difference between adjacent unit durations
        if i > 0 and bump:
            params[int(st)] = \
                (dur[i] + dur[i - 1]) / 2. - (abs(dur[i] - dur[i - 1]))

        # Handle gaps in labels similarly to silences
        if st > prev_end and i > 1:
            params[int(prev_end + (st - prev_end) / 2.0)] = min_dur
        prev_end = en

    # set endpoints to mean in order to avoid large "valleys"
    params[0] = np.mean(dur)
    params[-1] = np.mean(dur)

    # make continous duration contour and smooth a bit
    params = smooth_and_interp.interpolate_zeros(params, 'pchip')
    params = smooth_and_interp.smooth(params, 20)

    return params


def get_duration_signal(
    tiers=[],
    weights=[],
    sil_symbols=SIL_SYMBOLS,
    rate=1,
    linear=True,
    bump=False):
    """
    Construct duration contour from labels. If many tiers are selected,
    construct contours for each tier and return a weighted sum of those
    """
    durations = []
    for t in tiers:
        durations.append(
            misc.normalize_std(
                duration(
                    t,
                    rate=rate,
                    sil_symbols=sil_symbols,
                    linear=linear,
                    bump=bump)))
    durations = misc.match_length(durations)
    sum_durations = np.zeros(len(durations[0]))
    if len(weights) != len(tiers):
        weights = np.ones(len(tiers))
    for i in range(len(durations)):
        sum_durations += durations[i] * weights[i]
    return sum_durations
