import numpy as np

import emphases


###############################################################################
# Constants
###############################################################################


SILENCE_SYMBOLS = [
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
    '?',
    '<silent>']


###############################################################################
# Duration
###############################################################################


def _get_dur_stats(labels, rate=200):
    durations = []
    for i in range(len(labels)):
        st, en, unit = labels[i]
        st *= rate
        en *= rate
        if unit.lower() not in SILENCE_SYMBOLS:
            dur = en - st
            dur = np.log(dur + 1.)
            durations.append(dur)
    durations = np.array(durations)
    return np.min(durations), np.max(durations), np.mean(durations)


def get_rate(params, hp=10, lp=150):
    """
    estimation of speech rate as a center of gravity of wavelet spectrum
    similar to method described in "Boundary Detection using Continuous Wavelet Analysis" (2016)
    """
    params = emphases.baselines.prominence.smooth_and_interp.smooth(params, hp)
    params -= emphases.baselines.prominence.smooth_and_interp.smooth(params, lp)

    wavelet_matrix, *_ = emphases.baselines.prominence.cwt_utils.cwt_analysis(
        params,
        mother_name='Morlet',
        num_scales=80,
        scale_distance=.1,
        apply_coi=True,
        period=2)
    wavelet_matrix = abs(wavelet_matrix)

    rate = np.zeros(len(params))

    for i in range(0,wavelet_matrix.shape[1]):
        frame_en = np.sum(wavelet_matrix[:, i])
        # center of gravity
        rate[i] = np.nonzero(
            wavelet_matrix[:, i].cumsum() >= frame_en * .5)[0].min()

    return emphases.baselines.prominence.smooth_and_interp.smooth(rate, 30)


def duration(labels, rate=200):
    """Construct duration signal from labels"""
    dur = np.zeros(len(labels))
    params = np.zeros(int(labels[-1][1] * rate))
    prev_end = 0
    min_dur, *_ = _get_dur_stats(labels, rate=200)
    for i in range(0, len(labels)):
        st, en, unit = labels[i]
        st *= rate
        en *= rate
        dur[i] = en - st
        dur[i] = np.log(dur[i] + 1.)

        if unit.lower() in SILENCE_SYMBOLS:
            dur[i] = min_dur

        # skip very short units, likely labelling errors
        if en <= st + .01:
            continue

        # unit duration -> height of the duration contour in the middle of the unit
        index = min(len(params) - 1, int(st + (en - st) / 2.))
        params[index] = dur[i]

        # Handle gaps in labels similarly to silences
        if st > prev_end and i > 1:
            params[int(prev_end + (st - prev_end) / 2.)] = min_dur
        prev_end = en

    # set endpoints to mean in order to avoid large "valleys"
    params[0] = np.mean(dur)
    params[-1] = np.mean(dur)

    # make continous duration contour and smooth a bit
    params = emphases.baselines.prominence.smooth_and_interp.interpolate_zeros(params, 'pchip')
    return emphases.baselines.prominence.smooth_and_interp.smooth(params, 20)


def get_duration_signal(
    alignment,
    weights=[],
    rate=1):
    """
    Construct duration contour from labels. If many tiers are selected,
    construct contours for each tier and return a weighted sum of those
    """
    word_tier = [(word.start(), word.end(), str(word)) for word in alignment]
    phoneme_tier = [
        (phoneme.start(), phoneme.end(), str(phoneme))
    for phoneme in alignment.phonemes()]
    tiers = [phoneme_tier, word_tier]

    durations = []

    for tier in tiers:
        durations.append(
            emphases.baselines.prominence.normalize(
                duration(tier, rate=rate)))
    durations = match_length(durations)
    sum_durations = np.zeros(len(durations[0]))
    if len(weights) != len(tiers):
        weights = np.ones(len(tiers))
    for i in range(len(durations)):
        sum_durations += durations[i] * weights[i]
    return sum_durations


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
