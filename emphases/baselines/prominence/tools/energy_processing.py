import numpy as np
import scipy.signal
from . import filter, misc, smooth_and_interp


def extract_energy(
    waveform,
    fs=16000,
    min_freq=200,
    max_freq=3000,
    method='rms',
    target_rate=200):
    # Get butterworth bandpass filter parameters
    lp_waveform =  filter.butter_bandpass_filter(
        waveform,
        min_freq,
        max_freq,
        fs,
        order=5)

    # hilbert is sometimes prohibitively slow, should pad to next power of two
    if method == 'hilbert':
        energy = abs(scipy.signal.hilbert(lp_waveform))

    elif method == 'true_envelope':

        # window should be about one pitch period, ~ 5 ms
        win = .005 *fs
        energy = smooth_and_interp.peak_smooth(abs(lp_waveform), 200, win)

    elif method == 'rms':
        energy = np.sqrt(lp_waveform ** 2)

    # TODO
    return misc.resample(energy, fs, target_rate)
