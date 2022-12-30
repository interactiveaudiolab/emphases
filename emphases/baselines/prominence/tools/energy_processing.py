import numpy as np
from . import filter, misc

import emphases

def extract_energy(
    waveform,
    fs=16000,
    min_freq=emphases.baselines.prominence.ENERGY_MIN,
    max_freq=emphases.baselines.prominence.ENERGY_MAX,
    target_rate=200):
    # Get butterworth bandpass filter parameters
    lp_waveform =  filter.butter_bandpass_filter(
        waveform,
        min_freq,
        max_freq,
        fs,
        order=5)

    # Compute energy
    energy = np.sqrt(lp_waveform ** 2)

    # TODO
    return misc.resample(energy, fs, target_rate)
