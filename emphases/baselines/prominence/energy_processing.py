import numpy as np
from . import filter

import emphases


def extract_energy(
    waveform,
    sample_rate=16000,
    min_freq=emphases.baselines.prominence.ENERGY_MIN,
    max_freq=emphases.baselines.prominence.ENERGY_MAX,
    frame_rate=200):
    # Get butterworth bandpass filter parameters
    lp_waveform =  filter.butter_bandpass_filter(
        waveform,
        min_freq,
        max_freq,
        sample_rate,
        order=5)

    # Compute energy
    energy = np.sqrt(lp_waveform ** 2)

    # Resample to frame rate
    return emphases.baselines.prominence.resample(energy, sample_rate, frame_rate)
