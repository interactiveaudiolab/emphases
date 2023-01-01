import numpy as np

import emphases


def extract_energy(
    waveform,
    sample_rate=16000,
    min_freq=emphases.PROMINENCE_ENERGY_MIN,
    max_freq=emphases.PROMINENCE_ENERGY_MAX,
    frame_rate=200):
    # Get butterworth bandpass filter parameters
    lp_waveform =  emphases.baselines.prominence.filter.butter_bandpass_filter(
        waveform,
        min_freq,
        max_freq,
        sample_rate,
        order=5)

    # Compute energy
    energy = np.sqrt(lp_waveform ** 2)

    # Resample to frame rate
    return emphases.baselines.prominence.resample(energy, sample_rate, frame_rate)
