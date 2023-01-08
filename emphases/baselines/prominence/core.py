import fractions

import numpy as np
import torch
from scipy.signal import resample_poly

import emphases


###############################################################################
# Prominence API
###############################################################################


def infer(alignment, audio, sample_rate):
    """Compute per-word prominence from alignment and audio"""
    # Convert to numpy
    audio = audio.numpy()[0]

    # Compute energy
    energy = emphases.baselines.prominence.energy_processing.extract_energy(
        audio,
        sample_rate)
    energy = np.cbrt(energy + 1)

    # Smooth energy
    energy = emphases.baselines.prominence.smooth_and_interp.peak_smooth(energy, 30, 3)
    energy = emphases.baselines.prominence.smooth_and_interp.smooth(energy, 10)

    # Compute pitch
    pitch = emphases.baselines.prominence.pitch_tracker.inst_freq_pitch(audio, sample_rate)
    pitch = emphases.baselines.prominence.f0_processing.process(pitch)

    # Extract duration
    duration = emphases.baselines.prominence.duration_processing.get_duration_signal(
        alignment,
        weights=[.5, .5])

    # Slice features
    min_length = np.min([len(pitch), len(energy), len(duration)])
    pitch = pitch[:min_length]
    energy = energy[:min_length]
    duration = duration[:min_length]

    # Combine features
    combined = (
        emphases.PROMINENCE_PITCH_WEIGHT * normalize(pitch) +
        emphases.PROMINENCE_ENERGY_WEIGHT * normalize(energy) +
        emphases.PROMINENCE_DURATION_WEIGHT * normalize(duration))
    combined = normalize(
        emphases.baselines.prominence.smooth_and_interp.remove_bias(combined, 800))

    # Continuous wavelet transform analysis
    cwt, scales, freqs = emphases.baselines.prominence.cwt_utils.cwt_analysis(
        combined,
        mother_name='mexican_hat',
        period=3,
        num_scales=34,
        scale_distance=.25,
        apply_coi=False)
    cwt = np.real(cwt)
    scales *= 200

    # Distance between adjacent scales (.25 means 4 scales per octave)
    scale_dist = .25  # octaves

    # Get scale that minimizes distance with average word length
    average_duration = alignment.end() / len(alignment)
    scales = 1. / freqs * 200 * .5
    scale = np.argmin(np.abs(scales - average_duration))

    # Get line of maximum amplitude
    loma_start = scale + int(emphases.LOMA_START / scale_dist)
    loma_end = scale + int(emphases.LOMA_END / scale_dist)
    loma = emphases.baselines.prominence.loma.get_loma(
        cwt,
        scales,
        loma_start,
        loma_end)

    # Get prominence
    return torch.tensor(
        emphases.baselines.prominence.loma.get_prominences(
            loma,
            alignment))


###############################################################################
# Utilities
###############################################################################


def normalize(features):
    """Normalize features"""
    return (features - np.nanmean(features)) / (np.nanstd(features) + 1e-7)


def resample(signal, original_sample_rate, target_sample_rate):
    """Resample signal"""
    ratio = fractions.Fraction(target_sample_rate, original_sample_rate)
    return resample_poly(signal, ratio.numerator, ratio.denominator)
