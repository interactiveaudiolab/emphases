import numpy as np

import emphases

from emphases.baselines.prominence.tools import (
    energy_processing,
    f0_processing,
    duration_processing,
    misc,
    smooth_and_interp,
    cwt_utils,
    loma)


###############################################################################
# Prominence API
###############################################################################


def infer(alignment, audio, sample_rate):
    """Compute per-word prominence from alignment and audio"""
    # Convert to numpy
    audio = audio.numpy()

    # Compute energy
    energy = energy_processing.extract_energy(
        audio,
        sample_rate)
    energy = np.cbrt(energy + 1)

    # Smooth energy
    energy = smooth_and_interp.peak_smooth(energy, 30, 3)
    energy = smooth_and_interp.smooth(energy, 10)

    # Compute pitch
    pitch = f0_processing.extract_f0(audio, sample_rate)
    pitch = f0_processing.process(pitch)

    # Extract duration
    duration = duration_processing.get_duration_signal(
        alignment,
        weights=[.5, .5])

    # Slice features
    min_length = np.min([len(pitch), len(energy), len(duration)])
    pitch = pitch[:min_length]
    energy = energy[:min_length]
    duration = duration[:min_length]

    # Combine features
    combined = (
        emphases.baselines.prominence.PITCH_WEIGHT * normalize(pitch) +
        emphases.baselines.prominence.ENERGY_WEIGHT * normalize(energy) +
        emphases.baselines.prominence.DURATION_WEIGHT * normalize(duration))
    combined = normalize(smooth_and_interp.remove_bias(combined, 800))

    # Continuous wavelet transform analysis
    cwt, scales, freqs = cwt_utils.cwt_analysis(
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

    # Get scale at average length of selected tier
    # TODO
    words = tiers['words']
    scales = 1. / freqs * 200 * .5
    unit_scale = misc.get_best_scale(scales, words)

    # Define the line of maximum amplitude scale information
    pos_loma_start_scale = unit_scale + int(
        emphases.baselines.prominence.LOMA_PROMINENCE_START / scale_dist)
    pos_loma_end_scale = unit_scale + int(
        emphases.baselines.prominence.LOMA_PROMINENCE_END / scale_dist)
    neg_loma_start_scale = unit_scale + int(
        emphases.baselines.prominence.LOMA_BOUNDARY_START / scale_dist)
    neg_loma_end_scale = unit_scale + int(
        emphases.baselines.prominence.LOMA_BOUNDARY_END / scale_dist)

    # Get line of maximum amplitude
    pos_loma = loma.get_loma(
        cwt,
        scales,
        pos_loma_start_scale,
        pos_loma_end_scale)
    neg_loma = loma.get_loma(
        -cwt,
        scales,
        neg_loma_start_scale,
        neg_loma_end_scale)
    max_loma = loma.get_prominences(pos_loma, words)

    # Get prominence and boundary
    prominences = np.array(max_loma)
    boundaries = np.array(loma.get_boundaries(max_loma, neg_loma, words))

    return prominences, boundaries


###############################################################################
# Utilities
###############################################################################


def normalize(features):
    """Normalize features"""
    return (features - np.nanmean(features)) / (np.nanstd(features) + 1e-7)
