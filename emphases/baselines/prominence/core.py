import os
import yaml
from collections import defaultdict

import numpy as np

from emphases.baselines.prominence.tools import (
    energy_processing,
    f0_processing,
    duration_processing,
    misc,
    smooth_and_interp,
    cwt_utils,
    loma,
    lab)


###############################################################################
# Prominence API
###############################################################################


def from_audio(audio, sample_rate):
    """Compute prominence from audio"""
    # TODO
    pass


###############################################################################
# Utilities
###############################################################################


def apply_configuration(current_configuration, updating_part):
    """Utils to update the current configuration using the updating part

    Parameters
    ----------
    current_configuration: dict
        The current state of the configuration

    updating_part: dict
        The information to add to the current configuration

    Returns
    -------
    dict
       the updated configuration
    """
    if not isinstance(current_configuration, dict):
        return updating_part

    if current_configuration is None:
        return updating_part

    if updating_part is None:
        return current_configuration

    for k in updating_part:
        if k not in current_configuration:
            current_configuration[k] = updating_part[k]
        else:
            current_configuration[k] = apply_configuration(
                current_configuration[k],
                updating_part[k])

    return current_configuration


def analysis(alignment, audio, sample_rate, cfg):
    # Convert to numpy
    audio = audio.numpy()

    # Compute energy
    energy = energy_processing.extract_energy(
        audio,
        sample_rate,
        cfg['energy']['band_min'],
        cfg['energy']['band_max'],
        cfg['energy']['calculation_method'])
    energy = np.cbrt(energy + 1)
    if cfg["energy"]["smooth_energy"]:
        energy = smooth_and_interp.peak_smooth(energy, 30, 3)
        energy = smooth_and_interp.smooth(energy, 10)

    # Compute pitch
    pitch = f0_processing.extract_f0(
        audio,
        sample_rate,
        f0_min=cfg['f0']['min_f0'],
        f0_max=cfg['f0']['max_f0'],
        voicing=cfg['f0']['voicing_threshold'],
        configuration=cfg['f0']['pitch_tracker'])
    pitch = f0_processing.process(pitch)

    # TODO - replace with pypar alignment for computing duration
    tiers = []
    rate = np.zeros(len(pitch))

    basename = os.path.splitext(os.path.basename(input_file))[0]
    grid =  os.path.join(annotation_dir, '%s.TextGrid' % basename)
    if os.path.exists(grid):
        tiers = lab.read_textgrid(grid)
    else:
        grid =  os.path.join(annotation_dir, '%s.lab' % basename)
        if not os.path.exists(grid):
            raise Exception('There is no annotations associated with %s' % input_file)
        tiers = lab.read_htk_label(grid)

    # Extract duration
    if len(tiers) > 0:
        dur_tiers = []
        for level in cfg['duration']['duration_tiers']:
            assert(level.lower() in tiers), level+' not defined in tiers: check that duration_tiers in config match the actual textgrid tiers'
            dur_tiers.append(tiers[level.lower()])

    if not cfg['duration']['acoustic_estimation']:
        rate = duration_processing.get_duration_signal(
            dur_tiers,
            weights=cfg['duration']['weights'],
            linear=cfg['duration']['linear'],
            sil_symbols=cfg['duration']['silence_symbols'],
            bump=cfg['duration']['bump'])

    else:

        rate = duration_processing.get_rate(energy)
        rate = smooth_and_interp.smooth(rate, 30)

    if cfg['duration']['delta_duration']:
            rate = np.diff(rate)

    # Slice features
    min_length = np.min([len(pitch), len(energy), len(rate)])
    pitch = pitch[:min_length]
    energy = energy[:min_length]
    rate = rate[:min_length]

    # Combine features
    weights = cfg['feature_combination']['weights']
    if cfg['feature_combination']['type'] == 'product':

        # Exponential weighting
        pitch = misc.normalize_minmax(pitch) ** weights['f0']
        energy = misc.normalize_minmax(energy) ** weights['energy']
        rate =  misc.normalize_minmax(rate) ** weights['duration']
        combined = pitch * energy * rate

    else:

        # Multiplicative weighting
        combined = (
            misc.normalize_std(pitch) * weights['f0'] +
            misc.normalize_std(energy) * weights['energy'] +
            misc.normalize_std(rate) * weights['duration'])

    # Maybe perform bias correction
    if cfg['feature_combination']['detrend']:
         combined = smooth_and_interp.remove_bias(combined, 800)

    # Normalize
    combined = misc.normalize_std(combined)

    # Continuous wavelet transform analysis
    cwt, scales, freqs = cwt_utils.cwt_analysis(
        combined,
        mother_name=cfg['wavelet']['mother_wavelet'],
        period=cfg['wavelet']['period'],
        num_scales=cfg['wavelet']['num_scales'],
        scale_distance=cfg['wavelet']['scale_distance'],
        apply_coi=False)
    cwt = np.real(cwt)
    scales *= 200

    # Compute lines of maximum amplitude
    labels = tiers[cfg['labels']['annotation_tier'].lower()]

    # Get scale at average length of selected tier
    scale_dist = cfg['wavelet']['scale_distance']
    scales = 1. / freqs * 200 * .5
    unit_scale = misc.get_best_scale(scales, labels)

    # Define the scale information
    # Three octaves down from average unit length
    pos_loma_start_scale = \
        unit_scale + int(cfg['loma']['prom_start'] / scale_dist)
    pos_loma_end_scale = unit_scale + int(cfg['loma']['prom_end'] / scale_dist)

    # Two octaves down
    neg_loma_start_scale = unit_scale + int(cfg['loma']['boundary_start'] / scale_dist)

    # One octave up
    neg_loma_end_scale = unit_scale + int(cfg['loma']['boundary_end'] / scale_dist)

    pos_loma = loma.get_loma(cwt, scales, pos_loma_start_scale, pos_loma_end_scale)
    neg_loma = loma.get_loma(-cwt, scales, neg_loma_start_scale, neg_loma_end_scale)
    max_loma = loma.get_prominences(pos_loma, labels)

    prominences = np.array(max_loma)
    boundaries = np.array(loma.get_boundaries(max_loma, neg_loma, labels))

    return prominences, boundaries


def main(args):
    """Main entry function"""
    # TODO
    configuration = defaultdict()
    config_base_path = "/home/pranav/prominence-estimation-exp/emphases/emphases/assets/configs/prominence"
    with open(config_base_path + "/default.yaml", 'r') as f:
        configuration = apply_configuration(configuration, defaultdict(lambda: False, yaml.load(f, Loader=yaml.FullLoader)))

    if args.config:
        with open(args.config, 'r') as f:
            configuration = apply_configuration(configuration, defaultdict(lambda: False, yaml.load(f, Loader=yaml.FullLoader)))
