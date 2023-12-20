import multiprocessing as mp
import warnings

import librosa
import numpy as np
import penn
import torch
import torchutil

import emphases


###############################################################################
# Interface
###############################################################################


def from_audio(audio, sample_rate=emphases.SAMPLE_RATE):
    """Compute mels from audio"""
    # Mayble resample
    audio = emphases.resample(audio, sample_rate)

    # Compute loudness
    return a_weighted(audio, sample_rate, hop_length=emphases.HOPSIZE)


def from_file(audio_file):
    """Load audio and compute mels"""
    audio = emphases.load.audio(audio_file)

    # Compute loudness
    return from_audio(audio)


def from_file_to_file(audio_file, output_file):
    """Compute loudness from audio file and save to disk"""
    loudness = from_file(audio_file)

    # Save to disk
    output_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save(loudness, output_file)


def from_files_to_files(audio_files, output_files):
    """Compute loudness for many files and save to disk"""
    torchutil.multiprocess_iterator(
        wrapper,
        zip(audio_files, output_files),
        'Preprocessing a-weighted loudness',
        total=len(audio_files),
        num_workers=emphases.NUM_WORKERS)


###############################################################################
# Loudness
###############################################################################


def a_weighted(audio, sample_rate, hop_length=None, pad=False):
    """Retrieve the per-frame loudness"""
    # Save device
    device = audio.device

    # Default hop length of 10 ms
    hop_length = sample_rate // 100 if hop_length is None else hop_length

    if audio.dim() == 2:
        audio = audio[:, None, :]
    elif audio.dim() == 1:
        audio = audio[None, None, :]

    # Pad audio
    p = (emphases.NUM_FFT - emphases.HOPSIZE) // 2
    audio = torch.nn.functional.pad(audio, (p, p), "reflect").squeeze(1)

    # Convert to numpy
    audio = audio.detach().cpu().numpy().squeeze(0)

    # Cache weights
    if not hasattr(a_weighted, 'weights'):
        a_weighted.weights = perceptual_weights()

    # Take stft
    stft = librosa.stft(
        audio,
        n_fft=penn.WINDOW_SIZE,
        hop_length=hop_length,
        win_length=penn.WINDOW_SIZE,
        center=pad,
        pad_mode='constant')

    # Compute magnitude on db scale
    db = librosa.amplitude_to_db(np.abs(stft))

    # Apply A-weighting
    weighted = db + a_weighted.weights

    # Threshold
    weighted[weighted < emphases.MIN_DB] = emphases.MIN_DB

    # Average over weighted frequencies
    loudness = torch.from_numpy(weighted.mean(axis=0)).float().to(device)[None]

    # Scale to roughly [0, 1]
    if emphases.NORMALIZE:
        return (loudness + 100.) / 100.
    return loudness


def perceptual_weights():
    """A-weighted frequency-dependent perceptual loudness weights"""
    frequencies = librosa.fft_frequencies(
        sr=penn.SAMPLE_RATE,
        n_fft=penn.WINDOW_SIZE)

    # A warning is raised for nearly inaudible frequencies, but it ends up
    # defaulting to -100 db. That default is fine for our purposes.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return librosa.A_weighting(frequencies)[:, None] - emphases.REF_DB

def wrapper(item):
    """Multiprocessing wrapper"""
    from_file_to_file(*item)
