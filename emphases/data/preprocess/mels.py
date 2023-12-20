import multiprocessing as mp
import os

import librosa
import torch
import torchutil

import emphases


###############################################################################
# Mel spectrogram
###############################################################################


def from_audio(audio):
    """Compute spectrogram from audio"""
    # Cache hann window
    if (
        not hasattr(from_audio, 'window') or
        from_audio.dtype != audio.dtype or
        from_audio.device != audio.device
    ):
        from_audio.window = torch.hann_window(
            emphases.WINDOW_SIZE,
            dtype=audio.dtype,
            device=audio.device)
        from_audio.dtype = audio.dtype
        from_audio.device = audio.device

    # Pad audio
    size = (emphases.NUM_FFT - emphases.HOPSIZE) // 2
    audio = torch.nn.functional.pad(
        audio,
        (size, size),
        mode='reflect')

    # Compute stft
    stft = torch.stft(
        audio.squeeze(1),
        emphases.NUM_FFT,
        hop_length=emphases.HOPSIZE,
        window=from_audio.window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True)
    stft = torch.view_as_real(stft)[0]

    # Compute magnitude
    spectrogram = torch.sqrt(stft.pow(2).sum(-1) + 1e-6)

    # Convert to mels
    mels = linear_to_mel(spectrogram)

    # Scale to roughly [0, 1]
    if emphases.NORMALIZE:
        return (mels + 10.) / 10.
    return mels


def from_file(audio_file):
    """Load audio and compute mels"""
    audio = emphases.load.audio(audio_file)

    # Compute mels
    return from_audio(audio)


def from_file_to_file(audio_file, output_file):
    """Compute mels from audio file and save to disk"""
    mels = from_file(audio_file)

    # Save to disk
    output_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save(mels, output_file)


def from_files_to_files(audio_files, output_files):
    """Compute mels for many files and save to disk"""
    torchutil.multiprocess_iterator(
        wrapper,
        zip(audio_files, output_files),
        'Preprocessing mels',
        total=len(audio_files),
        num_workers=emphases.NUM_WORKERS)


###############################################################################
# Utilities
###############################################################################


def linear_to_mel(spectrogram):
    # Create mel basis
    if not hasattr(linear_to_mel, 'mel_basis'):
        basis = librosa.filters.mel(
            sr=emphases.SAMPLE_RATE,
            n_fft=emphases.NUM_FFT,
            n_mels=emphases.NUM_MELS)
        basis = torch.from_numpy(basis)
        basis = basis.to(spectrogram.dtype).to(spectrogram.device)
        linear_to_mel.basis = basis

    # Convert to mels
    melspectrogram = torch.matmul(linear_to_mel.basis, spectrogram)

    # Apply dynamic range compression
    return torch.log(torch.clamp(melspectrogram, min=1e-5))

def wrapper(item):
    """Multiprocessing wrapper"""
    from_file_to_file(*item)
