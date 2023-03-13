import multiprocessing as mp
import os

import librosa
import numpy as np
import torch

import emphases


###############################################################################
# Interface
###############################################################################


def from_audio(audio, sample_rate=emphases.SAMPLE_RATE):
    """Compute mels from audio"""
    # Mayble resample
    audio = emphases.resample(audio, sample_rate)

    # Cache function for computing mels
    if not hasattr(from_audio, 'mels'):
        from_audio.mels = MelSpectrogram()

    # Make sure devices match. No-op if devices are the same.
    from_audio.mels.to(audio.device)

    # Compute mels
    return from_audio.mels(audio)


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
    with mp.get_context('spawn').Pool(os.cpu_count() // 2) as pool:
        pool.starmap(from_file_to_file, zip(audio_files, output_files))


###############################################################################
# Mel spectrogram
###############################################################################


class MelSpectrogram(torch.nn.Module):

    def __init__(self):
        super().__init__()
        window = torch.hann_window(emphases.NUM_FFT, dtype=torch.float)
        # TODO - replace with torchaudio
        mel_basis = librosa.filters.mel(
            sr=emphases.SAMPLE_RATE,
            n_fft=emphases.NUM_FFT,
            n_mels=emphases.NUM_MELS
        ).astype(np.float32)
        mel_basis = torch.from_numpy(mel_basis)
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)

    @property
    def device(self):
        return self.mel_basis.device

    def log_mel_spectrogram(self, audio):
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=emphases.NUM_FFT,
            hop_length=emphases.HOPSIZE,
            window=self.window,
            center=False,
            return_complex=False)

        # Compute magnitude spectrogram
        spectrogram = torch.sqrt(stft.pow(2).sum(-1) + 1e-9)

        # Compute melspectrogram
        melspectrogram = torch.matmul(self.mel_basis, spectrogram)

        # Compute logmelspectrogram
        return torch.log10(torch.clamp(melspectrogram, min=1e-5))

    def forward(self, audio):
        # Ensure correct shape
        if audio.dim() == 2:
            audio = audio[:, None, :]
        elif audio.dim() == 1:
            audio = audio[None, None, :]

        # Pad audio
        p = (emphases.NUM_FFT - emphases.HOPSIZE) // 2
        audio = torch.nn.functional.pad(audio, (p, p), "reflect").squeeze(1)

        # Compute logmelspectrogram
        return self.log_mel_spectrogram(audio)
