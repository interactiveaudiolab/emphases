import librosa
import numpy as np
import torch

import emphases


###############################################################################
# Interface
###############################################################################


def from_audio(audio, sample_rate=emphases.SAMPLE_RATE, gpu=None):
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


###############################################################################
# Mel spectrogram
###############################################################################


class MelSpectrogram(torch.nn.Module):

    def __init__(self):
        super().__init__()
        window = torch.hann_window(emphases.NUM_FFT, dtype=torch.float)
        mel_basis = librosa.filters.mel(
            emphases.SAMPLE_RATE,
            emphases.NUM_FFT,
            emphases.NUM_MELS
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
