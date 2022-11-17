import json

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import numpy as np

import emphases


###############################################################################
# Loading utilities
###############################################################################


def audio(file):
    """Load audio and maybe resample"""
    # Load
    audio, sample_rate = torchaudio.load(file)

    # Maybe resample
    if sample_rate != emphases.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            sample_rate,
            emphases.SAMPLE_RATE)
        audio = resampler(audio)

    return audio

def partition(dataset):
    """Load partitions for dataset"""
    with open(emphases.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)

def load_prominence(file):
    with open(file, 'r') as f:
        data = f.read()
    # first line is header, skip it
    lines = [x.split('\t') for x in data.split('\n')[1:]]
    proms = torch.tensor([float(x[4]) for x in lines[:-1]])
    return proms

###############################################################################
# Mel spectrogram
###############################################################################

def torch_melspectrogram(audio):
    win_length = None
    # TODO - padding for spectro, check center
    # 
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=emphases.SAMPLE_RATE,
        n_fft=emphases.NUM_FFT,
        win_length=None,
        hop_length=emphases.HOPSIZE,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=emphases.NUM_MELS,
        mel_scale="htk",
    )

    return mel_spectrogram(audio)


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


# NOTE - tgt is GPL licensed. We cannot use GPL code in our final codebase if
#        we want to release as MIT-licensed.
# def read_textgrid(filename, sample_rate=200):
#     import tgt
#     try:
#         tg = tgt.read_textgrid(filename) #, include_empty_intervals=True)
#     except:
#         print("reading "+filename+" failed")

#         return
#     tiers = []
#     labs = {}

#     for tier in tg.get_tier_names():
#         if (tg.get_tier_by_name(tier)).tier_type()!='IntervalTier':
#             continue
#         tiers.append(tg.get_tier_by_name(tier))

#         lab = []
#         for a in tiers[-1].annotations:

#             try:
#                 # this was for some past experiment
#                 if a.text in ["p1","p2","p3","p4","p5","p6","p7"]:
#                     lab[-1][-1]=lab[-1][-1]+"_"+a.text
#                 else:
#                 #lab.append([a.start_time*sample_rate,a.end_time*sample_rate,a.text.encode('utf-8')])
#                     lab.append([a.start_time*sample_rate,a.end_time*sample_rate,a.text])
#             except:
#                 pass
#             #print tiers[-1].encode('latin-1')
#         labs[tier.lower()] = lab
#     try:
#         for i in range(len(labs['prosody'])):
#             if labs['prosody'][i][2][-2:] not in ["p1","p2","p3","p4","p5","p6","p7"]:
#                 labs['prosody'][i][2]+="_p0"
#     except:
#         pass

#     return labs
