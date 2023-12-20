import penn
import torch
import torchutil

import emphases


###############################################################################
# Preprocess
###############################################################################


@torchutil.notify('preprocess')
def datasets(datasets, gpu=None):
    """Preprocess datasets"""
    for dataset in datasets:
        cache_directory = emphases.CACHE_DIR / dataset

        # Get audio files, from cache
        audio_files = sorted(cache_directory.rglob('*.wav'))

        # Preprocess mels
        mel_files = [
            cache_directory / 'mels' / f'{file.stem}.pt'
            for file in audio_files]
        emphases.data.preprocess.mels.from_files_to_files(
            audio_files,
            mel_files)

        # Preprocess loudness
        loudness_files = [
            cache_directory / 'loudness' / f'{file.stem}.pt'
            for file in audio_files]
        emphases.data.preprocess.loudness.from_files_to_files(
            audio_files,
            loudness_files)

        # Preprocess pitch, periodicity
        (cache_directory / 'pitch').mkdir(exist_ok=True, parents=True)
        pitch_files = [
            cache_directory / 'pitch' / f'{file.stem}'
            for file in audio_files]
        penn.from_files_to_files(
            audio_files,
            pitch_files,
            hopsize=emphases.convert.samples_to_seconds(emphases.HOPSIZE),
            fmin=emphases.FMIN,
            fmax=emphases.FMAX,
            batch_size=2048,
            center='half-hop',
            interp_unvoiced_at=emphases.VOICED_THRESHOLD,
            num_workers=emphases.NUM_WORKERS,
            gpu=gpu)

        # Pitch and periodicity use floating-point hopsize, while mels and
        # loudness use an integer hopsize in samples. This results in
        # single-frame differences when the audio length is within one sample
        # of a new frame due to floating-point error. We simply remove the last
        # frame in this rare case.
        for loudness_file, pitch_file in zip(loudness_files, pitch_files):
            loudness = torch.load(loudness_file)
            pitch = torch.load(f'{pitch_file}-pitch.pt')
            periodicity = torch.load(f'{pitch_file}-periodicity.pt')
            if pitch.shape[1] == loudness.shape[1] + 1:
                pitch = pitch[:, :-1]
                periodicity = periodicity[:, :-1]
            torch.save(pitch, f'{pitch_file}-pitch.pt')
            torch.save(periodicity, f'{pitch_file}-periodicity.pt')


def from_audio(audio, gpu=None):
    """Preprocess one audio file"""
    # Move to device (no-op if devices are the same)
    audio = audio.to('cpu' if gpu is None else f'cuda:{gpu}')

    features = []

    # Preprocess mels
    if emphases.MEL_FEATURE:
        features.append(emphases.data.preprocess.mels.from_audio(audio))

    # Preprocess pitch and periodicity
    if emphases.PITCH_FEATURE or emphases.PERIODICITY_FEATURE:
        pitch, periodicity = penn.from_audio(
            audio,
            emphases.SAMPLE_RATE,
            hopsize=emphases.convert.samples_to_seconds(emphases.HOPSIZE),
            fmin=emphases.FMIN,
            fmax=emphases.FMAX,
            pad=True,
            interp_unvoiced_at=emphases.VOICED_THRESHOLD,
            gpu=gpu)

        if emphases.PITCH_FEATURE:
            if emphases.NORMALIZE:
                features.append(
                    (torch.log2(pitch) - emphases.LOGFMIN) /
                    (emphases.LOGFMAX - emphases.LOGFMIN))
            else:
                features.append(torch.log2(pitch))

        if emphases.PERIODICITY_FEATURE:
            features.append(periodicity)

        # Pitch and periodicity use floating-point hopsize, while mels and
        # loudness use an integer hopsize in samples. This results in
        # single-frame differences when the audio length is within one sample
        # of a new frame due to floating-point error. We simply repeat the last
        # frame in this rare case.
        frames = emphases.convert.samples_to_frames(audio.shape[-1])
        if pitch.shape[1] == frames + 1:
            pitch = pitch[:, :-1]
            periodicity = periodicity[:, :-1]

    # Preprocess loudness
    if emphases.LOUDNESS_FEATURE:
        loudness = emphases.data.preprocess.loudness.from_audio(
            audio,
            emphases.SAMPLE_RATE)
        features.append(loudness.to(audio.device))

    # Concatenate features
    features = features[0] if len(features) == 1 else torch.cat(features)

    return features[None]
