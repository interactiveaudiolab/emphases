import emphases
import penn
import torch


###############################################################################
# Preprocess
###############################################################################


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
            pad=True,
            interp_unvoiced_at=emphases.PENN_VOICED_THRESHOLD,
            gpu=gpu)


def from_audio(audio, sample_rate=emphases.SAMPLE_RATE, gpu=None):
    """Preprocess one audio file"""
    # Move to device (no-op if devices are the same)
    audio = audio.to('cpu' if gpu is None else f'cuda:{gpu}')

    # Preprocess mels
    features = emphases.data.preprocess.mels.from_audio(audio, sample_rate)

    if emphases.PITCH_FEATURE or emphases.PERIODICITY_FEATURE:

        # Preprocess pitch and periodicity
        pitch, periodicity = penn.from_audio(
            audio,
            sample_rate,
            hopsize=emphases.convert.samples_to_seconds(emphases.HOPSIZE),
            fmin=emphases.FMIN,
            fmax=emphases.FMAX,
            pad=True,
            interp_unvoiced_at=emphases.PENN_VOICED_THRESHOLD,
            gpu=gpu)

        if emphases.PITCH_FEATURE:
            pitch = torch.log2(pitch)[None, :].to(audio.device)
            features = torch.cat((features, pitch), dim=1)

        if emphases.PERIODICITY_FEATURE:
            periodicity = periodicity[None, :].to(audio.device)
            features = torch.cat((features, periodicity), dim=1)

    if emphases.LOUDNESS_FEATURE:
        loudness = emphases.data.preprocess.loudness.from_audio(audio, sample_rate)
        loudness = loudness[None, :].to(audio.device)
        features = torch.cat((features, loudness), dim=1)

    return features
