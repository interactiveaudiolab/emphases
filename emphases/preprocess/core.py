import emphases


###############################################################################
# Preprocess
###############################################################################


def datasets(datasets):
    """Preprocess datasets"""
    for dataset in datasets:
        data_directory = emphases.DATA_DIR / dataset
        cache_directory = emphases.CACHE_DIR / dataset

        # Get audio files
        audio_files = sorted(data_directory.rglob('*.wav'))

        # Get output filepaths for mels
        mel_files = [
             cache_directory / 'mels' / f'{file.stem}.pt'
             for file in audio_files]

        # Preprocess mels
        emphases.preprocess.mels.from_files_to_files(audio_files, mel_files)


def from_audio(audio, sample_rate=emphases.SAMPLE_RATE, gpu=None):
    """Preprocess one audio file"""
    # Move to device (no-op if devices are the same)
    audio = audio.to('cpu' if gpu is None else f'cuda:{gpu}')

    # Preprocess mels
    return emphases.preprocess.mels.from_audio(audio, sample_rate)
