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
