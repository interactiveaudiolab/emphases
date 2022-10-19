import shutil
import ssl
import tarfile
import urllib

import torch
import torchaudio
import tqdm

import emphases


###############################################################################
# Constants
###############################################################################


# Speakers selected by sorting the train-clean-100 speakers by longest total
# recording duration and manually selecting speakers with more natural,
# conversational (as opposed to read) prosody
LIBRITTS_SPEAKERS = [
    # Female
    40,
    669,
    4362,
    5022,
    8123,

    # Male
    196,
    460,
    1355,
    3664,
    7067  # uses characters
]


###############################################################################
# Download datasets
###############################################################################


def datasets(datasets):
    """Download datasets"""
    for dataset in datasets:
        if dataset == 'libritts':
            libritts()


def libritts():
    """Download libritts dataset"""
    # Download
    url = 'https://us.openslr.org/resources/60/train-clean-100.tar.gz'
    file = emphases.DATA_DIR / 'libritts-train-clean-100.tar.gz'
    # download_file(url, file)

    # Unzip
    # with tarfile.open(file, 'r:gz') as tfile:
    #     tfile.extractall(emphases.DATA_DIR)

    # Rename folder
    directory = emphases.DATA_DIR / 'libritts'
    # shutil.move(emphases.DATA_DIR / 'LibriTTS', directory)

    # Get list of audio files for each speaker
    audio_files = {
        speaker: sorted(
            (directory / 'train-clean-100' / str(speaker)).rglob('*.wav'))
        for speaker in LIBRITTS_SPEAKERS}

    # Write audio to cache
    output_directory = emphases.CACHE_DIR / 'libritts'
    output_directory.mkdir(exist_ok=True, parents=True)
    iterator = tqdm.tqdm(
        audio_files.items(),
        desc='Formatting libritts',
        dynamic_ncols=True,
        total=len(audio_files))
    for speaker, files in iterator:

        # Organize by speaker
        speaker_directory = output_directory / f'{speaker:06d}'
        speaker_directory.mkdir(exist_ok=True, parents=True)

        for i, audio_file in enumerate(files):

            # Convert to 22.05k wav
            audio = emphases.load.audio(audio_file)

            # If audio is too quiet, increase the volume
            maximum = torch.abs(audio).max()
            if maximum < .35:
                audio *= .35 / maximum

            # Save to disk
            stem = f'{i:06d}'
            torchaudio.save(
                speaker_directory / f'{stem}.wav',
                audio,
                emphases.SAMPLE_RATE)

            # Copy text file
            shutil.copy2(
                audio_file.with_suffix('.normalized.txt'),
                speaker_directory / f'{stem}.txt')


###############################################################################
# Utilities
###############################################################################


def download_file(url, file):
    """Download file from url"""
    with urllib.request.urlopen(url, context=ssl.SSLContext()) as response, \
            open(file, 'wb') as output:
        shutil.copyfileobj(response, output)
