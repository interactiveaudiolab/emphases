import csv
import shutil
import ssl
import tarfile
import urllib

import pypar
import torch
import torchaudio

import emphases


###############################################################################
# Constants
###############################################################################


# List of tokens to filter from Buckeye annotations
BUCKEYE_FILTER_LIST = [
    '{B_TRANS}',
    '{E_TRANS}',
    '<SIL>',
    '<VOCNOISE>'
]

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
    7067  # uses character voices
]


###############################################################################
# Download datasets
###############################################################################


def datasets(datasets):
    """Download datasets"""
    for dataset in datasets:
        if dataset == 'libritts':
            libritts()
        elif dataset == 'buckeye':
            buckeye()
        else:
            raise ValueError(f'Dataset {dataset} is not defined')


def libritts():
    """Download libritts dataset"""
    # Setup source directory
    source_directory = emphases.SOURCE_DIR / 'libritts'
    source_directory.mkdir(exist_ok=True, parents=True)

    # Download
    url = 'https://us.openslr.org/resources/60/train-clean-100.tar.gz'
    file = source_directory / 'libritts-train-clean-100.tar.gz'
    download_file(url, file)

    # Unzip
    with tarfile.open(file, 'r:gz') as tfile:
        tfile.extractall(emphases.DATA_DIR)

    # Rename folder
    directory = emphases.DATA_DIR / 'libritts'
    shutil.move(emphases.DATA_DIR / 'LibriTTS', directory)

    # Get list of audio files for each speaker
    audio_files = {
        speaker: sorted(
            (directory / 'train-clean-100' / str(speaker)).rglob('*.wav'))
        for speaker in LIBRITTS_SPEAKERS}

    # Write audio to cache
    output_directory = emphases.CACHE_DIR / 'libritts'
    output_directory.mkdir(exist_ok=True, parents=True)
    iterator = emphases.iterator(
        audio_files.items(),
        'Formatting libritts',
        total=len(audio_files))
    for speaker, files in iterator:

        # Organize by speaker
        speaker_directory = output_directory / f'{speaker:06d}'
        speaker_directory.mkdir(exist_ok=True, parents=True)

        for i, audio_file in enumerate(files):

            # Load and resample
            audio = emphases.load.audio(audio_file)

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


def buckeye():
    """Download buckeye dataset"""
    # Extract tar file to data directory
    file = emphases.SOURCE_DIR / 'buckeye' / 'buckeye.tar.gz'
    with tarfile.open(file, 'r:gz') as tfile:
        tfile.extractall(emphases.DATA_DIR)

    # Setup cache directory
    cache_directory = emphases.CACHE_DIR / 'buckeye'
    cache_directory.mkdir(exist_ok=True, parents=True)

    # Create subdirectories
    features = ['alignment', 'audio', 'scores']
    for feature in features:
        (cache_directory / feature).mkdir(exist_ok=True, parents=True)

    # Copy alignments and filter out unused tokens
    data_directory = emphases.DATA_DIR / 'buckeye'
    alignment_files = (data_directory / 'alignment').glob('*.TextGrid')
    for file in alignment_files:

        # Load alignment
        alignment = pypar.Alignment(file)

        # Filter
        for word in alignment:
            if str(word) in BUCKEYE_FILTER_LIST:
                word.word = pypar.SILENCE
                for phoneme in word:
                    phoneme.phoneme = pypar.SILENCE

        # Save alignment
        alignment.save(cache_directory / 'alignment' / f'{file.stem}.TextGrid')

    # Get audio files
    audio_files = sorted((data_directory / 'audio').glob('*.wav'))

    # Resample audio
    for audio_file in audio_files:

        # Load and resample
        audio = emphases.load.audio(audio_file)

        # Save to disk
        torchaudio.save(
            cache_directory / 'audio' / audio_file.name,
            audio,
            emphases.SAMPLE_RATE)

    # Read buckeye annotations
    data_directory = emphases.DATA_DIR / 'buckeye'
    with open(data_directory  / 'annotations.csv') as file:
        reader = csv.DictReader(file)
        annotations = [row for row in reader]

    # Extract per-word emphasis scores
    alignment_files = (cache_directory / 'alignment').glob('*.TextGrid')
    for file in alignment_files:

        # Load alignment
        alignment = pypar.Alignment(file)

        # Get words from annotation
        words = [word for word in annotations if word['filename'] == file.stem]

        # Get per-word emphasis scores
        j = 0
        scores = torch.zeros(len(alignment))
        for i, word in enumerate(alignment):

            # Keep silences as zero
            if str(word) == pypar.SILENCE:
                continue

            # Make sure alignments are aligned
            assert str(word).lower() == words[j]['word'].lower()

            # Update scores
            # pa.32 is the average of 32 human judgments of the perception of
            # prominence based on acoustic features
            scores[i] == words[j]['pa.32']

            j += 1

        # Save scores
        torch.save(scores, file.parent / 'scores' / f'{file.stem}.pt')


###############################################################################
# Utilities
###############################################################################


def download_file(url, file):
    """Download file from url"""
    with urllib.request.urlopen(url, context=ssl.SSLContext()) as response, \
            open(file, 'wb') as output:
        shutil.copyfileobj(response, output)
