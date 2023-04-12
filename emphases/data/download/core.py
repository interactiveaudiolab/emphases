import csv
import shutil
import ssl
import tarfile
import urllib
import yaml
import json

import pyfoal
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
    '<CUTOFF-i=?>',
    '<CUTOFF-ta=taking?>',
    '<CUTOFF-th=that>',
    '<IVER>',
    '<LAUGH>',
    '<SIL>',
    '<VOCNOISE>',
    '<and>',
    '<i>',
    '<out>',
    '<that>',
    '<think>',
    '<so>',
    '<some>',
    '<um>',
    '<xx>',
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
        elif dataset == 'annotate':
            annotate()
        else:
            raise ValueError(f'Dataset {dataset} is not defined')


###############################################################################
# Individual dataset downloaders
###############################################################################


def annotate():
    """Download annotated dataset"""
    # Extract annotations to data directory
    with open(emphases.DEFAULT_ANNOTATION_CONFIG, "r") as stream:
        try:
            annotation_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    source_directory = emphases.ANNOTATION_DIR
    data_directory = emphases.DATA_DIR / 'annotate' / '01' / annotation_config['name']
    response_file = data_directory / 'tables' / 'responses.csv'
    results_file = data_directory / 'results.json'
    shutil.copytree(source_directory / 'input', data_directory / 'input', dirs_exist_ok=True)

    # Setup cache directory
    cache_directory = emphases.CACHE_DIR / 'annotate'
    cache_directory.mkdir(exist_ok=True, parents=True)

    # Create subdirectories
    features = ['alignment', 'audio', 'scores']
    for feature in features:
        (cache_directory / feature).mkdir(exist_ok=True, parents=True)

    # Read annotation data
    annotation_data = json.loads(open(results_file, 'r').read())
    annotated_samples = [stem for stem in annotation_data['scores'].keys()]

    # Get annotated audio files
    audio_files = sorted((data_directory / 'input').glob('*.wav'))
    audio_files = [file for file in audio_files if file.stem.replace('libritts-', '') in annotated_samples]

    # Get output alignment files
    alignment_files = []

    # Iterate over files
    for i, audio_file in enumerate(audio_files):

        save_stem = audio_file.stem.replace('libritts-', '')

        # Load and resample audio
        audio = emphases.load.audio(audio_file)

        # If audio is too quiet, increase the volume
        maximum = torch.abs(audio).max()
        if maximum < .35:
            audio *= .35 / maximum

        # Save to disk
        torchaudio.save(
            cache_directory / 'audio' / f'{save_stem}.wav',
            audio,
            emphases.SAMPLE_RATE)

        # Save alignment file path
        alignment_files.append(
            cache_directory / 'alignment' / f'{save_stem}.TextGrid')

    # Get corresponding text files
    text_files = [
        file.parent / f"{file.stem}-words.txt" for file in audio_files]

    # Align text and audio
    pyfoal.from_files_to_files(
        text_files,
        audio_files,
        alignment_files,
        'p2fa')

    # Extract per-word emphasis scores
    for file in alignment_files:

        # Load alignment
        alignment = pypar.Alignment(file)

        # Get words from annotation
        labels = annotation_data['scores'][file.stem]

        # Get per-word emphasis scores
        j = 0
        scores = torch.zeros(len(alignment))
        for i, word in enumerate(alignment):

            # Keep silences as zero
            if str(word) == pypar.SILENCE:
                continue

            # Update scores
            scores[i] = float(labels[j])

            j += 1

        # Save scores
        torch.save(scores, cache_directory / 'scores' / f'{file.stem}.pt')


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
                word.phonemes = [
                    pypar.Phoneme(pypar.SILENCE, word.start(), word.end())]

        # Deduplicate silence tokens
        i = 0
        words = alignment.words()
        prev_silence = False
        while i < len(words):
            word = words[i]
            if str(word) == pypar.SILENCE:
                if prev_silence:
                    words[i - 1][-1]._end = word.end()
                    del words[i]
                else:
                    prev_silence = True
                    i += 1
            else:
                prev_silence = False
                i += 1

        # Save alignment
        pypar.Alignment(words).save(
            cache_directory / 'alignment' / f'{file.stem}.TextGrid')

    # Get audio files
    audio_files = sorted((data_directory / 'audio').glob('*.wav'))

    # Resample audio
    for audio_file in audio_files:

        # Load and resample
        audio = emphases.load.audio(audio_file)

        # If audio is too quiet, increase the volume
        maximum = torch.abs(audio).max()
        if maximum < .35:
            audio *= .35 / maximum

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
        words = sorted(words, key=lambda x: float(x['wordmin']))

        # Get per-word emphasis scores
        j = 0
        scores = torch.zeros(len(alignment))
        for i, word in enumerate(alignment):

            # Keep silences as zero
            if str(word) == pypar.SILENCE:
                continue

            # Make sure alignments are aligned
            assert str(word).lower() == words[j]['word'].lower()
            assert (word.start() - float(words[j]['wordmin'])) < 1e-4
            assert (word.end() - float(words[j]['wordmax'])) < 1e-4

            # Update scores
            # pa.32 is the average of 32 human judgments of the perception of
            # prominence based on acoustic features
            scores[i] = float(words[j]['pa.32'])

            j += 1

        # Save scores
        torch.save(scores, cache_directory / 'scores' / f'{file.stem}.pt')


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
    shutil.rmtree(directory, ignore_errors=True)
    shutil.move(emphases.DATA_DIR / 'LibriTTS', directory)

    # Get list of audio files
    audio_files = list(directory.rglob('*.wav'))

    # Setup cache directory
    cache_directory = emphases.CACHE_DIR / 'libritts'
    cache_directory.mkdir(exist_ok=True, parents=True)

    # Create subdirectories
    features = ['alignment', 'audio']
    for feature in features:
        (cache_directory / feature).mkdir(exist_ok=True, parents=True)

    # Iterate over files
    iterator = emphases.iterator(
        audio_files,
        'Formatting libritts',
        total=len(audio_files))
    for audio_file in iterator:

        # Load and resample audio
        audio = emphases.load.audio(audio_file)

        # If audio is too quiet, increase the volume
        maximum = torch.abs(audio).max()
        if maximum < .35:
            audio *= .35 / maximum

        # Save audio
        stem = audio_file.stem
        torchaudio.save(
            cache_directory / 'audio' / f'{stem}.wav',
            audio,
            emphases.SAMPLE_RATE)

    # Align text and audio
    text_files = [
        file.with_suffix('.normalized.txt') for file in audio_files]
    alignment_files = [
        cache_directory / 'alignment' / f'{file.stem}.TextGrid'
        for file in audio_files]
    pyfoal.from_files_to_files(
        text_files,
        audio_files,
        alignment_files,
        'p2fa')


###############################################################################
# Utilities
###############################################################################


def download_file(url, file):
    """Download file from url"""
    with urllib.request.urlopen(url, context=ssl.SSLContext()) as response, \
            open(file, 'wb') as output:
        shutil.copyfileobj(response, output)
