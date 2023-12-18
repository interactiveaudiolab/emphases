import csv
import json
import shutil
import ssl
import tarfile
import urllib
import yaml

import pyfoal
import pypar
import torch
import torchutil
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
    # Top 5 Female (primarily by length)
    40,
    669,
    4362,
    5022,
    8123,

    # Additional female speakers to get to 1/8th of train-clean-100
    5022,
    696,
    6272,
    5163,

    # Top 5 Male (primarily by length)
    196,
    460,
    1355,
    3664,
    7067,

    # Additional male speakers to get to 1/8th of train-clean-100
    405,
    6437,
    446,
    4397
]


###############################################################################
# Download datasets
###############################################################################


@torchutil.notify('download')
def datasets(datasets, gpu=None):
    """Download datasets"""
    for dataset in datasets:
        if dataset == 'automatic':
            automatic(gpu=gpu)
        elif dataset == 'buckeye':
            buckeye()
        elif dataset == 'crowdsource':
            crowdsource()
        elif dataset == 'libritts':
            libritts()
        else:
            raise ValueError(f'Dataset {dataset} is not defined')


###############################################################################
# Individual dataset downloaders
###############################################################################


def automatic(gpu=None):
    """Create dataset from trained model"""
    # Setup directories
    cache_directory = emphases.CACHE_DIR / 'automatic'
    cache_directory.mkdir(exist_ok=True, parents=True)

    # Create subdirectories
    features = ['alignment', 'audio', 'scores']
    for feature in features:
        (cache_directory / feature).mkdir(exist_ok=True, parents=True)

    # Get files
    audio_files = list(
        (emphases.CACHE_DIR / 'libritts' / 'audio').rglob('*.wav'))
    stems = [file.stem for file in audio_files]

    # Copy from LibriTTS cache to annotation cache
    for stem in stems:

        # Copy audio
        audio_file = (
            emphases.CACHE_DIR / 'automatic' / 'audio' / f'{stem}.wav')
        shutil.copyfile(
            emphases.CACHE_DIR / 'libritts' / 'audio' / f'{stem}.wav',
            audio_file)

        # Copy alignment
        shutil.copyfile(
            emphases.CACHE_DIR / 'libritts' / 'alignment' / f'{stem}.TextGrid',
            emphases.CACHE_DIR / 'automatic' / 'alignment' / f'{stem}.TextGrid')

        # Load alignment
        alignment = pypar.Alignment(
            emphases.CACHE_DIR / 'automatic' / 'alignment' / f'{stem}.TextGrid')

        # Load audio
        audio, _ = torchaudio.load(audio_file)

        # Infer scores
        scores = emphases.from_alignment_and_audio(
            alignment,
            audio,
            emphases.SAMPLE_RATE,
            gpu=gpu).detach().cpu()

        # Save scores
        torch.save(scores, cache_directory / 'scores' / f'{stem}.pt')


def crowdsource():
    """Prepare crowdsourced dataset"""
    # Get annotation config
    with open(emphases.DEFAULT_ANNOTATION_CONFIG, "r") as stream:
        annotation_config = yaml.safe_load(stream)

    # Setup directories
    data_directory = emphases.DATA_DIR / 'crowdsource'
    cache_directory = emphases.CACHE_DIR / 'crowdsource'
    cache_directory.mkdir(exist_ok=True, parents=True)

    # Create subdirectories
    features = ['alignment', 'audio', 'scores']
    for feature in features:
        (cache_directory / feature).mkdir(exist_ok=True, parents=True)

    # Load annotations data
    annotation_data = {}
    for directory in data_directory.glob('*'):

        source_directory =  directory / annotation_config['name']
        table_directory = source_directory / 'tables'

        # Participant data
        participants = {}
        with open(table_directory / 'participants.csv') as file:
            for row in csv.DictReader(file):
                try:

                    # Crowdsourced annotation
                    participants[row['ID']] = {
                        'language': row['Language'],
                        'country': row['Country'],
                        'annotations': []}

                except KeyError as error:

                    # Manual annotation
                    participants[row['ID']] = {
                        'language': 'English',
                        'country': 'United States',
                        'annotations': []}

        # Response data
        with open(table_directory / 'responses.csv') as file:
            for row in csv.DictReader(file):
                participant = row['Participant']

                # Add participant
                if participant not in annotation_data:
                    annotation_data[participant] = participants[participant]

                # Get word start and end times
                alignment = pypar.Alignment(
                    emphases.CACHE_DIR /
                    'libritts' /
                    'alignment' /
                    f'{row["Stem"]}.TextGrid')
                words = [
                    (str(word).lower(), word.start(), word.end())
                    for word in alignment
                    if str(word) != pypar.SILENCE]

                # Format annotation
                entry = {
                    'stem': row['Stem'],
                    'score': [float(c) for c in row['Response']],
                    'words': words}
                assert len(entry['words']) == len(entry['score'])

                # Add annotation
                annotation_data[participant]['annotations'].append(entry)

    # Get worker ID correspondence
    correspondence = {}
    for directory in data_directory.glob('*'):
        file = (
            directory /
            annotation_config['name'] /
            'crowdsource' /
            'crowdsource.json')
        with open(file) as file:
            contents = json.load(file)
            for content in contents:
                correspondence |= {content['ParticipantID']: content['WorkerId']}

    # Crowdsourced annotation
    if correspondence:

        # Filter out where incomplete or > 1/3 examples have > 2/3 words selected
        def valid(items):
            if not hasattr(valid, 'count'):
                valid.count = 0
            sums = [sum(item['score']) for item in items]
            counts = [len(item['score']) for item in items]
            invalids = [s > .67 * c for s, c in zip(sums, counts)]
            is_valid = sum(invalids) < .33 * len(invalids)
            valid.count += 1 - int(is_valid)
            return is_valid

        # Join participants with same worker ID
        joined = {}
        for participant, contents in annotation_data.items():

            # Filter out bad batches
            if (
                len(contents['annotations']) < 20 or
                len(contents['annotations']) % 10 > 0 or
                not valid(contents['annotations'])
            ):
                continue

            worker = correspondence[participant]
            if worker in joined:
                joined[worker]['annotations'].extend(contents['annotations'])
            else:
                joined[worker] = contents

    # Manual annotation
    else:
        joined = annotation_data

    # Anonymize
    anonymized = {}
    for i, contents in enumerate(joined.values()):
        anonymized[f'{i:06d}'] = contents

    # Save annotations in release format
    with open(cache_directory / 'annotations.json', 'w') as file:
        json.dump(anonymized, file, sort_keys=True, indent=True)

    # Merge binary annotations to floats
    annotations = merge_annotations(anonymized)

    # Save dictionary containing annotation counts
    with open(cache_directory / 'counts.json', 'w') as file:
        json.dump(annotations['stems'], file, sort_keys=True, indent=True)

    # Get annotated stems
    stems = [
        file.replace('libritts-', '')
        for file in annotations['stems'].keys()]

    # Copy from LibriTTS cache to annotation cache
    for i, stem in enumerate(stems):

        # Get normalized scores
        count = annotations['stems'][stem]
        labels = [score / count for score in annotations['scores'][stem]]

        # Copy audio
        shutil.copyfile(
            emphases.CACHE_DIR / 'libritts' / 'audio' / f'{stem}.wav',
            emphases.CACHE_DIR / 'crowdsource' / 'audio' / f'{stem}.wav')

        # Copy alignment
        shutil.copyfile(
            emphases.CACHE_DIR / 'libritts' / 'alignment' / f'{stem}.TextGrid',
            emphases.CACHE_DIR / 'crowdsource' / 'alignment' / f'{stem}.TextGrid')

        # Load alignment
        alignment = pypar.Alignment(
            emphases.CACHE_DIR / 'crowdsource' / 'alignment' / f'{stem}.TextGrid')

        # Match alignment and scores (silences get a score of zero)
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
        torch.save(scores, cache_directory / 'scores' / f'{stem}.pt')


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
    torchutil.download.file(url, file)

    # Unzip
    with tarfile.open(file, 'r:gz') as tfile:
        tfile.extractall(emphases.DATA_DIR)

    # Rename folder
    directory = emphases.DATA_DIR / 'libritts'
    shutil.rmtree(directory, ignore_errors=True)
    shutil.move(emphases.DATA_DIR / 'LibriTTS', directory)

    # Download annotations from zenodo
    url = 'https://zenodo.org/records/10402793/files/libritts-emphasis-annotations.json?download=1'
    file = source_directory / 'annotations.json'
    torchutil.download.file(url, file)

    # Load annotations
    with open(source_directory / 'annotations.json') as file:
        annotations = json.load(file)

    # Merge annotations to floats
    annotations = merge_annotations(annotations)

    # Get list of audio files
    audio_files = list(directory.rglob('*.wav'))
    audio_files = [
        file for file in audio_files if file.stem in annotations['stems']]

    # Setup cache directory
    cache_directory = emphases.CACHE_DIR / 'libritts'
    cache_directory.mkdir(exist_ok=True, parents=True)

    # Create subdirectories
    features = ['alignment', 'audio', 'scores']
    for feature in features:
        (cache_directory / feature).mkdir(exist_ok=True, parents=True)

    # Iterate over files
    for audio_file in torchutil.iterator(
        audio_files,
        'Formatting libritts',
        total=len(audio_files)
    ):

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

    for i, stem in enumerate([file.stem for file in audio_files]):

        # Load alignment
        alignment = pypar.Alignment(
            cache_directory / 'alignment' / f'{stem}.TextGrid')

        # Get ground truth
        count = annotations['stems'][stem]
        labels = [score / count for score in annotations['scores'][stem]]

        # Match alignment and scores (silences get a score of zero)
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
        torch.save(scores, cache_directory / 'scores' / f'{stem}.pt')


###############################################################################
# Utilities
###############################################################################


def download_file(url, file):
    """Download file from url"""
    with urllib.request.urlopen(url, context=ssl.SSLContext()) as response, \
            open(file, 'wb') as output:
        shutil.copyfileobj(response, output)


def merge_annotations(annotations):
    """Merge crowdsourced annotations"""
    merged = {'samples': 0, 'scores': {}, 'stems': {}}
    for _, responses in annotations.items():

        # Iterate over stems
        for response in responses['annotations']:
            stem = response['stem']
            score = [float(c) for c in list(response['score'])]

            # Merge stem annotations
            if stem in merged['stems']:

                # Maybe cap the number of allowed annotations
                if (
                    emphases.MAX_ANNOTATIONS is not None and
                    merged['stems'][stem] == emphases.MAX_ANNOTATIONS
                ):
                    continue

                # Update sums and counts
                for i in range(len(score)):
                    merged['scores'][stem][i] += score[i]
                merged['stems'][stem] += 1

            # Add new stem
            else:
                merged['scores'][stem] = score
                merged['stems'][stem] = 1

            # Update total number of samples
            merged['samples'] += 1

    # Maybe cap the minimum required annotations
    if emphases.MIN_ANNOTATIONS is not None:
        merged['stems'] = {
            stem: count for stem, count in merged['stems'].items()
            if count == emphases.MIN_ANNOTATIONS}
        merged['scores'] = {
            stem: scores for stem, scores in merged['scores'].items()
            if stem in merged['stems']}

    return merged
