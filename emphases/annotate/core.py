import json
import random
import shutil

import pypar
import reseval

import emphases


###############################################################################
# Annotate emphases
###############################################################################


def datasets(
    stem_file,
    annotation_config=emphases.DEFAULT_ANNOTATION_CONFIG,
    dataset='libritts',
    directory=emphases.ANNOTATION_DIR,
    remote=False,
    production=False,
    interval=120):
    """Perform emphasis annotation on datasets"""
    # Create input and output directories
    directory.mkdir(exist_ok=True, parents=True)
    index = f'{len(list(directory.glob("*"))):02}'
    input_directory = directory / index
    input_directory.mkdir(exist_ok=True, parents=True)
    output_directory = emphases.DATA_DIR / 'annotate' / index
    output_directory.mkdir(exist_ok=True, parents=True)

    # Get stems
    with open(stem_file) as file:
        monitor = json.load(file)

        # Remove completed
        stems = [key for key, value in monitor.items() if value[0] < value[1]]

    # Get audio files
    cache_directory = emphases.CACHE_DIR / dataset
    audio_files = sorted(
        cache_directory / 'audio' / f'{stem}.wav' for stem in stems)

    # Deterministic shuffle
    random.seed(emphases.RANDOM_SEED)
    random.shuffle(audio_files)

    # Iterate over audio files
    for audio_file in audio_files:

        # Save audio
        shutil.copyfile(audio_file, input_directory / audio_file.name)

        # Load alignment
        alignment = pypar.Alignment(
            cache_directory /
            'alignment' /
            f'{audio_file.stem}.TextGrid')

        # Save text
        text_file = input_directory / f'{audio_file.stem}-words.txt'
        with open(text_file, 'w') as file:
            file.write(
                ' '.join([
                    str(word) for word in alignment
                    if str(word) != pypar.SILENCE]))

    # Run annotation
    reseval.run(
        annotation_config,
        input_directory,
        output_directory,
        not remote,
        production,
        interval)
