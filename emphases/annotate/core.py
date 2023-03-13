import random
import shutil

import pypar
import reseval

import emphases


###############################################################################
# Annotate emphases
###############################################################################


def datasets(
    annotation_config=emphases.DEFAULT_ANNOTATION_CONFIG,
    datasets=['libritts'],
    directory=emphases.ANNOTATION_DIR,
    remote=False,
    production=False,
    interval=120,
    fraction=None):
    """Perform emphasis annotation on datasets"""
    # Create input and output directories
    directory.mkdir(exist_ok=True, parents=True)
    index = f'{len(directory.glob("*"))}:02'
    input_directory = directory / index
    output_directory = emphases.DATA_DIR / 'annotate' / index
    output_directory.mkdir(exist_ok=True, parents=True)

    # Populate input directory with speech and text files
    for dataset in datasets:

        # Get audio files
        audio_files = (emphases.CACHE_DIR / dataset / 'audio').glob('*')

        # Deterministic shuffle
        random.seed(emphases.RANDOM_SEED)
        random.shuffle(audio_files)

        # Maybe annotate only a fraction
        if fraction is not None:
            audio_files = audio_files[:int(len(audio_files) * fraction)]

        # Iterate over audio files
        for audio_file in audio_files:

            # Save audio
            shutil.copyfile(audio_file, input_directory / audio_file.name)

            # Load alignment
            alignment = pypar.Alignment(
                audio_file.parent.parent /
                'alignment' /
                f'{audio_file.stem}.TextGrid')

            # Save text
            text_file = input_directory.parent / f'{audio_file.stem}.txt'
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
