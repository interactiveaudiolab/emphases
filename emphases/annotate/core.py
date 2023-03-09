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
    interval=120):
    """Perform emphasis annotation on datasets"""
    # Create input and output directories
    input_directory = directory / 'input'
    input_directory.mkdir(exist_ok=True, parents=True)
    output_directory = emphases.DATA_DIR / 'annotate'
    output_directory.mkdir(exist_ok=True, parents=True)

    # Populate input directory with speech and text files
    for dataset in datasets:

        # Iterate over audio files
        for audio_file in (emphases.CACHE_DIR / dataset / 'audio').glob('*'):

            # Save audio
            output_audio_file = \
                input_directory / f'{dataset}-{audio_file.name}'
            shutil.copyfile(audio_file, output_audio_file)

            # Load alignment
            alignment = pypar.Alignment(
                audio_file.parent.parent /
                'alignment' /
                f'{audio_file.stem}.TextGrid')

            # Save text
            output_text_file = (
                output_audio_file.parent /
                f'{output_audio_file.stem}-words.txt')
            with open(output_text_file, 'w') as file:
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
