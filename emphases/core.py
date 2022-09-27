import json

# import pyfoal
import soundfile
import tqdm

import emphases


def from_file(text_file, audio_file):
    """Determine locations of emphases for a speech audio file"""
    # Load text
    with open(text_file, encoding='utf-8') as file:
        text = file.read()

    # Load audio
    audio, sample_rate = soundfile.read(audio_file)

    # Detect emphases
    return from_text_and_audio(text, audio, sample_rate)


def from_file_to_file(text_file, audio_file, output_file=None):
    """Determine locations of emphases for a speech audio file"""
    if output_file is None:
        output_file = text_file.with_suffix('.json')

    # Detect emphases
    alignment, results = from_file(text_file, audio_file)

    # Format results
    results_list = [
        (str(word), word.start(), word.end(), result)
        for word, result in zip(alignment.words(), results)]

    # Save results
    with open(output_file, 'w') as file:
        json.dump(results_list, file, indent=4)


def from_files_to_files(text_files, audio_files, output_files=None):
    """Determine locations of emphases for many speech audio files"""
    # Set default output path
    if output_files is None:
        output_files = [file.with_suffix('.json') for file in text_files]

    # Detect emphases
    for files in tqdm.tqdm(zip(text_files, audio_files, output_files)):
        from_file_to_file(*files)


def from_text_and_audio(text, audio, sample_rate):
    """Determine locations of emphases"""
    # Compute combined features
    combined = emphases.features.combined(audio, sample_rate)

    # Continuous wavelet transform
    cwt = emphases.transform.wavelet(combined)

    # Line of maximum amplitude
    loma = emphases.transform.loma(cwt)

    # Get prominence
    prominence = emphases.features.prominence(loma)

    # Get alignment
    alignment = pyfoal.align(text, audio, sample_rate)

    # Detect emphases from prominence and alignment
    results = [False] * len(alignment.words())
    for word in alignment.words():

        # Start and end time in seconds
        start, end = word.start(), word.end()

        # TODO - Determine if the prominence between start and end is
        #        large enough to mark the word as emphasized.
        pass

    return alignment, results
