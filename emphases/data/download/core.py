import io
import requests
import shutil
import ssl
import tarfile
import urllib
import zipfile

import pandas as pd
import pypar
import torchaudio

import emphases


###############################################################################
# Constants
###############################################################################


# BuckEye corpus speakers under consideration
# https://buckeyecorpus.osu.edu/php/speech.php?PHPSESSID=njvutg9l90fc3ebg7v30vnhmk1
BUCKEYE_SPEAKERS = {
    's02-1': [
        "https://buckeyecorpus.osu.edu/speechfiles/s02/s0201a.zip",
        "https://buckeyecorpus.osu.edu/speechfiles/s02/s0201b.zip"],
    's03-1': [
        "https://buckeyecorpus.osu.edu/speechfiles/s03/s0301a.zip",
        "https://buckeyecorpus.osu.edu/speechfiles/s03/s0301b.zip"],
    's04-1': [
        "https://buckeyecorpus.osu.edu/speechfiles/s04/s0401a.zip",
        "https://buckeyecorpus.osu.edu/speechfiles/s04/s0401b.zip"],
    's10-1': [
        "https://buckeyecorpus.osu.edu/speechfiles/s10/s1001a.zip",
        "https://buckeyecorpus.osu.edu/speechfiles/s10/s1001b.zip"],
    's11-1': [
        "https://buckeyecorpus.osu.edu/speechfiles/s11/s1101a.zip",
        "https://buckeyecorpus.osu.edu/speechfiles/s11/s1101b.zip"],
    's14-1': [
        "https://buckeyecorpus.osu.edu/speechfiles/s14/s1401a.zip",
        "https://buckeyecorpus.osu.edu/speechfiles/s14/s1401b.zip"],
    's16-1': [
        "https://buckeyecorpus.osu.edu/speechfiles/s16/s1601a.zip",
        "https://buckeyecorpus.osu.edu/speechfiles/s16/s1601b.zip"],
    's17-1': [
        "https://buckeyecorpus.osu.edu/speechfiles/s17/s1701a.zip",
        "https://buckeyecorpus.osu.edu/speechfiles/s17/s1701b.zip"],
    's21-1': [
        "https://buckeyecorpus.osu.edu/speechfiles/s21/s2101a.zip",
        "https://buckeyecorpus.osu.edu/speechfiles/s21/s2101b.zip"],
    's22-1': [
        "https://buckeyecorpus.osu.edu/speechfiles/s22/s2201a.zip",
        "https://buckeyecorpus.osu.edu/speechfiles/s22/s2201b.zip"],
    's24-1': [
        "https://buckeyecorpus.osu.edu/speechfiles/s24/s2401a.zip",
        "https://buckeyecorpus.osu.edu/speechfiles/s24/s2401b.zip"],
    's25-1': [
        "https://buckeyecorpus.osu.edu/speechfiles/s25/s2501a.zip",
        "https://buckeyecorpus.osu.edu/speechfiles/s25/s2501b.zip"],
    's26-1': [
        "https://buckeyecorpus.osu.edu/speechfiles/s26/s2601a.zip",
        "https://buckeyecorpus.osu.edu/speechfiles/s26/s2601b.zip"],
    's32-1': [
        "https://buckeyecorpus.osu.edu/speechfiles/s32/s3201a.zip",
        "https://buckeyecorpus.osu.edu/speechfiles/s32/s3201b.zip"],
    's33-1': [
        "https://buckeyecorpus.osu.edu/speechfiles/s33/s3301a.zip",
        "https://buckeyecorpus.osu.edu/speechfiles/s33/s3301b.zip"],
    's35-1': [
        "https://buckeyecorpus.osu.edu/speechfiles/s35/s3501a.zip",
        "https://buckeyecorpus.osu.edu/speechfiles/s35/s3501b.zip"]
}

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
    # Setup data directory
    data_directory = emphases.DATA_DIR / 'buckeye'
    data_directory.mkdir(exist_ok=True, parents=True)

    # Download data for all speakers
    for speaker in BUCKEYE_SPEAKERS:
        speaker_directory = data_directory / speaker
        speaker_directory.mkdir(exist_ok=True, parents=True)

        # Download and extract all zip files for this speaker
        for url in BUCKEYE_SPEAKERS[speaker]:

            # Download
            response = requests.get(url)
            file = zipfile.ZipFile(io.BytesIO(response.content))

            # Extract
            file.extractall(speaker_directory)

    # Setup cache directory
    cache_directory = emphases.CACHE_DIR / 'buckeye'
    cache_directory.mkdir(exist_ok=True, parents=True)

    # Read buckeye annotations
    annotation_file = emphases.SOURCE_DIR / 'buckeye' / 'annotations.csv'
    annotations = pd.read_csv(annotation_file)

    # Create subdirectories
    features = ['alignment', 'audio', 'scores', 'text']
    for feature in features:
        (cache_directory / feature).mkdir(exist_ok=True, parents=True)

    # Get phoneme alignment files
    phoneme_files = sorted(data_directory.rglob('*.phones'))

    # Get word alignment files
    word_files = sorted(data_directory.rglob('*.words'))

    # Build alignments using pypar
    for phoneme_file, word_file in zip(phoneme_files, word_files):

        # Make alignment
        alignment = make_buckeye_alignment(phoneme_file, word_file)

        # Save
        alignment.save(
            cache_directory / 'alignment' / f'{phoneme_file.stem}.TextGrid')

    # Get audio files
    audio_files = sorted(data_directory.rglob('*.wav'))

    # Resample audio
    for audio_file in audio_files:

        # Load and resample
        audio = emphases.load.audio(audio_file)

        # Save to disk
        torchaudio.save(
            cache_directory / 'audio' / audio_file.name,
            audio,
            emphases.SAMPLE_RATE)

    # TODO - extract per-word emphasis scores and text

    # # REFACTOR
    # # generate the TextGrid files and prominence ground truth files
    # speakers = data_directory.glob('*')
    # if dirc:
    #     for subdir in dirc:
    #         words = glob.glob(os.path.join(subdir, '*.words'))
    #         for word in words:
    #             basename = word.split('/')[-1].replace('.words', '')
    #             word_file = os.path.join(subdir, basename+'.words')
    #             phones_file = os.path.join(subdir, basename+'.phones')
    #             textgrid_path = os.path.join(
    #                 cache_directory / 'alignment', f"{basename}.TextGrid")
    #             prominence_annotation_path = os.path.join(
    #                  cache_directory / 'annotation', f"{basename}.prom")

    #             speaker_df = annotations[annotations['filename'] == basename][[
    #                 'filename', 'wordmin', 'wordmax', 'word', 'pa.32']]
    #             speaker_df.sort_values(by='wordmin').to_csv(
    #                 prominence_annotation_path, index=False)

    #             if os.path.exists(textgrid_path):
    #                 json_save_path = os.path.join(
    #                     output_directory / 'alignment', f"{basename}.json")
    #                 save_corrected_textgrid(
    #                     annotations, basename, textgrid_path, json_save_path)
    #             else:
    #                 emphases.build_textgrid_buckeye.build_textgrid(
    #                     word_file, phones_file, output_directory / 'alignment')
    # else:
    #     textgrid_files = glob.glob(os.path.join(input_directory, '*.TextGrid'))
    #     for textgrid_path in textgrid_files:
    #         if os.path.exists(textgrid_path):
    #             basename = textgrid_path.split(
    #                 '/')[-1].replace('.TextGrid', '')
    #             prominence_annotation_path = os.path.join(
    #                 output_directory / 'annotation', f"{basename}.prom")

    #             speaker_df = annotations[annotations['filename'] == basename][[
    #                 'filename', 'wordmin', 'wordmax', 'word', 'pa.32']]
    #             speaker_df.sort_values(by='wordmin').to_csv(prominence_annotation_path,
    #                                                         sep='\t',
    #                                                         encoding='utf-8',
    #                                                         index=False)

    #             json_save_path = os.path.join(
    #                 output_directory / 'alignment', f"{basename}.json")
    #             save_corrected_textgrid(
    #                 annotations, basename, textgrid_path, json_save_path)



###############################################################################
# Utilities
###############################################################################


def download_file(url, file):
    """Download file from url"""
    with urllib.request.urlopen(url, context=ssl.SSLContext()) as response, \
            open(file, 'wb') as output:
        shutil.copyfileobj(response, output)


# def save_corrected_textgrid(annotations, speaker_id, textgrid_path, json_save_path):
#     """
#     It will compare the prominence annotation tokens with textgrid file,
#     and save a new corrected .json textgrid file with only common tokens
#     """

#     speaker_df = annotations[annotations['filename'] == speaker_id][[
#         'filename', 'wordmin', 'wordmax', 'word', 'pa.32']]
#     speaker_df = speaker_df.sort_values(by='wordmin')
#     avail_start_times = speaker_df['wordmin'].apply(
#         lambda x: round(x, 5)).tolist()
#     avail_end_times = speaker_df['wordmax'].apply(
#         lambda x: round(x, 5)).tolist()

#     alignment = pypar.Alignment(textgrid_path)

#     json_align = alignment.json()
#     new_json = {'words': []}
#     for obj in json_align['words']:
#         if obj['start'] in avail_start_times or obj['end'] in avail_end_times:
#             new_json['words'].append(obj)

#     if len(new_json['words']) != len(speaker_df):
#         print(
#             f"WARNING: {speaker_id} Formulated alignment not matching with speaker annotation length dimensions")

#     new_alignment = pypar.Alignment(new_json)
#     new_alignment.save_json(json_save_path)


def make_buckeye_alignment(word_file, phoneme_file):
    """Create phoneme alignment object from buckeye alignment format"""
    # Load words
    with open(word_file) as file:
        words = file.read()
    words = [
        row.strip().split(';')[0].split() for row in words.split('\n')[9:-1]]

    # Load phonemes
    with open(phoneme_file) as file:
        phonemes = file.read()
    phonemes = [row.strip().split(';')[0].split()
                       for row in phonemes.split('\n')[9:-1]]

    # TODO - create pypar alignment
    alignment = None

    # grid_word_tuples = []
    # start = 0.0
    # for row in words[1:-1]:
    #     # getting rid of the redundant tokens (<IVER>, <VOCNOISE>, etc.)
    #     if not row[-1].startswith('<') and not row[-1].startswith('{'):
    #         end = row[0]
    #         if start == end:
    #             continue
    #         tup = (float(start), float(end), row[-1])
    #         grid_word_tuples.append(tup)
    #         start = row[0]

    # grid_phones_tuples = []
    # start = 0.0
    # for row in phonemes[1:-1]:
    #     # assuming all phoneme values are always lowercased,
    #     # getting rid of the redundant tokens (IVER, VOCNOISE, etc.)
    #     if not row[-1].isupper():
    #         end = row[0]
    #         if start == end:
    #             continue
    #         tup = (float(start), float(end), row[-1])
    #         grid_phones_tuples.append(tup)
    #         start = row[0]

    # # Build the grids
    # tg = textgrid.Textgrid()

    # if grid_word_tuples:
    #     end_time = grid_word_tuples[-1][1]
    # else:
    #     end_time = 1

    # wordTier = textgrid.IntervalTier('words', grid_word_tuples, 0, end_time)
    # phoneTier = textgrid.IntervalTier(
    #     'phones', grid_phones_tuples, 0, end_time)

    # tg.addTier(wordTier)
    # tg.addTier(phoneTier)

    return alignment
