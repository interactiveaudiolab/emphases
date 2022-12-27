import io
import requests
import shutil
import ssl
import tarfile
import urllib
import zipfile

import torchaudio
import tqdm

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
        if dataset == 'buckeye':
            buckeye(dataset)


def libritts():
    """Download libritts dataset"""
    # Download
    url = 'https://us.openslr.org/resources/60/train-clean-100.tar.gz'
    file = emphases.SOURCE_DIR / 'libritts-train-clean-100.tar.gz'
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


def buckeye(dataset):
    """Download buckeye dataset"""
    directory = emphases.DATA_DIR / dataset
    directory.mkdir(exist_ok=True, parents=True)

    # Download data for all speakers
    for speaker in tqdm.tqdm(BUCKEYE_SPEAKERS):
        speaker_directory = directory / speaker
        speaker_directory.mkdir(exist_ok=True, parents=True)

        # Download and extract all zip files for this speaker
        for url in BUCKEYE_SPEAKERS[speaker]:

            # Download
            response = requests.get(url)
            file = zipfile.ZipFile(io.BytesIO(response.content))

            # Extract
            file.extractall(speaker_directory)

    # TODO - format
    pass


###############################################################################
# Utilities
###############################################################################


def download_file(url, file):
    """Download file from url"""
    with urllib.request.urlopen(url, context=ssl.SSLContext()) as response, \
            open(file, 'wb') as output:
        shutil.copyfileobj(response, output)
