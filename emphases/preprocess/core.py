import glob
import os
import shutil

import numpy as np
import pandas as pd
import pypar
import torchaudio

import emphases


###############################################################################
# Preprocess
###############################################################################


def datasets(datasets):
    """Preprocess datasets"""
    for dataset in datasets:
        if dataset == 'buckeye':
            buckeye()
        elif dataset == 'libritts':
            libritts()
        else:
            raise ValueError(f'Dataset {dataset} is not defined')


###############################################################################
# Individual datasets
###############################################################################


def buckeye():
    """Preprocess buckeye dataset"""
    input_directory = emphases.DATA_DIR / 'buckeye'
    output_directory = emphases.CACHE_DIR / 'buckeye'
    output_directory.mkdir(exist_ok=True, parents=True)

    # Read buckeye annotations
    annotation_file = emphases.DATA_DIR / 'buckeye' / 'annotations.csv'
    annotations = pd.read_csv(annotation_file)

    # Create subdirectories
    features = ['alignment', 'mels', 'text']
    for feature in features:
        (output_directory / feature).mkdir(exist_ok=True, parents=True)

    # TODO - build alignments using pypar
    # TODO - preprocess mels
    # TODO - copy text files

    # generate the TextGrid files and prominence ground truth files
    print('generating TextGrid alignment and Prominence ground truth files')
    speakers = input_directory.glob('*')
    if dirc:
        for subdir in dirc:
                words = glob.glob(os.path.join(subdir, '*.words'))
                for word in words:
                    basename = word.split('/')[-1].replace('.words', '')
                    word_file = os.path.join(subdir, basename+'.words')
                    phones_file = os.path.join(subdir, basename+'.phones')
                    textgrid_path = os.path.join(output_directory / 'alignment', f"{basename}.TextGrid")
                    prominence_annotation_path = os.path.join(output_directory / 'annotation', f"{basename}.prom")

                    speaker_df = annotations[annotations['filename']==basename][['filename', 'wordmin', 'wordmax', 'word', 'pa.32']]
                    speaker_df.sort_values(by='wordmin').to_csv(prominence_annotation_path, index=False)

                    if os.path.exists(textgrid_path):
                        json_save_path = os.path.join(output_directory / 'alignment',f"{basename}.json")
                        save_corrected_textgrid(annotations, basename, textgrid_path, json_save_path)
                    else:
                        emphases.build_textgrid_buckeye.build_textgrid(
                            word_file, phones_file, output_directory / 'alignment')
    else:
        textgrid_files = glob.glob(os.path.join(input_directory, '*.TextGrid'))
        for textgrid_path in textgrid_files:
            if os.path.exists(textgrid_path):
                basename = textgrid_path.split('/')[-1].replace('.TextGrid', '')
                prominence_annotation_path = os.path.join(output_directory / 'annotation', f"{basename}.prom")

                speaker_df = annotations[annotations['filename']==basename][['filename', 'wordmin', 'wordmax', 'word', 'pa.32']]
                speaker_df.sort_values(by='wordmin').to_csv(prominence_annotation_path,
                                                            sep='\t',
                                                            encoding='utf-8',
                                                            index=False)

                json_save_path = os.path.join(output_directory / 'alignment',f"{basename}.json")
                save_corrected_textgrid(annotations, basename, textgrid_path, json_save_path)

    # save files in cache
    print('Populating the cache folder')
    dirc = glob.glob(os.path.join(input_directory, '*/'))
    if dirc:
        for subdir in dirc:
            words = glob.glob(os.path.join(subdir, '*.words'))
            for word in words:
                basename = word.split('/')[-1].replace('.words', '')

                audio_file = os.path.join(subdir, basename+'.wav')
                text_file = os.path.join(subdir, basename+'.txt')

                # save audio fles using torchaudio, to maintain standrad loading
                audio = emphases.load.audio(audio_file)
                torchaudio.save(os.path.join(output_directory / 'audio', basename+'.wav'), audio, emphases.SAMPLE_RATE)

                mel_spectrogram = emphases.preprocess.mels.from_audio(audio)
                mel_spectrogram_numpy = mel_spectrogram.numpy()
                np.save(os.path.join(output_directory / 'mels', basename+'.npy'), mel_spectrogram_numpy)

                shutil.copy(text_file, os.path.join(output_directory / 'text', basename+'.txt'))
    else:
        audio_files = glob.glob(os.path.join(input_directory, '*.wav'))
        for audio_file in audio_files:
            basename = audio_file.split('/')[-1].replace('.wav', '')

            # save audio fles using torchaudio, to maintain standrad loading
            audio = emphases.load.audio(audio_file)
            torchaudio.save(os.path.join(output_directory / 'audio', basename+'.wav'), audio, emphases.SAMPLE_RATE)

            mel_spectrogram = emphases.preprocess.mels.from_audio(audio)
            mel_spectrogram_numpy = mel_spectrogram.numpy()
            np.save(os.path.join(output_directory / 'mels', basename+'.npy'), mel_spectrogram_numpy)


def libritts():
    """Preprocess libritts dataset"""
    # TODO
    pass


###############################################################################
# Utilities
###############################################################################


def save_corrected_textgrid(annotations, speaker_id, textgrid_path, json_save_path):
    """
    It will compare the prominence annotation tokens with textgrid file,
    and save a new corrected .json textgrid file with only common tokens
    """

    speaker_df = annotations[annotations['filename']==speaker_id][['filename', 'wordmin', 'wordmax', 'word', 'pa.32']]
    speaker_df = speaker_df.sort_values(by='wordmin')
    avail_start_times = speaker_df['wordmin'].apply(lambda x: round(x, 5)).tolist()
    avail_end_times = speaker_df['wordmax'].apply(lambda x: round(x, 5)).tolist()

    alignment = pypar.Alignment(textgrid_path)

    json_align = alignment.json()
    new_json = {'words':[]}
    for obj in json_align['words']:
        if obj['start'] in avail_start_times or obj['end'] in avail_end_times:
            new_json['words'].append(obj)

    if len(new_json['words'])!=len(speaker_df):
        print(f"WARNING: {speaker_id} Formulated alignment not matching with speaker annotation length dimensions")

    new_alignment = pypar.Alignment(new_json)
    new_alignment.save_json(json_save_path)
