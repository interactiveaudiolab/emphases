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
        input_directory = emphases.DATA_DIR / dataset
        output_directory = emphases.CACHE_DIR / dataset
        output_directory.mkdir(exist_ok=True, parents=True)
        annotation_file = emphases.DATA_DIR / dataset / 'annotations.csv'

        if dataset == 'buckeye':
            buckeye(input_directory, output_directory, annotation_file)
        elif dataset == 'libritts':
            libritts()
        else:
            raise ValueError(f'Dataset {dataset} is not defined')


###############################################################################
# Individual datasets
###############################################################################


def buckeye(input_directory, output_directory, annotation_file):
    annotations = pd.read_csv(annotation_file)

    WAVS_DIR = os.path.join(output_directory, 'wavs')
    MEL_DIR = os.path.join(output_directory, 'mels')
    WORDS_DIR = os.path.join(output_directory, 'words')
    PHONES_DIR = os.path.join(output_directory, 'phones')
    ALIGNMENT_DIR = os.path.join(output_directory, 'alignment')
    TEXT_DIR = os.path.join(output_directory, 'txt')
    LOGS_DIR = os.path.join(output_directory, 'logs')
    ANNOTATION_DIR = os.path.join(output_directory, 'annotation')

    if not os.path.isdir(WAVS_DIR):
        os.mkdir(WAVS_DIR)

    if not os.path.isdir(MEL_DIR):
        os.mkdir(MEL_DIR)

    if not os.path.isdir(WORDS_DIR):
        os.mkdir(WORDS_DIR)

    if not os.path.isdir(PHONES_DIR):
        os.mkdir(PHONES_DIR)

    if not os.path.isdir(ALIGNMENT_DIR):
        os.mkdir(ALIGNMENT_DIR)

    if not os.path.isdir(ANNOTATION_DIR):
        os.mkdir(ANNOTATION_DIR)

    if not os.path.isdir(TEXT_DIR):
        os.mkdir(TEXT_DIR)

    if not os.path.isdir(LOGS_DIR):
        os.mkdir(LOGS_DIR)

    # generate the TextGrid files and prominence ground truth files
    print('generating TextGrid alignment and Prominence ground truth files')
    dirc = glob.glob(os.path.join(input_directory, '*/'))
    if dirc:
        for subdir in dirc:
                words = glob.glob(os.path.join(subdir, '*.words'))
                for word in words:
                    basename = word.split('/')[-1].replace('.words', '')
                    word_file = os.path.join(subdir, basename+'.words')
                    phones_file = os.path.join(subdir, basename+'.phones')
                    textgrid_path = os.path.join(ALIGNMENT_DIR, f"{basename}.TextGrid")
                    prominence_annotation_path = os.path.join(ANNOTATION_DIR, f"{basename}.prom")

                    speaker_df = annotations[annotations['filename']==basename][['filename', 'wordmin', 'wordmax', 'word', 'pa.32']]
                    speaker_df.sort_values(by='wordmin').to_csv(prominence_annotation_path, index=False)

                    if os.path.exists(textgrid_path):
                        # grid = textgrids.TextGrid(textgrid_path)
                        # grid.xmin = grid.interval_tier_to_array('words')[0]['begin']
                        # grid.xmax = grid.interval_tier_to_array('words')[-1]['end']
                        # grid.write(textgrid_path)
                        # shutil.copy(textgrid_path, os.path.join(ALIGNMENT_DIR,f"{basename}.TextGrid"))
                        json_save_path = os.path.join(ALIGNMENT_DIR,f"{basename}.json")
                        save_corrected_textgrid(annotations, basename, textgrid_path, json_save_path)
                    else:
                        emphases.build_textgrid_buckeye.build_textgrid(
                            word_file, phones_file, ALIGNMENT_DIR)
    else:
        textgrid_files = glob.glob(os.path.join(input_directory, '*.TextGrid'))
        for textgrid_path in textgrid_files:
            if os.path.exists(textgrid_path):
                basename = textgrid_path.split('/')[-1].replace('.TextGrid', '')
                prominence_annotation_path = os.path.join(ANNOTATION_DIR, f"{basename}.prom")

                speaker_df = annotations[annotations['filename']==basename][['filename', 'wordmin', 'wordmax', 'word', 'pa.32']]
                speaker_df.sort_values(by='wordmin').to_csv(prominence_annotation_path,
                                                            sep='\t',
                                                            encoding='utf-8',
                                                            index=False)

                # grid = textgrids.TextGrid(textgrid_path)
                # grid.xmin = grid.interval_tier_to_array('words')[0]['begin']
                # grid.xmax = grid.interval_tier_to_array('words')[-1]['end']
                # grid.write(textgrid_path)
                # shutil.copy(textgrid_path, os.path.join(ALIGNMENT_DIR,f"{basename}.TextGrid"))

                json_save_path = os.path.join(ALIGNMENT_DIR,f"{basename}.json")
                save_corrected_textgrid(annotations, basename, textgrid_path, json_save_path)

    # save files in cache
    print('Populating the cache folder')
    dirc = glob.glob(os.path.join(input_directory, '*/'))
    if dirc:
        for subdir in dirc:
                words = glob.glob(os.path.join(subdir, '*.words'))
                for word in words:
                    basename = word.split('/')[-1].replace('.words', '')

                    wav_file = os.path.join(subdir, basename+'.wav')
                    word_file = os.path.join(subdir, basename+'.words')
                    phones_file = os.path.join(subdir, basename+'.phones')
                    text_file = os.path.join(subdir, basename+'.txt')
                    log_file = os.path.join(subdir, basename+'.log')

                    # save audio fles using torchaudio, to maintain standrad loading
                    audio = emphases.load.audio(wav_file)
                    torchaudio.save(os.path.join(WAVS_DIR, basename+'.wav'), audio, emphases.SAMPLE_RATE)

                    mel_spectrogram = emphases.preprocess.mels.from_audio(audio)
                    mel_spectrogram_numpy = mel_spectrogram.numpy()
                    np.save(os.path.join(MEL_DIR, basename+'.npy'), mel_spectrogram_numpy)

                    shutil.copy(word_file, os.path.join(WORDS_DIR, basename+'.words'))
                    shutil.copy(phones_file, os.path.join(PHONES_DIR, basename+'.phones'))
                    shutil.copy(text_file, os.path.join(TEXT_DIR, basename+'.txt'))
                    shutil.copy(log_file, os.path.join(LOGS_DIR, basename+'.log'))
    else:
        wav_files = glob.glob(os.path.join(input_directory, '*.wav'))
        for wav_file in wav_files:
            basename = wav_file.split('/')[-1].replace('.wav', '')

            # save audio fles using torchaudio, to maintain standrad loading
            audio = emphases.load.audio(wav_file)
            torchaudio.save(os.path.join(WAVS_DIR, basename+'.wav'), audio, emphases.SAMPLE_RATE)

            mel_spectrogram = emphases.preprocess.mels.from_audio(audio)
            mel_spectrogram_numpy = mel_spectrogram.numpy()
            np.save(os.path.join(MEL_DIR, basename+'.npy'), mel_spectrogram_numpy)


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
