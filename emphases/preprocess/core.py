"""core.py - data preprocessing"""

import os
import glob
import sys
import emphases
from emphases.build_textgrid_buckeye import build_textgrid
import textgrids
from tqdm import tqdm
import shutil
###############################################################################
# Preprocess
###############################################################################


def datasets(datasets):
    """Preprocess a dataset

    Arguments
        name - string
            The name of the dataset to preprocess
    """
    for dataset in datasets:
        input_directory = emphases.DATA_DIR / dataset
        output_directory = emphases.CACHE_DIR / dataset

        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)

        WAVS_DIR = os.path.join(output_directory, 'wavs')
        WORDS_DIR = os.path.join(output_directory, 'words')
        PHONES_DIR = os.path.join(output_directory, 'phones')
        ANNOTATION_DIR = os.path.join(output_directory, 'annotation')
        TEXT_DIR = os.path.join(output_directory, 'txt')
        LOGS_DIR = os.path.join(output_directory, 'logs')

        if not os.path.isdir(WAVS_DIR):
            os.mkdir(WAVS_DIR)

        if not os.path.isdir(WORDS_DIR):
            os.mkdir(WORDS_DIR)

        if not os.path.isdir(PHONES_DIR):
            os.mkdir(PHONES_DIR)
            
        if not os.path.isdir(ANNOTATION_DIR):
            os.mkdir(ANNOTATION_DIR)

        if not os.path.isdir(TEXT_DIR):
            os.mkdir(TEXT_DIR)

        if not os.path.isdir(LOGS_DIR):
            os.mkdir(LOGS_DIR)

        # generate the TextGrid files
        print('generating TextGrid annotation')
        dirc = glob.glob(os.path.join(input_directory, '*/'))
        if dirc:
            for subdir in dirc:
                    words = glob.glob(os.path.join(subdir, '*.words'))
                    for word in words:
                        basename = word.split('/')[-1].replace('.words', '')
                        word_file = os.path.join(subdir, basename+'.words')
                        phones_file = os.path.join(subdir, basename+'.phones')
                        textgrid_path = os.path.join(ANNOTATION_DIR, f"{basename}.TextGrid")

                        if os.path.exists(textgrid_path):
                            grid = textgrids.TextGrid(textgrid_path)
                            grid.xmin = grid.interval_tier_to_array('words')[0]['begin']
                            grid.xmax = grid.interval_tier_to_array('words')[-1]['end']
                            grid.write(textgrid_path)
                            shutil.copy(textgrid_path, os.path.join(ANNOTATION_DIR,f"{basename}.TextGrid"))
                        else:
                            build_textgrid(word_file, phones_file, ANNOTATION_DIR)
        else:
            wav_files = glob.glob(os.path.join(input_directory, '*.wav'))
            textgrid_files = glob.glob(os.path.join(input_directory, '*.TextGrid'))
            for textgrid_path in textgrid_files:
                if os.path.exists(textgrid_path):
                    basename = textgrid_path.split('/')[-1].replace('.TextGrid', '')
                    grid = textgrids.TextGrid(textgrid_path)
                    grid.xmin = grid.interval_tier_to_array('words')[0]['begin']
                    grid.xmax = grid.interval_tier_to_array('words')[-1]['end']
                    grid.write(textgrid_path)
                    shutil.copy(textgrid_path, os.path.join(ANNOTATION_DIR,f"{basename}.TextGrid"))

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

                        shutil.copy(wav_file, os.path.join(WAVS_DIR, basename+'.wav'))
                        shutil.copy(word_file, os.path.join(WORDS_DIR, basename+'.words'))
                        shutil.copy(phones_file, os.path.join(PHONES_DIR, basename+'.phones'))
                        shutil.copy(text_file, os.path.join(TEXT_DIR, basename+'.txt'))
                        shutil.copy(log_file, os.path.join(LOGS_DIR, basename+'.log'))
        else:
            wav_files = glob.glob(os.path.join(input_directory, '*.wav'))
            textgrid_files = glob.glob(os.path.join(input_directory, '*.TextGrid'))
            for wav in wav_files:
                basename = wav.split('/')[-1].replace('.wav', '')
                shutil.copy(wav, os.path.join(WAVS_DIR, basename+'.wav'))

