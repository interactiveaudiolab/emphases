"""core.py - data preprocessing"""

import os
import glob
import emphases
from emphases.build_textgrid_buckeye import build_textgrid
from tqdm import tqdm
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

        dirc = glob.glob(os.path.join(input_directory, '*/'))
        for subdir in dirc:
                words = glob.glob(os.path.join(subdir, '*.words'))
                for word in words:
                    basename = word.split('/')[-1].replace('.words', '')
                    word_file = os.path.join(subdir, basename+'.words')
                    phones_file = os.path.join(subdir, basename+'.phones')
                    build_textgrid(word_file, phones_file, subdir)
                    
        # raise NotImplementedError
