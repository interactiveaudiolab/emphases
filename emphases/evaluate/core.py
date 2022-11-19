"""core.py - model evaluation"""


import emphases
from emphases.evaluate.metrics import eval_similarity
import os
import glob
import sys
import pandas as pd
import numpy as np

###############################################################################
# Evaluate
###############################################################################


def datasets(datasets, checkpoint=emphases.DEFAULT_CHECKPOINT, gpu=None):
    """Perform evaluation"""
    for dataset in datasets:
        # TODO - perform evaluation
        eval_dir = emphases.EVAL_DIR / dataset
        print('evaluating: ', dataset)
        if dataset.startswith('Buckeye'):
            ground_truth_file = "utils/BuckEye-annotations.csv"
            # buckeye(ground_truth_file, eval_dir)
            # turning it off for now
            pass

def buckeye(ground_truth_file, eval_dir):
    for prom_file in os.listdir(eval_dir):
        filename = prom_file.split('/')[-1].replace('.prom', '')
        prom_file = os.path.join(eval_dir, prom_file)
        prominence_cosine_similarity = eval_similarity(prom_file, ground_truth_file)
        print(f"cosine similarity for {filename}: {prominence_cosine_similarity} \n")
