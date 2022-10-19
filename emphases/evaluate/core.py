"""core.py - model evaluation"""


import emphases
import os
import glob
import sys
import pandas as pd
import numpy as np
from scipy import spatial

###############################################################################
# Evaluate
###############################################################################


def datasets(datasets, checkpoint=emphases.DEFAULT_CHECKPOINT, gpu=None):
    """Perform evaluation"""
    for dataset in datasets:
        # TODO - perform evaluation
        eval_dir = emphases.EVAL_DIR / dataset
        print(dataset, 'perform eval \n')
        if dataset.startswith('Buckeye'):
            ground_truth_file = "utils/BuckEye-annotations.csv"
            for prom_file in os.listdir(eval_dir):
                filename = prom_file.split('/')[-1].replace('.prom', '')
                prom_file = os.path.join(eval_dir, prom_file)
                prominence_cosine_similarity = eval_similarity(prom_file, ground_truth_file)
                print(f"cosine similarity for {filename}: {prominence_cosine_similarity} \n")

def eval_similarity(prom_file, ground_truth_file):
    annotations = pd.read_csv(ground_truth_file)
    filename = prom_file.split('/')[-1].replace('.prom', '')
    print(f">>> Processing {filename}")
    predictions = pd.read_table(prom_file, header=None)
    avail_window = annotations[annotations['filename']==filename].reset_index(drop=True)
    target_window = predictions.reset_index(drop=True)

    l, r = 0, 0
    pairs = []

    while l<len(target_window) and r<len(avail_window):
        if target_window[3][l]==avail_window['word'][r]:
            pairs.append([(target_window[3][l], target_window[4][l]), (avail_window['word'][r], avail_window['pa.32'][r])])
            l+=1
            r+=1
        else:
            # print('mismatched pair', target_window[3][l], avail_window['word'][r], l, r)
            l+=1

    v1 = []
    v2 = []
    
    t1 = []
    t2 = []
    
    for pair in pairs:
        v1.append(pair[0][-1])
        v2.append(pair[1][-1])
        
        t1.append(pair[0][0])
        t2.append(pair[1][0])

    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    
    # print(v1)
    # print(v2)

    # print(t1)
    # print(t2)

    assert t1==t2
    assert v1.shape[0]==v2.shape[0]
    
    print(f"{v1.shape[0]} tokens compared, {len(avail_window)} tokens were available in annotation")
    if len(v1)>0 and len(v2)>0:
        result = 1 - spatial.distance.cosine(v1, v2)
    else:
        result = None

    return result
