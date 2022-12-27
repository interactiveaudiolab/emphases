import numpy as np
import pandas as pd
import torch
import scipy

import emphases


###############################################################################
# Aggregate metric
###############################################################################


class Metrics:

    def __init__(self):
        self.similarity = CosineSimilarity()
        self.loss = Loss()

    def __call__(self):
        return self.similarity() | self.loss()

    def update(self, scores, target):
        # Detach from graph
        scores = scores.detach()

        # Update loss
        self.similarity.update(scores, target)
        self.loss.update(scores, target)

    def reset(self):
        self.similarity.reset()
        self.loss.reset()


###############################################################################
# Individual metrics
###############################################################################


class CosineSimilarity:

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'similarity': (self.total / self.count).item()}

    def update(self, scores, targets, mask):
        # TODO - masking
        self.total += torch.nn.functional.cosine_similarity(scores, targets)
        self.count += scores.shape[-1]

    def reset(self):
        self.count = 0
        self.total = 0.


class Loss():

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'loss': (self.total / self.count).item()}

    def update(self, scores, targets, mask):
        self.total += emphases.train.loss(scores, targets, mask)
        self.count += scores.numel()

    def reset(self):
        self.count = 0
        self.total = 0.


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

    assert t1==t2
    assert v1.shape[0]==v2.shape[0]

    print(f"{v1.shape[0]} tokens compared, {len(avail_window)} tokens were available in annotation")
    if len(v1)>0 and len(v2)>0:
        result = 1 - scipy.spatial.distance.cosine(v1, v2)
    else:
        result = None

    return result
