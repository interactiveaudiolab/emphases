import torch

# import penne

from scipy import spatial
import pandas as pd
import numpy as np

###############################################################################
# Individual metrics
###############################################################################

def cents(a, b):
    """Compute pitch difference in cents"""
    return penne.OCTAVE * torch.log2(a / b)

def cosine_sim(v1, v2):
    if torch.is_tensor(v1):
        v1 = v1.numpy()

    if torch.is_tensor(v2):
        v2 = v2.numpy()

    assert v1.shape[0]==v2.shape[0]
    
    if len(v1)>0 and len(v2)>0:
        result = 1 - spatial.distance.cosine(v1, v2)
    else:
        result = None

    return result

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


###############################################################################
# Constants
###############################################################################


# Evaluation threshold for RPA and RCA
THRESHOLD = 50  # cents


###############################################################################
# Aggregate metric
###############################################################################


# class Metrics:

#     def __init__(self):
#         self.accuracy = Accuracy()
#         self.f1 = F1()
#         self.l1 = L1()
#         self.loss = Loss()
#         self.rca = RCA()
#         self.rmse = RMSE()
#         self.rpa = RPA()

#     def __call__(self):
#         return (
#             self.accuracy() |
#             self.f1() |
#             self.l1() |
#             self.loss() |
#             self.rca() |
#             self.rmse() |
#             self.rpa())

#     def update(self, logits, bins, target, voiced):
#         # Detach from graph
#         logits = logits.detach()

#         # Update loss
#         self.loss.update(logits[:, :penne.PITCH_BINS], bins)

#         # Decode bins, pitch, and periodicity
#         with penne.time.timer('decode'):
#             predicted, pitch, periodicity = penne.postprocess(logits)

#         # Mask unvoiced
#         pitch, target = pitch[voiced], target[voiced]

#         # Update pitch metrics
#         self.accuracy.update(predicted[voiced], bins[voiced])
#         self.l1.update(pitch, target)
#         self.rca.update(pitch, target)
#         self.rmse.update(pitch, target)
#         self.rpa.update(pitch, target)

#         # Update periodicity metrics
#         self.f1.update(periodicity, voiced)

#     def reset(self):
#         self.accuracy.reset()
#         self.f1.reset()
#         self.l1.reset()
#         self.loss.reset()
#         self.rca.reset()
#         self.rmse.reset()
#         self.rpa.reset()


# ###############################################################################
# # Individual metrics
# ###############################################################################


# class Accuracy:

#     def __init__(self):
#         self.reset()

#     def __call__(self):
#         return {'accuracy': (self.true_positives / self.count).item()}

#     def update(self, predicted, target):
#         self.true_positives += (predicted == target).sum()
#         self.count += predicted.shape[-1]

#     def reset(self):
#         self.true_positives = 0
#         self.count = 0


# class F1:

#     def __init__(self, thresholds=None):
#         if thresholds is None:
#             thresholds = sorted(list(set(
#                 [2 ** -i for i in range(1, 11)] +
#                 [.1 * i for i in range(10)])))
#         self.thresholds = thresholds
#         self.precision = [Precision() for _ in range(len(thresholds))]
#         self.recall = [Recall() for _ in range(len(thresholds))]

#     def __call__(self):
#         result = {}
#         iterator = zip(self.thresholds, self.precision, self.recall)
#         for threshold, precision, recall in iterator:
#             precision = precision()['precision']
#             recall = recall()['recall']
#             try:
#                 f1 = 2 * precision * recall / (precision + recall)
#             except ZeroDivisionError:
#                 f1 = 0.
#             result |= {
#                 f'f1-{threshold:.6f}': f1,
#                 f'precision-{threshold:.6f}': precision,
#                 f'recall-{threshold:.6f}': recall}
#         return result

#     def update(self, periodicity, voiced):
#         iterator = zip(self.thresholds, self.precision, self.recall)
#         for threshold, precision, recall in iterator:
#             predicted = periodicity > threshold
#             precision.update(predicted, voiced)
#             recall.update(predicted, voiced)

#     def reset(self):
#         """Reset the F1 score"""
#         for precision, recall in zip(self.precision, self.recall):
#             precision.reset()
#             recall.reset()


# class L1:
#     """L1 pitch distance in cents"""

#     def __init__(self):
#         self.reset()

#     def __call__(self):
#         return {'l1': (self.sum / self.count).item()}

#     def update(self, predicted, target):
#         self.sum += torch.abs(cents(predicted, target)).sum()
#         self.count += predicted.shape[-1]

#     def reset(self):
#         self.count = 0
#         self.sum = 0.


# class Loss():

#     def __init__(self):
#         self.reset()

#     def __call__(self):
#         return {'loss': (self.total / self.count).item()}

#     def update(self, logits, bins):
#         self.total += penne.train.loss(logits, bins)
#         self.count += bins.shape[-1]

#     def reset(self):
#         self.count = 0
#         self.total = 0.


# class Precision:

#     def __init__(self):
#         self.reset()

#     def __call__(self):
#         precision = (
#             self.true_positives /
#             (self.true_positives + self.false_positives)).item()
#         return {'precision': precision}

#     def update(self, predicted, voiced):
#         self.true_positives += (predicted & voiced).sum()
#         self.false_positives += (predicted & ~voiced).sum()

#     def reset(self):
#         self.true_positives = 0
#         self.false_positives = 0


# class Recall:

#     def __init__(self):
#         self.reset()

#     def __call__(self):
#         recall = (
#             self.true_positives /
#             (self.true_positives + self.false_negatives)).item()
#         return {'recall': recall}

#     def update(self, predicted, voiced):
#         self.true_positives += (predicted & voiced).sum()
#         self.false_negatives += (~predicted & voiced).sum()

#     def reset(self):
#         self.true_positives = 0
#         self.false_negatives = 0


# class RCA:
#     """Raw chroma accuracy"""

#     def __init__(self):
#         self.reset()

#     def __call__(self):
#         return {'rca': (self.sum / self.count).item()}

#     def update(self, predicted, target):
#         # Compute pitch difference in cents
#         difference = cents(predicted, target)

#         # Forgive octave errors
#         difference[difference > (penne.OCTAVE - THRESHOLD)] -= penne.OCTAVE
#         difference[difference < -(penne.OCTAVE - THRESHOLD)] += penne.OCTAVE

#         # Count predictions that are within 50 cents of target
#         self.sum += (torch.abs(difference) < THRESHOLD).sum()
#         self.count += predicted.shape[-1]

#     def reset(self):
#         self.count = 0
#         self.sum = 0


# class RMSE:
#     """Root mean square error of pitch distance in cents"""

#     def __init__(self):
#         self.reset()

#     def __call__(self):
#         return {'rmse': torch.sqrt(self.sum / self.count).item()}

#     def update(self, predicted, target):
#         self.sum += (cents(predicted, target) ** 2).sum()
#         self.count += predicted.shape[-1]

#     def reset(self):
#         self.count = 0
#         self.sum = 0.


# class RPA:
#     """Raw prediction accuracy"""

#     def __init__(self):
#         self.reset()

#     def __call__(self):
#         return {'rpa': (self.sum / self.count).item()}

#     def update(self, predicted, target):
#         difference = cents(predicted, target)
#         self.sum += (torch.abs(difference) < THRESHOLD).sum()
#         self.count += predicted.shape[-1]

#     def reset(self):
#         self.count = 0
#         self.sum = 0


