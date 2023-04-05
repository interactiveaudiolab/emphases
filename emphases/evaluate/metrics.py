import torch

import emphases


###############################################################################
# Aggregate metric
###############################################################################


class Metrics:

    def __init__(self, stats):
        self.pearson_correlation = PearsonCorrelation(stats)
        self.loss = Loss()

    def __call__(self):
        return self.pearson_correlation() | self.loss()

    def update(self, scores, targets, word_bounds, mask=None):
        # Detach from graph
        scores = scores.detach()

        # Default to evaluating on all sequence elements
        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)

        # Update
        self.pearson_correlation.update(scores, targets, mask)
        self.loss.update(scores, targets, word_bounds, mask)

    def reset(self):
        self.pearson_correlation.reset()
        self.loss.reset()


###############################################################################
# Individual metrics
###############################################################################


class PearsonCorrelation:

    def __init__(self, stats):
        self.reset()
        self.mean = stats.get('prediction_mean', None)
        self.std = stats.get('prediction_std', None)
        self.target_mean = stats.get('target_mean', None)
        self.target_std = stats.get('target_std', None)

    def __call__(self):
        correlation = (
            1. / (self.std * self.target_std) *
            (self.total / self.count).item())
        return {'pearson_correlation': correlation}

    def update(self, scores, targets, mask):
        self.total += sum(
            (scores[mask] - self.mean) * (targets[mask] - self.target_mean))
        self.count += mask.sum()

    def reset(self):
        self.count = 0
        self.total = 0.

class Loss():

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'loss': (self.total / self.count).item()}

    def update(self, scores, targets, word_bounds, mask):
        self.total += emphases.train.loss(scores, targets, word_bounds, mask)
        self.count += mask.sum()

    def reset(self):
        self.count = 0
        self.total = 0.
