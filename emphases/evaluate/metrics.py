import torch

import emphases


###############################################################################
# Aggregate metric
###############################################################################


class Metrics:

    def __init__(self, *stats):
        self.pearson_correlation = PearsonCorrelation(*stats)
        self.loss = Loss()

    def __call__(self):
        return self.pearson_correlation() | self.loss()

    def update(self, scores, targets, word_bounds, mask=None):
        # Detach from graph
        scores = scores.detach()

        # Default to evaluating on all sequence elements
        if mask is None:
            mask = torch.ones_like(
                scores,
                dtype=torch.bool,
                device=scores.device)

        # Update
        self.pearson_correlation.update(scores, targets, mask)
        self.loss.update(scores, targets, word_bounds, mask)

    def reset(self):
        self.pearson_correlation.reset()
        self.loss.reset()


###############################################################################
# Individual metrics
###############################################################################


class Loss:

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


class PearsonCorrelation:

    def __init__(self, predicted_stats, target_stats):
        self.reset()
        self.mean, self.std = predicted_stats()
        self.target_mean, self.target_std = target_stats()

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


###############################################################################
# Utilities
###############################################################################


class Statistics:

    def __init__(self):
        self.reset()

    def __call__(self):
        variance = self.m2 / (self.count - 1)
        return self.mean, variance

    def update(self, x):
        for y in x.flatten():
            self.count += 1
            delta = y - self.mean
            self.mean += delta / self.count
            delta2 = y - self.mean
            self.m2 += delta * delta2

    def reset(self):
        self.m2 = 0.
        self.mean = 0.
        self.count = 0
