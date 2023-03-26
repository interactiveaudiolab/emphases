import torch

import emphases


###############################################################################
# Aggregate metric
###############################################################################


class Metrics:

    def __init__(self):
        self.pearson_correlation = PearsonCorrelation()
        self.loss = Loss()

    def __call__(self):
        return self.pearson_correlation() | self.loss()

    def update(self, scores, targets, word_bounds, mask=None):
        # Detach from graph
        scores = scores.detach()

        # Default to evaluating on all sequence elements
        if mask is None:
            mask = torch.ones_like(scores)

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

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'pearson_correlation': (self.total / self.count).item()}

    def update(self, scores, targets, mask):
        # TODO - this has a bug that biases words in short sentences
        scores[mask == 0] = 0.
        targets[mask == 0] = 0.
        self.total += torch.corrcoef(
            torch.cat([scores, targets]).squeeze(1)
        )[:, 0][-1]
        self.count += scores.shape[0]

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
