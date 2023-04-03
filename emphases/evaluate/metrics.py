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
        self.pearson_correlation.update(scores, targets, word_bounds, mask)
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

    def update(self, scores, targets, word_bounds, mask):
        scores[mask == 0] = 0.
        targets[mask == 0] = 0.

        if emphases.DOWNSAMPLE_LOCATION=="intermediate":
            corr_matrix = torch.corrcoef(
                torch.cat([scores, targets]).squeeze(1)
            )
            n = scores.shape[0]*2
            rows = torch.arange(0, n//2)
            cols = torch.arange(n // 2, n)
            batch_sum = sum(torch.diagonal(corr_matrix[rows[:, None], cols[None, :]], 0))

            self.total += batch_sum
            self.count += scores.shape[0]

        else:
            # TODO
            pass
            
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
