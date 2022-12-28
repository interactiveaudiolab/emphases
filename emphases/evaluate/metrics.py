import torch

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

    def update(self, scores, targets, lengths):
        # Detach from graph
        scores = scores.detach()

        # Update loss
        self.similarity.update(scores, targets, lengths)
        self.loss.update(scores, targets, lengths)

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
        scores, targets = scores.where(mask), targets.where(mask)
        self.total += torch.nn.functional.cosine_similarity(scores, targets)
        self.count += mask.sum()

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
        self.count += mask.sum()

    def reset(self):
        self.count = 0
        self.total = 0.
