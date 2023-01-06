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
        scores[mask == 0] = 0.
        targets[mask == 0] = 0.
        self.total += torch.nn.functional.cosine_similarity(
            scores.squeeze(1), targets.squeeze(1)).sum()
        self.count += scores.shape[0]

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
