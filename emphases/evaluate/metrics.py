import math

import torch

import emphases


###############################################################################
# Aggregate metric
###############################################################################


class Metrics:

    def __init__(self, *stats):
        self.correlation = PearsonCorrelation(*stats)
        self.bce = BinaryCrossEntropy()
        self.mse = MeanSquaredError()

    def __call__(self):
        return self.correlation() | self.bce() | self.mse()

    def update(
        self,
        logits,
        targets,
        frame_lengths,
        word_bounds,
        word_lengths):
        # Detach from graph
        logits = logits.detach()

        # Update cross entropy
        self.bce.update(
            logits,
            targets,
            frame_lengths,
            word_bounds,
            word_lengths)

        # Update squared error
        self.mse.update(
            logits if emphases.LOSS == 'mse' else emphases.postprocess(logits),
            targets,
            frame_lengths,
            word_bounds,
            word_lengths)

        # Update pearson correlation
        self.correlation.update(
            emphases.postprocess(logits),
            targets,
            word_lengths)

    def reset(self):
        self.correlation.reset()
        self.bce.reset()
        self.mse.reset()


###############################################################################
# Individual metrics
###############################################################################


class BinaryCrossEntropy:

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'bce': (self.total / self.count).item()}

    def update(
        self,
        scores,
        targets,
        frame_lengths,
        word_bounds,
        word_lengths):
        if emphases.LOSS == 'mse':
            scores = torch.clamp(scores, 1e-4, 1 - 1e-4)
            scores = torch.log(scores / (1 - scores))
        self.total += emphases.train.loss(
            scores,
            targets,
            frame_lengths,
            word_bounds,
            word_lengths,
            loss_fn='bce')
        self.count += word_lengths.sum()

    def reset(self):
        self.count = 0
        self.total = 0.


class MeanSquaredError:

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'rmse': torch.sqrt(self.total / self.count).item()}

    def update(
        self,
        scores,
        targets,
        frame_lengths,
        word_bounds,
        word_lengths):
        self.total += emphases.train.loss(
            scores,
            targets,
            frame_lengths,
            word_bounds,
            word_lengths,
            loss_fn='mse')
        self.count += word_lengths.sum()

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
            1. / (self.std * self.target_std + 1e-6) *
            (self.total / self.count).item())
        return {'pearson_correlation': correlation}

    def update(self, scores, targets, word_lengths):
        # Word resolution sequence mask
        mask = emphases.model.mask_from_lengths(word_lengths)

        # Update
        self.total += sum(
            (scores[mask] - self.mean) * (targets[mask] - self.target_mean))
        self.count += word_lengths.sum()

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
        std = math.sqrt(self.m2 / (self.count - 1))
        return self.mean, std

    def update(self, x):
        for y in x.flatten().tolist():
            self.count += 1
            delta = y - self.mean
            self.mean += delta / self.count
            delta2 = y - self.mean
            self.m2 += delta * delta2

    def reset(self):
        self.m2 = 0.
        self.mean = 0.
        self.count = 0
