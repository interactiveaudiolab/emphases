import torch
import torchutil

import emphases


###############################################################################
# Aggregate metric
###############################################################################


class Metrics:

    def __init__(self, predicted_stats, target_stats):
        self.correlation = torchutil.metrics.PearsonCorrelation(
            *predicted_stats(),
            *target_stats())
        self.bce = BinaryCrossEntropy()
        self.mse = MeanSquaredError()

    def __call__(self):
        return {
            'pearson_correlation': self.correlation(),
            'bce': self.bce(),
            'mse': self.mse()}

    def update(
        self,
        logits,
        targets,
        word_lengths):
        # Detach from graph
        logits = logits.detach()

        # Word resolution sequence mask
        mask = emphases.model.mask_from_lengths(word_lengths)
        logits, targets = logits[mask], targets[mask]

        # Update cross entropy
        self.bce.update(logits, targets)

        # Update squared error
        self.mse.update(emphases.postprocess(logits), targets)

        # Update pearson correlation
        self.correlation.update(emphases.postprocess(logits), targets)

    def reset(self):
        self.correlation.reset()
        self.bce.reset()
        self.mse.reset()


###############################################################################
# Individual metrics
###############################################################################


class BinaryCrossEntropy(torchutil.metrics.Average):

    def update(self, scores, targets):
        if emphases.LOSS == 'bce':

            # Get values from logits
            values = torch.nn.functional.binary_cross_entropy_with_logits(
                scores,
                targets,
                reduction='none')

        else:

            # Get values from probabilities
            x, y = torch.clamp(scores, 0., 1.), targets
            values = -(
                y * torch.log(x + 1e-6) + (1 - y) * torch.log(1 - x + 1e-6))

        # Update
        super().update(values, values.numel())


# TODO - fix scaling
class MeanSquaredError(torchutil.metrics.Average):

    def update(
        self,
        scores,
        targets):
        # Compute sum of MSE
        values = torch.nn.functional.mse_loss(
            scores,
            targets,
            reduction='none')

        # Update
        super().update(values, values.numel())


###############################################################################
# Utilities
###############################################################################


class Statistics(torchutil.metrics.MeanStd):

    def update(self, values, lengths):
        # Sequence mask
        mask = emphases.model.mask_from_lengths(lengths)

        # Update
        super().update(values[mask].flatten().tolist())
