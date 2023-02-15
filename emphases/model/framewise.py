import functools

import torch

import emphases


###############################################################################
# Framewise model
###############################################################################


class Framewise(torch.nn.Module):

    def __init__(
        self):
        super().__init__()

        self.encoder = emphases.model.Component()


    def forward(self, features, word_bounds, word_lengths, mask=None):
        scores = self.encoder(features, word_bounds, word_lengths, mask)
        if not emphases.MODEL_TO_WORDS:
            return scores
        word_centers = \
            word_bounds[:, 0] + (word_bounds[:, 1] - word_bounds[:, 0]) // 2
    
        #Allocate tensors for wordwise scores and targets
        word_scores = torch.zeros(word_centers.shape, device=scores.device)

        for stem in range(word_centers.shape[0]): #Iterate over batch
            for i, (start, end) in enumerate(word_bounds[stem].T):
                word_outputs = scores.squeeze(1)[stem, start:end]
                method = emphases.FRAMEWISE_RESAMPLE
                if method == 'max':
                    word_score = word_outputs.max()
                elif method == 'avg':
                    word_score = word_outputs.mean()
                else:
                    raise ValueError(f'Interpolation method {method} is not defined')
                word_scores[stem, i] = word_score
        return word_scores.unsqueeze(1)
