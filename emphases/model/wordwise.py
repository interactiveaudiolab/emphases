import functools

import torch

import emphases


###############################################################################
# Wordwise model
###############################################################################


class Wordwise(torch.nn.Module):

    def __init__(
            self,
            hidden_channels=emphases.HIDDEN_CHANNELS):
        super().__init__()
        
        #Setup frame encoder (output_channels=hidden channels because stopping at latent space)
        self.frame_encoder = emphases.model.Component(output_channels=hidden_channels)

        # Setup word decoder
        self.word_decoder = emphases.model.Component(input_channels=hidden_channels)

    def forward(self, features, word_bounds, word_lengths, *_):
        # Embed frames
        frame_embedding = self.frame_encoder(features, word_bounds, word_lengths)

        # Get maximum number of words
        max_word_length = word_lengths.max().item()

        # Allocate memory for word embeddings
        word_embeddings = torch.zeros(
            (
                frame_embedding.shape[0],
                frame_embedding.shape[1],
                max_word_length
            ),
            device=features.device)

        # Populate word embeddings
        i = 0
        iterator = enumerate(zip(frame_embedding, word_bounds, word_lengths))
        for i, (embedding, bounds, length) in iterator:
            for j in range(length):
                start, end = bounds[0, j], bounds[1, j]
                word_embeddings[i, :, j] = embedding[:, start:end].mean(dim=1)

        # Infer emphasis scores from word embeddings
        return self.word_decoder(word_embeddings, word_bounds, word_lengths)
