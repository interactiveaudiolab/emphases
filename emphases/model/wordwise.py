import functools

import torch

import emphases


###############################################################################
# Wordwise model
###############################################################################


class Wordwise(torch.nn.Module):

    def __init__(
            self,
            input_channels=emphases.NUM_FEATURES,
            output_channels=1,
            hidden_channels=128,
            kernel_size=5):
        super().__init__()

        # Setup frame encoder
        conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=kernel_size,
            padding='same')
        self.frame_encoder = torch.nn.Sequential(
            conv_fn(input_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, hidden_channels))

        # Setup word decoder
        self.word_decoder = torch.nn.Sequential(
            conv_fn(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, output_channels))

    def forward(self, features, word_bounds, word_lengths):
        # Embed frames
        frame_embedding = self.frame_encoder(features)

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
        return self.word_decoder(word_embeddings)
