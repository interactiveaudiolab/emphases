import torch

import emphases


###############################################################################
# Batch collation
###############################################################################


def collate(batch):
    """Batch collation"""
    # Unpack
    features, scores, word_bounds, alignments, audios, stems = zip(*batch)

    # Get word lengths
    word_lengths = torch.tensor(
        [bounds.shape[-1] for bounds in word_bounds],
        dtype=torch.long)
    max_word_length = word_lengths.max().item()

    # Get frame lengths
    frame_lengths = torch.tensor(
        [feat.shape[-1] for feat in features],
        dtype=torch.long)
    max_frame_length = frame_lengths.max().item()

    # Network output lengths
    if emphases.METHOD in ['framewise', 'attention']:
        output_lengths = frame_lengths
        max_output_length = max_frame_length
    elif emphases.METHOD in ['prominence', 'pitch_variance', 'duration_variance', 'wordwise']:
        output_lengths = word_lengths
        max_output_length = max_word_length
    else:
        raise ValueError(f'Inference method {emphases.METHOD} is not defined')

    # Allocate padded tensors
    padded_features = torch.zeros((len(features), emphases.NUM_FEATURES, max_frame_length))
    padded_scores = torch.zeros((len(scores), 1, max_output_length))
    padded_bounds = torch.zeros(
        (len(word_bounds), 2, max_word_length),
        dtype=torch.long)
    padded_audio = torch.zeros(
        (len(audios), 1, max_frame_length * emphases.HOPSIZE))
    mask = torch.zeros((len(scores), 1, max_output_length))

    # Place batch in padded tensors
    iterator = enumerate(
        zip(
            word_bounds,
            audios,
            features,
            scores,
            frame_lengths,
            word_lengths,
            output_lengths))
    for (
        i,
        (bounds, audio, feat, score, frame_length, word_length, output_length)
    ) in iterator:

        # Pad features
        padded_features[i, :, :frame_length] = feat

        # Pad scores
        padded_scores[i, :, :output_length] = score[:, :output_length]

        # Pad word bounds
        padded_bounds[i, :, :word_length] = bounds[:, :word_length]

        # Pad audio
        end_sample = frame_length * emphases.HOPSIZE
        padded_audio[i, :, :end_sample] = audio[:, :end_sample]

        # Create mask
        mask[i, :, :output_length] = 1.

    return (
        padded_features,
        padded_scores,
        padded_bounds,
        word_lengths,
        mask,
        alignments,
        padded_audio,
        stems)
