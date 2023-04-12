import matplotlib.pyplot as plt

import torch


###############################################################################
# Plot prominence
###############################################################################


def scores(alignment, scores, targets=None):
    """Plot the aligned prominence scores"""
    figure, axis = plt.subplots(figsize=(30, 5))
    axis.set_axis_off()
    axis.set_ylim([0., 1.])

    # Get words, start times, and durations
    centers = [word.start() + word.duration() / 2. for word in alignment]
    duration = [word.duration() for word in alignment]

    # Plot scores
    axis.bar(
        centers,
        scores,
        duration,
        edgecolor='black')

    # Plot words and dividers
    for word in alignment:
        axis.text(
            word.start() + word.duration() / 2,
            .015,
            str(word),
            fontsize=10,
            rotation=90,
            horizontalalignment='center')
        axis.axvline(
            word.start(),
            color='gray',
            linewidth=.5,
            ymin=0.,
            ymax=1.,
            clip_on=False,
            linestyle='--')
    axis.axvline(
        alignment.duration(),
        color='gray',
        linewidth=.5,
        ymin=0.,
        ymax=1.,
        clip_on=False,
        linestyle='--')

    if targets is not None:

        # Plot targets
        axis.bar(centers, targets, duration)

        # Plot overlap
        overlap = torch.minimum(scores, targets)
        axis.bar(centers, overlap, duration, color='gray')

    return figure
