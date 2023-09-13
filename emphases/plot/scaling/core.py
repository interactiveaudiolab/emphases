import matplotlib
import matplotlib.pyplot as plt
import torch

import emphases


###############################################################################
# Plot scaling laws
###############################################################################


def scaling_laws(
    evaluations,
    xlabel,
    output_file,
    yticks,
    scores=None,
    steps=None,
    sizes=None,
    text_offsets=None):
    """Plot scaling laws"""
    # Load evaluation results
    if scores is None or steps is None:
        scores, steps = [], []
        for evaluation in evaluations:
            path, score = emphases.checkpoint.best_path(
                emphases.RUNS_DIR / evaluation)
            checkpoint = torch.load(path, map_location='cpu')
            scores.append(score)
            steps.append(checkpoint['step'])

    # Get x values
    x = [int(eval.split('-')[-1]) for eval in evaluations]

    # Create plot
    figure, axis = plt.subplots(figsize=(8, 2))

    # Remove frame
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)

    # Format x axis
    x_range = max(x) - min(x)
    axis.set_xlim([0, max(x) + 0.1 * x_range])
    axis.get_xaxis().set_ticks(x)
    axis.set_xlabel(xlabel)
    axis.xaxis.set_ticks(x)
    axis.xaxis.set_ticklabels(x)

    # Format y axis
    axis.get_yaxis().set_ticks(yticks)
    axis.set_ylim([min(yticks) - .002, max(yticks) + .002])
    axis.tick_params(axis=u'both', which=u'both',length=0)
    axis.set_ylabel('Pearson correlation')

    # Grid lines
    for tick in yticks:
        axis.axhline(tick, color='gray', linestyle='--', linewidth=.8)

    # Plot
    colors = ['blue', 'orange', 'purple', 'red']
    for i in range(len(x)):
        axis.scatter(x[i], scores[i], color=colors[i])

    # Default text offset
    if text_offsets is None:
        text_offsets = [0.011] * len(evaluations)

    # Annotate
    for i in range(len(evaluations)):
        text = f'steps={steps[i]}'
        if sizes is not None:
            text += f'\nutterances={sizes[i]}'
        axis.text(
            x[i],
            scores[i] - text_offsets[i],
            text,
            horizontalalignment='center')

    # Save
    figure.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300)
