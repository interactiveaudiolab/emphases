import json
import math

import emphases


###############################################################################
# Create figure
###############################################################################


def from_evaluations(names, evaluations, x, output_file, x_label):
    """Plot periodicity thresholds"""
    import matplotlib.pyplot as plt

    # Create plot
    figure, axis = plt.subplots(figsize=(7, 3))

    # Make pretty
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)
    x_range = max(x) - min(x)
    x_ticks = [min(x) + x_range * 0.25 * step for step in range(0, 5)]
    axis.set_xlim([min(x) - 0.1 * x_range, max(x) + 0.1 * x_range])
    y_ticks = [step / 4 - 1 for step in range(0, 9)]
    axis.get_xaxis().set_ticks(x_ticks)
    axis.get_yaxis().set_ticks(y_ticks)
    axis.tick_params(axis=u'both', which=u'both',length=0)
    axis.set_xlabel(x_label)
    axis.set_ylabel('Pearson correlation')
    for tick in y_ticks:
        axis.axhline(tick, color='gray', linestyle='--', linewidth=.8)

    y = []
    # Iterate over evaluations to plot
    for name, evaluation in zip(names, evaluations):
        directory = emphases.EVAL_DIR / evaluation

        # Load results
        with open(directory / 'overall.json') as file:
            y_val = json.load(file)['aggregate']['pearson_correlation']

        y.append(y_val)

        # Plot
    
    axis.plot(x, y)

    # Add legend
    # axis.legend(frameon=False, loc='upper right')

    # Save
    figure.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300)