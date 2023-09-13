import argparse
from pathlib import Path

import emphases


###############################################################################
# Scaling laws plot
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Create scaling law figure')
    parser.add_argument(
        '--evaluations',
        type=str,
        nargs='+',
        required=True,
        help='The evaluations to plot')
    parser.add_argument(
        '--xlabel',
        type=str,
        required=True,
        help='Label for x axis')
    parser.add_argument(
        '--output_file',
        type=Path,
        required=True,
        help='The output jpg file')
    parser.add_argument(
        '--yticks',
        type=float,
        nargs='+',
        required=True,
        help='The y axis tick mark locations')
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        help='The number of utterances used in each evaluation')
    parser.add_argument(
        '--scores',
        type=float,
        nargs='+',
        help='The Pearson Correlation y values')
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        help='The number of training steps')
    parser.add_argument(
        '--text_offsets',
        type=float,
        nargs='+',
        help='The amount to space the text below the plot point')
    return parser.parse_args()


if __name__ == '__main__':
    emphases.plot.scaling.scaling_laws(**vars(parse_args()))
