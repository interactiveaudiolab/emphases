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
        type=Path,
        nargs='+',
        help='The evaluations to plot')
    parser.add_argument(
        '--x',
        type=float,
        nargs='+',
        help='x values for each evaluation')
    parser.add_argument(
        '--output_file',
        type=Path,
        required=True,
        help='The output jpg file')
    parser.add_argument(
        '--x_label',
        type=str,
        required=True,
        help='Label for x axis')
    parser.add_argument(
        '--data',
        type=Path,
        help='CSV file with evaluations and x values'
    )
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    emphases.plot.scaling_laws(**vars(parse_args()))
