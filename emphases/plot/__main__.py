import argparse
from pathlib import Path

import emphases

###############################################################################
# Periodicity threshold figure
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Create periodicity threshold figure')
    parser.add_argument(
        '--names',
        required=True,
        nargs='+',
        help='Corresponding labels for each evaluation')
    parser.add_argument(
        '--evaluations',
        type=Path,
        required=True,
        nargs='+',
        help='The evaluations to plot')
    parser.add_argument(
        '--x',
        type=float,
        required=True,
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
    return parser.parse_known_args()[0]


emphases.plot.from_evaluations(**vars(parse_args()))