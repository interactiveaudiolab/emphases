import argparse
from pathlib import Path

import emphases


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=emphases.EVALUATION_DATASETS,
        help='The datasets to evaluate')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='The checkpoint file to evaluate')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for evaluation')

    return parser.parse_known_args()[0]


if __name__ == '__main__':
    emphases.evaluate.datasets(**vars(parse_args()))
