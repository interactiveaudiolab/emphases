"""__main__.py - entry point for emphases.evaluate"""


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
        help='The datasets to evaluate')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=emphases.DEFAULT_CHECKPOINT,
        help='The checkpoint file to evaluate')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for evaluation')

    return parser.parse_args()


if __name__ == '__main__':
    emphases.evaluate.datasets(**vars(parse_args()))
