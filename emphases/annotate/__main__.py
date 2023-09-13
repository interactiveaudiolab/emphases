import argparse
from pathlib import Path

import emphases


###############################################################################
# Annotate emphases
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Perform emphasis annotation')
    parser.add_argument(
        '--annotation_config',
        type=Path,
        default=emphases.DEFAULT_ANNOTATION_CONFIG,
        help='The ReSEval configuration file for the annotation task')
    parser.add_argument(
        '--dataset',
        default='libritts',
        help='The dataset to annotate')
    parser.add_argument(
        '--directory',
        type=Path,
        default=emphases.ANNOTATION_DIR,
        help='The directory to save results to')
    parser.add_argument(
        '--remote',
        action='store_true',
        help='Run subjective evaluation remotely')
    parser.add_argument(
        '--production',
        action='store_true',
        help='Deploy the subjective evaluation to crowdsource participants')
    parser.add_argument(
        '--interval',
        type=int,
        default=120,
        help='The time between monitoring updates in seconds')
    return parser.parse_args()


if __name__ == '__main__':
    emphases.annotate.datasets(**vars(parse_args()))
