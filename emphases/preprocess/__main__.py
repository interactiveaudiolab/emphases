import argparse

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
        default=emphases.DATASETS,
        help='The datasets to preprocess')
    return parser.parse_args()


if __name__ == '__main__':
    emphases.preprocess.datasets(**vars(parse_args()))
