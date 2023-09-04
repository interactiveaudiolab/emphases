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
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to run inference on')
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    emphases.data.preprocess.datasets(**vars(parse_args()))
