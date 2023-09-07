import argparse

import emphases


###############################################################################
# Download datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=emphases.DATASETS,
        help='The datasets to download')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to run inference on')
    return parser.parse_known_args()[0]


emphases.data.download.datasets(**vars(parse_args()))
