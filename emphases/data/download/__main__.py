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
        required=True,
        default=emphases.DATASETS,
        help='The datasets to download')
    return parser.parse_args()


emphases.data.download.datasets(**vars(parse_args()))
