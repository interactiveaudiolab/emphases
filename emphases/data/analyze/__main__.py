import argparse

import emphases


###############################################################################
# Download datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Analyze datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=emphases.DATASETS,
        help='The datasets to analyze')
    return parser.parse_args()


emphases.data.analyze.analysis(**vars(parse_args()))
