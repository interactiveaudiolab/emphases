import argparse

import emphases


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Partition datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=emphases.DATASETS,
        help='The datasets to partition')
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    emphases.partition.datasets(**vars(parse_args()))
