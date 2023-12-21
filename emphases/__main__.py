import argparse
from pathlib import Path

import emphases


###############################################################################
# Determine which words in a speech file are emphasized
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Determine which words in a speech file are emphasized')
    parser.add_argument(
        '--text_files',
        type=Path,
        nargs='+',
        required=True,
        help='The speech transcript (.txt) or alignment (.TextGrid) files')
    parser.add_argument(
        '--audio_files',
        type=Path,
        nargs='+',
        required=True,
        help='The corresponding speech audio files')
    parser.add_argument(
        '--output_prefixes',
        type=Path,
        nargs='+',
        required=False,
        help='output_prefixes: The output files. Defaults to text files stems.')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='The model checkpoint to use for inference')
    parser.add_argument(
        '--batch_size',
        type=int,
        help='The maximum number of frames per batch')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to run inference on')
    return parser.parse_args()


if __name__ == '__main__':
    emphases.from_files_to_files(**vars(parse_args()))
