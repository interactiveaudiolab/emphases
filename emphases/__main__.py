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
        'text_files',
        type=Path,
        nargs='+',
        help='Text file containing transcripts')
    parser.add_argument(
        'audio_files',
        type=Path,
        nargs='+',
        help='The corresponding speech audio files')
    parser.add_argument(
        'output_file',
        type=Path,
        nargs='+',
        required=False,
        help='Json files to save results. ' +
             'Defaults to text files with json extension.')
    return parser.parse_args()


if __name__ == '__main__':
    emphases.from_files_to_files(**vars(parse_args()))
