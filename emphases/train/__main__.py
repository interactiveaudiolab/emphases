import argparse
import shutil
from pathlib import Path

import emphases


###############################################################################
# Entry point
###############################################################################


def main(config, dataset, gpus=None):
    # Create output directory
    directory = emphases.RUNS_DIR / config.stem
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copyfile(config, directory / config.name)

    # Train
    checkpoint = emphases.train.run(
        dataset,
        directory,
        directory,
        directory,
        gpus)

    # Evaluate
    emphases.evaluate.datasets(emphases.EVALUATION_DATASETS, checkpoint, gpus[0])


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--config',
        type=Path,
        default=emphases.DEFAULT_CONFIGURATION,
        help='The configuration file')
    parser.add_argument(
        '--dataset',
        default='buckeye',
        help='The dataset to train on')
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        help='The gpus to run training on')
    return parser.parse_args()


if __name__ == '__main__':
    main(**vars(parse_args()))
