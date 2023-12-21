import argparse
import shutil
from pathlib import Path

import torchutil

import emphases


###############################################################################
# Entry point
###############################################################################


def main(config, dataset, gpu=None):
    # Create output directory
    directory = emphases.RUNS_DIR / config.stem
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copyfile(config, directory / config.name)

    # Train
    emphases.train(dataset, directory, gpu)

    # Get best checkpoint
    checkpoint = torchutil.checkpoint.best_path(directory)[0]

    # Evaluate
    emphases.evaluate.datasets(emphases.EVALUATION_DATASETS, checkpoint, gpu)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--config',
        type=Path,
        help='The configuration file')
    parser.add_argument(
        '--dataset',
        default=emphases.TRAINING_DATASET,
        help='The dataset to train on')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The gpu to run training on')
    return parser.parse_args()


if __name__ == '__main__':
    main(**vars(parse_args()))
