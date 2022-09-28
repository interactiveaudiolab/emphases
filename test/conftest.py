from pathlib import Path

import pytest
import soundfile


TEST_ASSETS_DIR = Path(__file__).parent / 'assets'


###############################################################################
# Testing fixtures
###############################################################################


@pytest.fixture(scope='session')
def audio_and_sample_rate():
    """Retrieve the test audio"""
    return soundfile.read(path('test.wav'))

@pytest.fixture(scope='session')
def dataset():
    """Preload the dataset"""
    return emphases.data.Dataset('DATASET', 'valid')

@pytest.fixture(scope='session')
def model():
    """Preload the model"""
    return emphases.model.Model()

@pytest.fixture(scope='session')
def text():
    """Retrieve the test transcript"""
    with open(path('test.txt')) as file:
        return file.read()


###############################################################################
# Utilities
###############################################################################


def path(file):
    """Retrieve the path to the test file"""
    return Path(__file__) / 'assets' / file
