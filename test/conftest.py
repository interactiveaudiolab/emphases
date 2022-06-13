from pathlib import Path

import pytest
import soundfile



###############################################################################
# Testing fixtures
###############################################################################


@pytest.fixture(scope='session')
def audio_and_sample_rate():
    """Retrieve the test audio"""
    return soundfile.read(path('test.wav'))


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
