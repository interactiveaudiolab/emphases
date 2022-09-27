import numpy as np
import crepe

def combined(audio, sample_rate):
    """Extract the combined representation of pitch, duration, and loudness"""
    # Extract features
    features = (
        pitch(audio, sample_rate),
        duration(audio, sample_rate),
        loudness(audio, sample_rate))

    # TODO - Combine features
    pass


def duration(audio, sample_rate):
    """Extract duration features"""
    # TODO
    pass


def loudness(audio, sample_rate):
    """Extract loudness features"""
    # TODO
    pass


def pitch(audio, sample_rate, method="crepe"):
    """Extract pitch features (F0)"""
    # TODO

    if method=="crepe":
        time, frequency, confidence, activation = crepe.predict(audio, sample_rate, viterbi=False)
        return (time, frequency, confidence, activation)
    