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


def pitch(audio, sample_rate):
    """Extract pitch features"""
    # TODO
    pass
