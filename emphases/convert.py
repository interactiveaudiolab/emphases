import emphases


###############################################################################
# Time conversions
###############################################################################


def frames_to_samples(frames):
    """Convert number of frames to samples"""
    return frames * emphases.HOPSIZE


def frames_to_seconds(frames):
    """Convert number of frames to seconds"""
    return frames * emphases.HOPSIZE_SECONDS


def seconds_to_frames(seconds):
    """Convert seconds to number of frames"""
    return samples_to_frames(seconds_to_samples(seconds))


def seconds_to_samples(seconds):
    """Convert seconds to number of samples"""
    return seconds * emphases.SAMPLE_RATE


def samples_to_frames(samples):
    """Convert samples to number of frames"""
    return samples // emphases.HOPSIZE


def samples_to_seconds(samples):
    """Convert number of samples to seconds"""
    return samples / emphases.SAMPLE_RATE
