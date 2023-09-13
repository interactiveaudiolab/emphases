import json
import random
import emphases


###############################################################################
# Partition dataset
###############################################################################


def datasets(datasets):
    """Partition datasets"""
    for dataset in datasets:

        # Check if partition already exists
        file = emphases.PARTITION_DIR / f'{dataset}.json'

        # Random seed
        random.seed(emphases.RANDOM_SEED)

        # Make partition
        if dataset == 'automatic':
            partition = automatic()
        elif dataset == 'buckeye':
            partition = buckeye()
        elif dataset == 'libritts':
            partition = libritts()
        elif dataset == 'crowdsource':
            partition = crowdsource()
        else:
            raise ValueError(f'Dataset {dataset} is not defined')

        # Save to disk
        file.parent.mkdir(exist_ok=True, parents=True)
        with open(file, 'w') as file:
            json.dump(partition, file, ensure_ascii=False, indent=4)


###############################################################################
# Existing datasets
###############################################################################


def buckeye():
    """Partition buckeye dataset"""
    # Get audio files
    directory = emphases.CACHE_DIR / 'buckeye'
    audio_files = directory.rglob('*.wav')

    # Get stems
    stems = [file.stem for file in audio_files]

    # Partition
    return {'train': [], 'valid': [], 'test': stems}


def libritts():
    """Partition libritts dataset"""
    # Get audio files
    directory = emphases.CACHE_DIR / 'libritts'
    audio_files = directory.rglob('*.wav')

    # Get stems
    stems = [file.stem for file in audio_files]

    # Shuffle stems
    random.seed(emphases.RANDOM_SEED)
    random.shuffle(stems)

    # Get split locations
    left = int(emphases.SPLIT_SIZE_TRAIN * len(stems))
    right = left + int(emphases.SPLIT_SIZE_VALID * len(stems))

    # Only train on specified eighth for scaling law experiments
    if emphases.ONE_EIGHTH_UTTERANCES:

        # Partition
        speakers = [str(s) for s in emphases.data.download.LIBRITTS_SPEAKERS]
        train = [stem for stem in stems if stem.split('_')[0] in speakers]
        valid = [stem for stem in stems[left:right] if stem not in train]
        test = [stem for stem in stems[right:] if stem not in train]

    else:

        # Partition
        train = stems[:left]
        valid = stems[left:right]
        test = stems[right:]

    # Maybe limit training set size
    if emphases.MAX_TRAINING_UTTERANCES is not None:
        train = train[:emphases.MAX_TRAINING_UTTERANCES]

    return {'train': train, 'valid': valid, 'test': test}


###############################################################################
# Dataset creation
###############################################################################


def automatic():
    """Partition dataset created from trained model"""
    # Get audio files
    directory = emphases.CACHE_DIR / 'automatic'
    audio_files = directory.rglob('*.wav')

    # Get stems
    stems = [file.stem for file in audio_files]

    # Shuffle stems
    random.seed(emphases.RANDOM_SEED)
    random.shuffle(stems)

    # Get split locations
    left = int(emphases.SPLIT_SIZE_TRAIN * len(stems))
    right = left + int(emphases.SPLIT_SIZE_VALID * len(stems))

    # Partition
    return {
        'train': stems[:left],
        'valid': stems[left:right],
        'test': stems[right:]}


def crowdsource():
    """Partition crowdsourced dataset"""
    # Get audio files
    directory = emphases.CACHE_DIR / 'crowdsource'
    audio_files = directory.rglob('*.wav')

    # Get stems
    stems = [file.stem for file in audio_files]

    # Shuffle stems
    random.seed(emphases.RANDOM_SEED)
    random.shuffle(stems)

    # Get split locations
    left = int(emphases.SPLIT_SIZE_TRAIN * len(stems))
    right = left + int(emphases.SPLIT_SIZE_VALID * len(stems))

    # Only train on specified eighth for scaling law experiments
    if emphases.ONE_EIGHTH_UTTERANCES:

        # Partition
        speakers = [str(s) for s in emphases.data.download.LIBRITTS_SPEAKERS]
        train = [stem for stem in stems if stem.split('_')[0] in speakers]
        valid = [stem for stem in stems[left:right] if stem not in train]
        test = [stem for stem in stems[right:] if stem not in train]

    else:

        # Partition
        train = stems[:left]
        valid = stems[left:right]
        test = stems[right:]

    # Maybe limit training set size
    if emphases.MAX_TRAINING_UTTERANCES is not None:
        train = train[:emphases.MAX_TRAINING_UTTERANCES]

    return {'train': train, 'valid': valid, 'test': test}
