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
        if dataset == 'buckeye':
            partition = buckeye()
        elif dataset == 'libritts':
            partition = libritts()

        # Save to disk
        file.parent.mkdir(exist_ok=True, parents=True)
        with open(file, 'w') as file:
            json.dump(partition, file, ensure_ascii=False, indent=4)


def buckeye():
    """Partition buckeye dataset"""
    # Get audio files
    directory = emphases.CACHE_DIR / 'buckeye'
    audio_files = directory.rglob('*.wav')

    # Get speakers
    # TODO - is there really only one file per speaker?
    speakers = [file.stem for file in audio_files]

    # Shuffle speakers
    random.seed(emphases.RANDOM_SEED)
    random.shuffle(speakers)

    # Get split locations
    left = int(emphases.SPLIT_SIZE_TRAIN * len(speakers))
    right = left + int(emphases.SPLIT_SIZE_VALID * len(speakers))

    # Partition
    return {
        "train": speakers[:left],
        "valid": speakers[left:right],
        "test": speakers[right:]}


def libritts():
    """Partition libritts dataset"""
    # TODO
    pass
