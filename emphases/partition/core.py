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
        elif dataset == 'annotate':
            partition = annotate()

        # Save to disk
        file.parent.mkdir(exist_ok=True, parents=True)
        with open(file, 'w') as file:
            json.dump(partition, file, ensure_ascii=False, indent=4)

def annotate():
    """Partition Annotated dataset"""
    # Get audio files
    directory = emphases.CACHE_DIR / 'annotate'
    audio_files = directory.rglob('*.wav')

    # Get speakers
    stems = [file.stem for file in audio_files]

    # Shuffle stems
    random.seed(emphases.RANDOM_SEED)
    random.shuffle(stems)

    # Get split locations
    left = int(emphases.SPLIT_SIZE_TRAIN * len(stems))
    right = left + int(emphases.SPLIT_SIZE_VALID * len(stems))

    # Partition
    return {
        "train": stems[:left],
        "valid": stems[left:right],
        "test": stems[right:]}



def buckeye():
    """Partition buckeye dataset"""
    # Get audio files
    directory = emphases.CACHE_DIR / 'buckeye'
    audio_files = directory.rglob('*.wav')

    # Get speakers
    stems = [file.stem for file in audio_files]

    # Shuffle stems
    random.seed(emphases.RANDOM_SEED)
    random.shuffle(stems)

    # Get split locations
    left = int(emphases.SPLIT_SIZE_TRAIN * len(stems))
    right = left + int(emphases.SPLIT_SIZE_VALID * len(stems))

    # Partition
    return {
        "train": stems[:left],
        "valid": stems[left:right],
        "test": stems[right:]}


def libritts():
    """Partition libritts dataset"""
    # TODO
    pass
