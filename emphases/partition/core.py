import json
import random
import glob
import emphases
import os

def datasets(datasets, overwrite=False):
    """Partition datasets"""
    for dataset in datasets:

        # Check if partition already exists
        file = emphases.PARTITION_DIR / f'{dataset}.json'
        if file.exists():
            if not overwrite:
                print(f'Not overwriting existing partition {file}')
                continue

        # Random seed
        random.seed(emphases.RANDOM_SEED)

        # TODO - make partition
        partition = None
        if dataset=='Buckeye':
            partition = buckeye(dataset)

        # Save to disk
        file.parent.mkdir(exist_ok=True, parents=True)
        with open(file, 'w') as file:
            json.dump(partition, file, ensure_ascii=False, indent=4)


def buckeye(dataset):
    """Partition the Buckeye dataset"""

    cache_path = emphases.CACHE_DIR / dataset
    wavs_path = os.path.join(cache_path, 'wavs')
    wavs = glob.glob(os.path.join(cache_path, '*/*.wav'))
    speakers = [file_.split('/')[-1].replace('.wav', '') for file_ in wavs]
    random.seed(emphases.RANDOM_SEED)
    random.shuffle(speakers)

    length = len(speakers)

    # Get split locations
    left = int(emphases.SPLIT_SIZE_TRAIN * length)
    right = left + int(emphases.SPLIT_SIZE_VALID * length)

    partition = {
        "train": speakers[:left],
        "valid": speakers[left:right],
        "test": speakers[right:]
    }
    
    return partition
