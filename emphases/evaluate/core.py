import emphases
from emphases.evaluate.metrics import eval_similarity


###############################################################################
# Evaluate
###############################################################################


def datasets(datasets, checkpoint=emphases.DEFAULT_CHECKPOINT, gpu=None):
    """Perform evaluation"""
    for dataset in datasets:
        if dataset == 'buckeye':
            buckeye()
        elif dataset == 'libritts':
            libritts()
        else:
            raise ValueError(f'Dataset {dataset} is not defined')


###############################################################################
# Evaluate
###############################################################################


def buckeye(ground_truth_file='utils/BuckEye-annotations.csv'):
    """Evaluate on buckeye dataset"""
    directory = emphases.EVAL_DIR / 'buckeye'

    # Iterate over test partition
    # TODO - this is iterating over all partitions, not test partition
    for file in directory.glob('*'):
        prominence_cosine_similarity = eval_similarity(directory / file, ground_truth_file)


def libritts():
    """Evaluate on libritts dataset"""
    # TODO
    pass
