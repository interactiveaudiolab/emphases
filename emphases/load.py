import json

import emphases


###############################################################################
# Loading utilities
###############################################################################

def partition(dataset):
    """Load partitions for dataset"""
    with open(emphases.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)
