"""core.py - data preprocessing"""


import emphases


###############################################################################
# Preprocess
###############################################################################


def datasets(datasets):
    """Preprocess a dataset

    Arguments
        name - string
            The name of the dataset to preprocess
    """
    for dataset in datasets:
        input_directory = emphases.DATA_DIR / dataset
        output_directory = emphases.CACHE_DIR / dataset

        # TODO - Perform preprocessing
        raise NotImplementedError
