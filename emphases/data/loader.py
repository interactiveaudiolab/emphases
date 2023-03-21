import torch

import emphases


###############################################################################
# Dataloader
###############################################################################


def loader(dataset, partition=None, gpu=None, train_limit=None):
    """Retrieve a data loader"""
    # Get dataset
    dataset = emphases.data.Dataset(dataset, partition, train_limit=train_limit)

    # Get sampler
    sampler = emphases.data.sampler(dataset, partition)

    # Create loader
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=emphases.NUM_WORKERS,
        pin_memory=gpu is not None,
        collate_fn=emphases.data.collate,
        batch_sampler=sampler)
