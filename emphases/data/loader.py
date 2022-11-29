import torch

import emphases


def loaders(dataset, train_partition, valid_partition, gpu=None):
    """Setup data loaders"""
    # Get dataset
    train_dataset = emphases.data.Dataset(dataset, train_partition)
    valid_dataset = emphases.data.Dataset(dataset, valid_partition)

    # Get sampler
    if torch.distributed.is_initialized():
        train_sampler = emphases.data.sampler.DistributedSampler(
            train_dataset,
            shuffle=True)
    else:
        train_sampler = emphases.data.sampler.Sampler(train_dataset)

    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=emphases.NUM_WORKERS,
        shuffle=False,
        pin_memory=gpu is not None,
        collate_fn=emphases.data.collate,
        batch_sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        num_workers=emphases.NUM_WORKERS,
        shuffle=False,
        batch_size=emphases.BATCH_SIZE, # changing it from 1 to BATCH_SIZE
        pin_memory=gpu is not None,
        drop_last=False,
        collate_fn=emphases.data.collate)

    return train_loader, valid_loader
