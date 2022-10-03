import torch

import emphases


def loaders(dataset):
    """Retrieve data loaders for training and evaluation"""
    return loader(dataset, 'train'), loader(dataset, 'valid')


def loader(dataset, partition):
    """Retrieve a data loader"""
    return torch.utils.data.DataLoader(
        dataset=emphases.data.Dataset(dataset, partition),
        batch_size=emphases.BATCH_SIZE,
        shuffle=partition == 'train',
        num_workers=emphases.NUM_WORKERS,
        pin_memory=True,
        collate_fn=emphases.data.collate)
