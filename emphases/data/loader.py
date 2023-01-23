import torch

import emphases


###############################################################################
# Dataloader
###############################################################################


def loader(dataset, partition=None, gpu=None):
    """Retrieve a data loader"""
    # Get dataset
    dataset = emphases.data.Dataset(dataset, partition)

    # Get sampler
    sampler = emphases.data.sampler(dataset, partition)

    # Get batch size
    if partition == 'train':

        # Maybe split batch over GPUs
        if torch.distributed.is_initialized():
            batch_size = \
                emphases.BATCH_SIZE // torch.distributed.get_world_size()
        else:
            batch_size = emphases.BATCH_SIZE

    elif partition == 'valid':
        batch_size = emphases.BATCH_SIZE
    elif partition == 'test':
        batch_size = 1
    else:
        raise ValueError(f'Partition {partition} is not defined')

    # Create loader
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=emphases.NUM_WORKERS,
        pin_memory=gpu is not None,
        collate_fn=emphases.data.collate,
        sampler=sampler)
