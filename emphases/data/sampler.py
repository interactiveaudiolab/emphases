import torch

import emphases


###############################################################################
# Samplers
###############################################################################


# TODO - Set these boundaries to reasonable values to evenly divide your
#        dataset into buckets. You might consider more or fewer buckets
#        depending on the dataset.
# BOUNDARIES = [32, 300, 400, 500, 600, 700, 800, 900, 1000]
BOUNDARIES = [800, 1000, 1600]
# for audio lengths

###############################################################################
# Samplers
###############################################################################


class DistributedSampler(torch.utils.data.distributed.DistributedSampler):

    def __init__(self, dataset, shuffle=True):
        super().__init__(dataset, shuffle=shuffle)
        self.buckets, self.samples_per_bucket = create_buckets(
            dataset.lengths)
        self.total_size = sum(self.samples_per_bucket)

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            self.num_samples = self.total_size // world_size
        else:
            self.num_samples = self.total_size

    def __iter__(self):
      self.batches = make_batches(
          self.buckets,
          self.samples_per_bucket,
          self.epoch,
          self.shuffle)
      return iter(self.batches)

    def __len__(self):
        return self.num_samples // emphases.BATCH_SIZE


class Sampler(torch.utils.data.RandomSampler):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.buckets, self.samples_per_bucket = create_buckets(
            dataset.spectrogram_lengths)
        self.total_size = sum(self.samples_per_bucket)

    def __iter__(self):
        self.batches = make_batches(
            self.buckets,
            self.samples_per_bucket,
            self.epoch)
        return iter(self.batches)

    def __len__(self):
        """Retrieve the number of batches in an epoch"""
        return self.total_size // emphases.BATCH_SIZE

    def set_epoch(self, epoch):
        self.epoch = epoch


###############################################################################
# Sampler utilities
###############################################################################


def bisect(x, lo=0, hi=None):
    if hi is None:
        hi = len(BOUNDARIES) - 1

    if hi > lo:
        mid = (hi + lo) // 2
        if BOUNDARIES[mid] < x and x <= BOUNDARIES[mid+1]:
            return mid
        elif x <= BOUNDARIES[mid]:
            return bisect(x, lo, mid)
        else:
            return bisect(x, mid + 1, hi)
    return -1


def create_buckets(lengths):
    # Initialize buckets
    buckets = [[] for _ in range(len(BOUNDARIES) - 1)]

    # Populate buckets
    for i, length in enumerate(lengths):
        index = bisect(length)
        if index != -1:
            buckets[index].append(i)

    # Remove empty buckets
    buckets = [bucket for bucket in buckets if bucket]

    # Get number of GPUs
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    samples_per_bucket = []
    for bucket in buckets:
        total_batch_size = world_size * emphases.BATCH_SIZE
        remainder = (
            (total_batch_size - (len(bucket) % total_batch_size)) %
            total_batch_size)
        samples_per_bucket.append(len(bucket) + remainder)
    return buckets, samples_per_bucket


def make_batches(buckets, samples_per_bucket, epoch, shuffle=True):
    # Deterministic shuffling based on current epoch
    g = torch.Generator()
    g.manual_seed(epoch)

    indices = []
    if shuffle:
        for bucket in buckets:
            indices.append(torch.randperm(len(bucket), generator=g).tolist())
    else:
        for bucket in buckets:
            indices.append(list(range(len(bucket))))

    batches = []
    for i in range(len(buckets)):
        bucket = buckets[i]
        len_bucket = len(bucket)
        ids_bucket = indices[i]
        num_samples_bucket = samples_per_bucket[i]

        # Add extra samples to make it evenly divisible
        rem = num_samples_bucket - len_bucket
        ids_bucket = ids_bucket + ids_bucket * \
            (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

        # Subsample
        if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                ids_bucket = ids_bucket[rank::world_size]

        # Batch
        for j in range(len(ids_bucket) // emphases.BATCH_SIZE):
            batch = [
                bucket[idx]
                for idx in ids_bucket[j * emphases.BATCH_SIZE:(j + 1) * emphases.BATCH_SIZE]]
            batches.append(batch)

    if shuffle:
        batch_ids = torch.randperm(len(batches), generator=g).tolist()
        batches = [batches[i] for i in batch_ids]

    return batches
