import json

import torch

import emphases

import numpy as np

###############################################################################
# Evaluate
###############################################################################


def datasets(datasets, checkpoint=emphases.DEFAULT_CHECKPOINT, gpu=None):
    """Perform evaluation"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Containers for results
    overall, granular = {}, {}

    # Get metric class
    metric_fn = emphases.evaluate.Metrics

    # Per-file metrics
    file_metrics = metric_fn()

    # Per-dataset metrics
    dataset_metrics = metric_fn()

    # Aggregate metrics over all datasets
    aggregate_metrics = metric_fn()

    # Evaluate each dataset
    for dataset in datasets:

        # Reset dataset metrics
        dataset_metrics.reset()

        # Setup test dataset
        iterator = emphases.iterator(
            emphases.data.loader(dataset, 'test'),
            f'Evaluating {emphases.CONFIG} on {dataset}')

        # Iterate over test set
        for _, targets, word_bounds, _, _, alignments, audio, stem_name in iterator:

            # Reset file metrics
            file_metrics.reset()

            # Get predicted scores
            scores = emphases.from_alignment_and_audio(
                alignments[0],
                audio[0],
                emphases.SAMPLE_RATE,
                checkpoint=checkpoint,
                batch_size=emphases.MAX_FRAMES_PER_BATCH,
                pad=True,
                gpu=gpu)[None]
            
            if isinstance(scores, np.ndarray):
                scores = torch.from_numpy(scores)
            
            if emphases.METHOD in ['framewise', 'attention'] and emphases.FRAMEWISE_RESAMPLE is not None:
                # Get center time of each word in frames (we know that the targets are accurate here since they're interpolated from here)
                word_centers = \
                    word_bounds[:, 0] + (word_bounds[:, 1] - word_bounds[:, 0]) // 2

                #Allocate tensors for wordwise scores and targets
                word_scores = torch.zeros(word_centers.shape, device=device)
                word_targets = torch.zeros(word_centers.shape, device=device)
                word_masks = torch.zeros(word_centers.shape, device=device)

                for stem in range(targets.shape[1]): #Iterate over batch
                    stem_word_centers = word_centers[stem]
                    stem_word_targets = targets.squeeze(1)[stem, stem_word_centers]
                    stem_word_mask = torch.where(stem_word_centers == 0, 0, 1)

                    word_targets[stem] = stem_word_targets
                    word_masks[stem] = stem_word_mask

                    for i, (start, end) in enumerate(word_bounds[stem].T):
                        word_outputs = scores.squeeze(1)[stem, start:end]
                        method = emphases.FRAMEWISE_RESAMPLE
                        if method == 'max':
                            word_score = word_outputs.max()
                        elif method == 'avg':
                            word_score = word_outputs.mean()
                        else:
                            raise ValueError(f'Interpolation method {method} is not defined')
                        word_scores[stem, i] = word_score

                scores = word_scores
                targets = word_targets
                mask = word_masks

            # Update metrics
            lengths = torch.tensor(
                len(scores),
                dtype=torch.long,
                device=device)
            file_metrics.update(scores, targets.to(device), mask)
            dataset_metrics.update(scores, targets.to(device), mask)
            aggregate_metrics.update(scores, targets.to(device), mask)

            # Copy results
            granular[f'{dataset}/{stem_name[0]}'] = file_metrics()
        overall[dataset] = dataset_metrics()
    overall['aggregate'] = aggregate_metrics()

    # Write to json files
    directory = emphases.EVAL_DIR / emphases.CONFIG
    directory.mkdir(exist_ok=True, parents=True)
    with open(directory / 'overall.json', 'w') as file:
        json.dump(overall, file, indent=4)
    with open(directory / 'granular.json', 'w') as file:
        json.dump(granular, file, indent=4)
