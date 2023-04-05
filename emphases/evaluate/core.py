import json

import torch

import emphases


###############################################################################
# Evaluate
###############################################################################


def datasets(datasets, checkpoint=emphases.DEFAULT_CHECKPOINT, gpu=None):
    """Perform evaluation"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Containers for results
    overall, granular = {}, {}

    # target, prediction statistics of validation set
    validation_stats = emphases.train.load_target_stats('valid', emphases.dataset)

    # Get metric class
    metric_fn = emphases.evaluate.Metrics

    # Per-file metrics
    file_metrics = metric_fn(validation_stats)

    # Per-dataset metrics
    dataset_metrics = metric_fn(validation_stats)

    # Aggregate metrics over all datasets
    aggregate_metrics = metric_fn(validation_stats)

    # Evaluate each dataset
    for dataset in datasets:

        # Reset dataset metrics
        dataset_metrics.reset()

        # Setup test dataset
        iterator = emphases.iterator(
            emphases.data.loader(dataset, 'test', gpu),
            f'Evaluating {emphases.CONFIG} on {dataset}')
        
        # Iterate over test set
        for batch in iterator:

            # Unpack
            (_, _, word_bounds, _, targets, alignments, audio, stems) = batch

            # Reset file metrics
            file_metrics.reset()

            # Get predicted scores
            scores = emphases.from_alignment_and_audio(
                alignments[0],
                audio[0],
                emphases.SAMPLE_RATE,
                checkpoint=checkpoint,
                pad=True,
                gpu=gpu)[None]

            # Update metrics
            args = (scores, targets.to(device), word_bounds)
            file_metrics.update(*args)
            dataset_metrics.update(*args)
            aggregate_metrics.update(*args)

            # Copy results
            granular[f'{dataset}/{stems[0]}'] = file_metrics()
        overall[dataset] = dataset_metrics()
    overall['aggregate'] = aggregate_metrics()

    # Write to json files
    directory = emphases.EVAL_DIR / emphases.CONFIG
    directory.mkdir(exist_ok=True, parents=True)
    with open(directory / 'overall.json', 'w') as file:
        json.dump(overall, file, indent=4)
    with open(directory / 'granular.json', 'w') as file:
        json.dump(granular, file, indent=4)
