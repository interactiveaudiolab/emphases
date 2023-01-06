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
        for _, targets, _, _, _, alignments, audio, stem in iterator:

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

            # Update metrics
            lengths = torch.tensor(
                len(scores),
                dtype=torch.long,
                device=device)
            file_metrics.update(scores, targets.to(device), lengths)
            dataset_metrics.update(scores, targets.to(device), lengths)
            aggregate_metrics.update(scores, targets.to(device), lengths)

            # Copy results
            granular[f'{dataset}/{stem[0]}'] = file_metrics()
        overall[dataset] = dataset_metrics()
    overall['aggregate'] = aggregate_metrics()

    # Write to json files
    directory = emphases.EVAL_DIR / emphases.CONFIG
    directory.mkdir(exist_ok=True, parents=True)
    with open(directory / 'overall.json', 'w') as file:
        json.dump(overall, file, indent=4)
    with open(directory / 'granular.json', 'w') as file:
        json.dump(granular, file, indent=4)
