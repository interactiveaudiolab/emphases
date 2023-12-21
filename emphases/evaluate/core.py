import json

import torch
import torchutil

import emphases


###############################################################################
# Evaluate
###############################################################################


@torchutil.notify('evaluate')
def datasets(datasets, checkpoint=None, gpu=None):
    """Perform evaluation"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Containers for results
    overall, granular = {}, {}

    # Evaluate each dataset
    for dataset in datasets:

        # Get data loader
        loader = emphases.data.loader(dataset, 'test', gpu)

        # Get mean and variance for Pearson Correlation
        target_stats = emphases.evaluate.metrics.Statistics()
        predicted_stats = emphases.evaluate.metrics.Statistics()
        for batch in loader:

            # Unpack
            _, _, _, word_lengths, targets, alignments, audio, _ = batch

            # Get predicted scores
            scores = emphases.from_alignment_and_audio(
                alignments[0],
                audio[0],
                emphases.SAMPLE_RATE,
                checkpoint=checkpoint,
                gpu=gpu)

            # Update statistics
            target_stats.update(targets, word_lengths)
            predicted_stats.update(scores[None], word_lengths)

        # Get metric class
        metric_fn = emphases.evaluate.Metrics

        # Per-file metrics
        file_metrics = metric_fn(predicted_stats, target_stats)

        # Per-dataset metrics
        dataset_metrics = metric_fn(predicted_stats, target_stats)

        # Iterate over test set
        for batch in torchutil.iterator(
            loader,
            f'Evaluating {emphases.CONFIG} on {dataset}',
            total=len(loader)
        ):

            # Unpack
            (
                _,
                frame_lengths,
                word_bounds,
                word_lengths,
                targets,
                alignments,
                audio,
                stems
             ) = batch

            # Reset file metrics
            file_metrics.reset()

            if emphases.METHOD == 'neural':

                # Get predicted scores
                scores = []

                # Preprocess audio
                for features, word_bounds in emphases.preprocess(
                    alignments[0],
                    audio[0],
                    gpu=gpu
                ):

                    # Infer
                    logits = emphases.infer(
                        features,
                        word_bounds,
                        checkpoint).detach()

                    # Skip postprocessing
                    scores.append(logits)

                # Concatenate results
                scores = torch.cat(scores, 2)

            else:

                # Baseline method inference
                scores = emphases.from_alignment_and_audio(
                    alignments[0],
                    audio[0],
                    emphases.SAMPLE_RATE,
                    gpu=gpu)[None]

            # Update metrics
            args = (scores, targets.to(device), word_lengths.to(device))
            file_metrics.update(*args)
            dataset_metrics.update(*args)

            # Copy results
            granular[f'{dataset}/{stems[0]}'] = file_metrics()
        overall[dataset] = dataset_metrics()

    # Write to json files
    directory = emphases.EVAL_DIR / emphases.CONFIG
    directory.mkdir(exist_ok=True, parents=True)
    with open(directory / 'overall.json', 'w') as file:
        json.dump(overall, file, indent=4)
    with open(directory / 'granular.json', 'w') as file:
        json.dump(granular, file, indent=4)
