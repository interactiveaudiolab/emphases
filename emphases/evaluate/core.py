import json

import torch

import emphases


###############################################################################
# Evaluate
###############################################################################


def datasets(datasets, checkpoint=emphases.DEFAULT_CHECKPOINT, gpu=None):
    """Perform evaluation"""
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
            emphases.data.loader([dataset], 'test'),
            f'Evaluating {emphases.CONFIG} on {dataset}')

        # Iterate over test set
        for alignment, _, _, mels, targets, _, stem in iterator:

            # Reset file metrics
            file_metrics.reset()

            # Get predicted scores
            scores = emphases.from_alignment_and_audio(
                alignment,
                mels,
                emphases.SAMPLE_RATE,
                checkpoint=checkpoint,
                batch_size=emphases.EVALUATION_BATCH_SIZE,
                pad=True,
                gpu=gpu)

            # Update metrics
            lengths = torch.tensor(
                len(scores),
                dtype=torch.long,
                device=scores.device)
            file_metrics.update(scores, targets, lengths)
            dataset_metrics.update(scores, targets, lengths)
            aggregate_metrics.update(scores, targets, lengths)

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


# TODO - can we only eval on some subset of the data? if so, can we fix that?
# def eval_similarity(prom_file, ground_truth_file):
#     annotations = pd.read_csv(ground_truth_file)
#     filename = prom_file.split('/')[-1].replace('.prom', '')
#     print(f">>> Processing {filename}")
#     predictions = pd.read_table(prom_file, header=None)
#     avail_window = annotations[annotations['filename']
#                                == filename].reset_index(drop=True)
#     target_window = predictions.reset_index(drop=True)

#     l, r = 0, 0
#     pairs = []

#     while l < len(target_window) and r < len(avail_window):
#         if target_window[3][l] == avail_window['word'][r]:
#             pairs.append([(target_window[3][l], target_window[4][l]),
#                          (avail_window['word'][r], avail_window['pa.32'][r])])
#             l += 1
#             r += 1
#         else:
#             l += 1

#     v1 = []
#     v2 = []

#     t1 = []
#     t2 = []

#     for pair in pairs:
#         v1.append(pair[0][-1])
#         v2.append(pair[1][-1])

#         t1.append(pair[0][0])
#         t2.append(pair[1][0])

#     v1 = np.asarray(v1)
#     v2 = np.asarray(v2)

#     assert t1 == t2
#     assert v1.shape[0] == v2.shape[0]

#     print(
#         f"{v1.shape[0]} tokens compared, {len(avail_window)} tokens were available in annotation")
#     if len(v1) > 0 and len(v2) > 0:
#         result = 1 - scipy.spatial.distance.cosine(v1, v2)
#     else:
#         result = None

#     return result
