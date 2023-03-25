import contextlib
import functools
import os

import torch

import emphases


###############################################################################
# Training interface
###############################################################################


def run(
    dataset,
    checkpoint_directory,
    output_directory,
    log_directory,
    gpus=None):
    """Run model training"""
    # Distributed data parallelism
    if gpus and len(gpus) > 1:
        args = (
            dataset,
            checkpoint_directory,
            output_directory,
            log_directory,
            gpus)
        torch.multiprocessing.spawn(
            train_ddp,
            args=args,
            nprocs=len(gpus),
            join=True)

    else:

        # Single GPU or CPU training
        train(
            dataset,
            checkpoint_directory,
            output_directory,
            log_directory,
            None if gpus is None else gpus[0])

    # Return path to model checkpoint
    return emphases.checkpoint.latest_path(output_directory)


###############################################################################
# Training
###############################################################################


def train(
    dataset,
    checkpoint_directory,
    output_directory,
    log_directory,
    gpu=None):
    """Train a model"""
    # Get DDP rank
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = None

    # Get torch device
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(emphases.RANDOM_SEED)
    train_loader = emphases.data.loader(dataset, 'train', gpu, train_limit=emphases.TRAIN_DATA_LIMIT)
    valid_loader = emphases.data.loader(dataset, 'valid', gpu)

    #################
    # Create models #
    #################

    if emphases.METHOD == 'wordwise':
        model = emphases.model.Wordwise().to(device)
    elif emphases.METHOD == 'framewise':
        model = emphases.model.Framewise().to(device)
    else:
        raise ValueError(f'Method {emphases.METHOD} is not defined')

    ####################
    # Create optimizer #
    ####################

    optimizer = torch.optim.Adam(model.parameters())

    ##############################
    # Maybe load from checkpoint #
    ##############################

    path = emphases.checkpoint.latest_path(
        checkpoint_directory,
        '*.pt'),

    # For some reason, returning None from latest_path returns (None,)
    path = None if path == (None,) else path

    if path is not None:

        # Load model
        (
            model,
            optimizer,
            step
        ) = emphases.checkpoint.load(
            path[0],
            model,
            optimizer
        )

    else:

        # Train from scratch
        step = 0

    ##################################################
    # Maybe setup distributed data parallelism (DDP) #
    ##################################################

    if rank is not None:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank])

    #########
    # Train #
    #########

    # Automatic mixed precision (amp) gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    # Setup progress bar
    if not rank:
        progress = emphases.iterator(
            range(step, emphases.NUM_STEPS),
            f'Training {emphases.CONFIG}',
            step,
            emphases.NUM_STEPS)
    while step < emphases.NUM_STEPS:

        # Seed sampler
        epoch = step // len(train_loader.dataset)
        train_loader.batch_sampler.set_epoch(epoch)

        for batch in train_loader:

            # Unpack batch
            (
                features,
                targets,
                word_bounds,
                word_lengths,
                mask,
                _,           # alignment
                _,           # audio
                _            # stem
            ) = batch

            # Copy to GPU
            features = features.to(device)
            targets = targets.to(device)
            word_bounds = word_bounds.to(device)
            word_lengths = word_lengths.to(device)
            mask = mask.to(device)

            with torch.cuda.amp.autocast():

                # Forward pass
                scores = model(features, word_bounds, word_lengths, mask)

                # Compute loss
                if emphases.USE_BCE_LOGITS_LOSS:
                    train_loss = bceLogitsloss(scores, targets, mask)
                else:
                    train_loss = loss(scores, targets, mask)

            ######################
            # Optimize model #
            ######################

            optimizer.zero_grad()

            # Backward pass
            scaler.scale(train_loss).backward()

            # Update weights
            scaler.step(optimizer)

            # Update gradient scaler
            scaler.update()

            ###########
            # Logging #
            ###########

            if not rank:

                ############
                # Evaluate #
                ############

                if step % emphases.EVALUATION_INTERVAL == 0:
                    evaluate_fn = functools.partial(
                        evaluate,
                        log_directory,
                        step,
                        model,
                        gpu)
                    evaluate_fn('train', train_loader)
                    evaluate_fn('valid', valid_loader)

                ###################
                # Save checkpoint #
                ###################

                if step and step % emphases.CHECKPOINT_INTERVAL == 0:
                    emphases.checkpoint.save(
                        model,
                        optimizer,
                        step,
                        output_directory / f'{step:08d}.pt')

            # Update training step count
            if step >= emphases.NUM_STEPS:
                break
            step += 1

            # Update progress bar
            if not rank:
                progress.update()

    # Close progress bar
    if not rank:
        progress.close()

    # Save final model
    emphases.checkpoint.save(
        model,
        optimizer,
        step,
        output_directory / f'{step:08d}.pt')


###############################################################################
# Evaluation
###############################################################################


def evaluate(directory, step, model, gpu, condition, loader):
    """Perform model evaluation"""
    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    # Setup evaluation metrics
    metrics = emphases.evaluate.Metrics()

    # Prepare model for inference
    with emphases.inference_context(model):

        for batch in loader:

            # Unpack batch
            (
                features,
                targets,
                word_bounds,
                word_lengths,
                mask,
                _,           # alignment
                _,           # audio
                _            # stems
            ) = batch

            # Copy to GPU
            features = features.to(device)
            targets = targets.to(device)
            word_bounds = word_bounds.to(device)
            word_lengths = word_lengths.to(device)
            mask = mask.to(device)

            # Forward pass
            scores = model(features, word_bounds, word_lengths, mask)

            if emphases.METHOD == 'framewise' and not emphases.MODEL_TO_WORDS and emphases.FRAMES_TO_WORDS_RESAMPLE is not None:
                # Get center time of each word in frames (we know that the targets are accurate here since they're interpolated from here)
                word_centers = \
                    word_bounds[:, 0] + (word_bounds[:, 1] - word_bounds[:, 0]) // 2
                
                #Allocate tensors for wordwise scores and targets
                word_scores = torch.zeros(word_centers.shape, device=scores.device)
                word_targets = torch.zeros(word_centers.shape, device=scores.device)
                word_masks = torch.zeros(word_centers.shape, device=scores.device)
                
                for stem in range(targets.shape[0]): #Iterate over batch
                    stem_word_centers = word_centers[stem]
                    stem_word_targets = targets.squeeze(1)[stem, stem_word_centers]
                    stem_word_mask = torch.where(stem_word_centers == 0, 0, 1)

                    word_targets[stem] = stem_word_targets
                    word_masks[stem] = stem_word_mask

                    for i, (start, end) in enumerate(word_bounds[stem].T):
                        word_outputs = scores.squeeze(1)[stem, start:end]
                        if word_outputs.shape[0] == 0:
                            continue
                        word_score = emphases.frames_to_words(word_outputs)
                        word_scores[stem, i] = word_score
                
                scores = word_scores
                targets = word_targets
                mask = word_masks

            # Update metrics
            metrics.update(scores, targets, mask)

    # Format results
    scalars = {
        f'{key}/{condition}': value for key, value in metrics().items()}

    # Write to tensorboard
    emphases.write.scalars(directory, step, scalars)


###############################################################################
# Distributed data parallelism
###############################################################################


def train_ddp(rank, dataset, directory, gpus):
    """Train with distributed data parallelism"""
    with ddp_context(rank, len(gpus)):
        train(dataset, directory, gpus)


@contextlib.contextmanager
def ddp_context(rank, world_size):
    """Context manager for distributed data parallelism"""
    # Setup ddp
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='12355'
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank)

    try:

        # Execute user code
        yield

    finally:

        # Close ddp
        torch.distributed.destroy_process_group()


###############################################################################
# Utilities
###############################################################################


def loss(scores, targets, mask):
    """Compute masked loss"""
    return torch.nn.functional.mse_loss(scores * mask, targets * mask)

def bceLogitsloss(scores, targets, mask):
    """Compute masked Binary Cross Entropy between target and input"""
    return torch.nn.functional.binary_cross_entropy_with_logits(scores * mask, targets * mask)
