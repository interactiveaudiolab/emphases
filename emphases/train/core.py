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
    train_loader = emphases.data.loader(
        dataset,
        'train',
        gpu,
        train_limit=emphases.TRAIN_DATA_LIMIT)
    # TEMPORARY - use buckeye for validation
    # valid_loader = emphases.data.loader(dataset, 'valid', gpu)
    valid_loader = emphases.data.loader('buckeye', 'test', gpu)

    ################
    # Create model #
    ################

    model = emphases.Model().to(device)

    ####################
    # Create optimizer #
    ####################

    optimizer = torch.optim.Adam(model.parameters())

    ##############################
    # Maybe load from checkpoint #
    ##############################

    path = emphases.checkpoint.latest_path(checkpoint_directory)

    if path is not None:

        # Load model
        model, optimizer, epoch, step = emphases.checkpoint.load(
            path,
            model,
            optimizer)

    else:

        # Train from scratch
        epoch, step = 0, 0

    ##################################################
    # Maybe setup distributed data parallelism (DDP) #
    ##################################################

    if rank is not None:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank])

    ########################################
    # Get target statistics for evaluation #
    ########################################

    if not rank:
        train_stats = emphases.evaluate.metrics.Statistics()
        valid_stats = emphases.evaluate.metrics.Statistics()
        for batch in train_loader:
            train_stats.update(batch[4])
        for batch in valid_loader:
            valid_stats.update(batch[4])

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
        train_loader.batch_sampler.set_epoch(epoch)

        for batch in train_loader:

            # Unpack batch
            (
                features,
                frame_lengths,
                word_bounds,
                word_lengths,
                targets,
                _,           # alignment
                _,           # audio
                _            # stem
            ) = batch

            # Copy to GPU
            features = features.to(device)
            frame_lengths = frame_lengths.to(device)
            word_bounds = word_bounds.to(device)
            word_lengths = word_lengths.to(device)
            targets = targets.to(device)
            with torch.autocast(device.type):

                # Forward pass
                scores = model(
                    features,
                    frame_lengths,
                    word_bounds,
                    word_lengths)

                # Compute loss
                train_loss = loss(
                    scores,
                    targets,
                    frame_lengths,
                    word_bounds,
                    word_lengths,
                    training=True)

            ##################
            # Optimize model #
            ##################

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

                if step % emphases.LOG_INTERVAL == 0:
                    evaluate_fn = functools.partial(
                        evaluate,
                        log_directory,
                        step,
                        model,
                        gpu)
                    evaluate_fn('train', train_loader, train_stats)
                    evaluate_fn('valid', valid_loader, valid_stats)

                ###################
                # Save checkpoint #
                ###################

                if step and step % emphases.CHECKPOINT_INTERVAL == 0:
                    emphases.checkpoint.save(
                        model,
                        optimizer,
                        epoch,
                        step,
                        output_directory / f'{step:08d}.pt')

            # End training after a certain number of steps
            if step >= emphases.NUM_STEPS:
                break

            if not rank:

                # Update training step count
                step += 1

                # Update progress bar
                progress.update()

        # Update epoch count
        if not rank:
            epoch += 1

    if not rank:

        # Close progress bar
        progress.close()

        # Save final model
        emphases.checkpoint.save(
            model,
            optimizer,
            epoch,
            step,
            output_directory / f'{step:08d}.pt')


###############################################################################
# Evaluation
###############################################################################


def evaluate(directory, step, model, gpu, condition, loader, stats):
    """Perform model evaluation"""
    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    # Get mean and variance for Pearson Correlation
    target_stats = emphases.evaluate.metrics.Statistics()
    predicted_stats = emphases.evaluate.metrics.Statistics()
    for i, batch in enumerate(loader):

            # Unpack batch
            (
                features,
                frame_lengths,
                word_bounds,
                word_lengths,
                targets,
                _,
                _,
                _
            ) = batch

            # Copy to GPU
            features = features.to(device)
            frame_lengths = frame_lengths.to(device)
            word_bounds = word_bounds.to(device)
            word_lengths = word_lengths.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(features, frame_lengths, word_bounds, word_lengths)

            # Update statistics
            target_stats.update(targets)
            predicted_stats.update(emphases.postprocess(logits))

            # Stop when we exceed some number of batches
            if i + 1 == emphases.LOG_STEPS:
                break

    # Setup evaluation metrics
    metrics = emphases.evaluate.Metrics(predicted_stats, target_stats)

    # Tensorboard audio and figures
    waveforms, figures = {}, {}

    # Prepare model for inference
    with emphases.inference_context(model):

        for i, batch in enumerate(loader):

            # Unpack batch
            (
                features,
                frame_lengths,
                word_bounds,
                word_lengths,
                targets,
                alignments,
                audios,
                stems
            ) = batch

            # Copy to GPU
            features = features.to(device)
            frame_lengths = frame_lengths.to(device)
            word_bounds = word_bounds.to(device)
            word_lengths = word_lengths.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(features, frame_lengths, word_bounds, word_lengths)

            # Update metrics
            metrics.update(
                logits,
                targets,
                frame_lengths,
                word_bounds,
                word_lengths)

            # Add audio and figures
            if i == 0 and condition == 'valid':

                # Postprocess network output
                scores = emphases.postprocess(logits)

                iterator = zip(
                    scores[:emphases.PLOT_EXAMPLES].cpu(),
                    targets[:emphases.PLOT_EXAMPLES].cpu(),
                    frame_lengths[:emphases.PLOT_EXAMPLES],
                    word_lengths[:emphases.PLOT_EXAMPLES],
                    alignments[:emphases.PLOT_EXAMPLES],
                    audios[:emphases.PLOT_EXAMPLES],
                    stems[:emphases.PLOT_EXAMPLES])
                for (
                    score,
                    target,
                    frame_length,
                    word_length,
                    alignment,
                    audio,
                    stem
                ) in iterator:

                    # Add audio
                    samples = emphases.convert.frames_to_samples(frame_length)
                    waveforms[f'audio/{stem}'] = audio[:, :samples]

                    # Add figure
                    figures[stem] = emphases.plot.scores(
                        alignment,
                        score[0, :word_length],
                        target[0, :word_length])

            # Stop when we exceed some number of batches
            if i + 1 == emphases.LOG_STEPS:
                break

    # Format results
    scalars = {
        f'{key}/{condition}': value for key, value in metrics().items()}

    # Write to tensorboard
    emphases.write.scalars(directory, step, scalars)
    emphases.write.figures(directory, step, figures)
    emphases.write.audio(directory, step, waveforms)


###############################################################################
# Loss function
###############################################################################


def loss(
    scores,
    targets,
    frame_lengths,
    word_bounds,
    word_lengths,
    training=False,
    loss_fn=emphases.LOSS):
    """Compute masked loss"""
    if emphases.DOWNSAMPLE_LOCATION == 'inference':

        if training:

            # If we are not downsampling the network output before the loss, we
            # must upsample the targets
            try:
                targets = emphases.upsample(
                    targets,
                    word_bounds,
                    word_lengths,
                    frame_lengths)
            except IndexError as error:
                print(error)
                import pdb; pdb.set_trace()
                pass

            # Linear interpolation can cause out-of-range
            if emphases.UPSAMPLE_METHOD == 'linear':
                targets = torch.clamp(targets, min=0., max=1.)

            # Frame resolution sequence mask
            mask = emphases.model.mask_from_lengths(frame_lengths)

        else:

            # Word resolution sequence mask
            mask = emphases.model.mask_from_lengths(word_lengths)

    else:

        # Word resolution sequence mask
        mask = emphases.model.mask_from_lengths(word_lengths)

    # Compute masked loss
    if loss_fn == 'bce':
        return torch.nn.functional.binary_cross_entropy_with_logits(
            scores[mask],
            targets[mask])
    elif loss_fn == 'mse':
        return torch.nn.functional.mse_loss(scores[mask], targets[mask])
    raise ValueError(f'Loss {loss_fn} is not recognized')


###############################################################################
# Distributed data parallelism
###############################################################################


def train_ddp(
    rank,
    dataset,
    checkpoint_directory,
    output_directory,
    log_directory,
    gpus):
    """Train with distributed data parallelism"""
    with ddp_context(rank, len(gpus)):
        train(
            dataset,
            checkpoint_directory,
            output_directory,
            log_directory,
            gpus[rank])


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
