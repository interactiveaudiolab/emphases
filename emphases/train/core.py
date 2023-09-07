import contextlib
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
    return emphases.checkpoint.best_path(output_directory)[0]


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

    # Training data
    train_loader = emphases.data.loader(dataset, 'train', gpu)

    # Validation data
    if emphases.VALIDATION_DATASET == 'buckeye':

        # This is just for generating scaling law plots for the paper
        valid_loader = emphases.data.loader('buckeye', 'test', gpu)

    else:

        valid_loader = emphases.data.loader(dataset, 'valid', gpu)

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
        model, optimizer, epoch, step, score, best = emphases.checkpoint.load(
            path,
            model,
            optimizer)

    else:

        # Train from scratch
        epoch, step, score, best = 0, 0, 0., 0.

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
                    score = evaluate(
                        log_directory,
                        step,
                        model,
                        gpu,
                        'valid',
                        valid_loader)

                ###################
                # Save checkpoint #
                ###################

                if step >= 300 and score > best:
                    emphases.checkpoint.save(
                        model,
                        optimizer,
                        epoch,
                        step,
                        score,
                        best,
                        output_directory / f'{step:08d}.pt')
                    best = score

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
            score,
            best,
            output_directory / f'{step:08d}.pt')


###############################################################################
# Evaluation
###############################################################################


def evaluate(directory, step, model, gpu, condition, loader):
    """Perform model evaluation"""
    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    # Tensorboard audio and figures
    waveforms, figures = {}, {}

    # Prepare model for inference
    with emphases.inference_context(model):

        # Cache results to evaluate
        results = []
        for i, batch in enumerate(loader):

            # Unpack batch
            (
                features,
                frame_lengths,
                word_bounds,
                word_lengths,
                targets,
                alignments,
                audio,
                stems
            ) = batch

            # Copy to GPU
            features = features.to(device)
            frame_lengths = frame_lengths.to(device)
            word_bounds = word_bounds.to(device)
            word_lengths = word_lengths.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(
                features,
                frame_lengths,
                word_bounds,
                word_lengths)

            # Cache results
            results.append((
                logits.detach().cpu(),
                targets.detach().cpu(),
                word_lengths.detach().cpu()))

            # Add audio and figures
            if condition == 'valid' and i < emphases.PLOT_EXAMPLES:

                # Postprocess network output
                scores = emphases.postprocess(logits)

                # Add audio
                samples = emphases.convert.frames_to_samples(frame_lengths[0])
                waveforms[f'audio/{stems[0]}'] = audio[0, :, :samples]

                # Add figure
                figures[stems[0]] = emphases.plot.scores(
                    alignments[0],
                    scores[0, 0, :word_lengths[0]].cpu(),
                    targets[0, 0, :word_lengths[0]].cpu())

            # Stop when we exceed some number of batches
            if i + 1 == emphases.LOG_STEPS:
                break

        # Setup batch statistics
        target_stats = emphases.evaluate.metrics.Statistics()
        predicted_stats = emphases.evaluate.metrics.Statistics()

        # Update statistics
        for logits, targets, word_lengths in results:
            target_stats.update(
                targets.to(device),
                word_lengths.to(device))
            predicted_stats.update(
                emphases.postprocess(logits.to(device)),
                word_lengths.to(device))

        # Setup evaluation metrics
        metrics = emphases.evaluate.Metrics(predicted_stats, target_stats)

        # Update metrics
        for logits, targets, word_lengths in results:
            metrics.update(
                logits.to(device),
                targets.to(device),
                word_lengths.to(device))

    # Format results
    scalars = {
        f'{key}/{condition}': value for key, value in metrics().items()}

    # Write to tensorboard
    emphases.write.scalars(directory, step, scalars)
    emphases.write.figures(directory, step, figures)
    emphases.write.audio(directory, step, waveforms)

    # Return Pearson correlation
    return scalars[f'pearson_correlation/{condition}']


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
    if training and emphases.DOWNSAMPLE_LOCATION == 'inference':

        # If we are not downsampling the network output before the loss, we
        # must upsample the targets
        targets = emphases.upsample(
            targets,
            word_bounds,
            word_lengths,
            frame_lengths)

        # Linear interpolation can cause out-of-range
        if emphases.UPSAMPLE_METHOD == 'linear':
            targets = torch.clamp(targets, min=0., max=1.)

        # Frame resolution sequence mask
        mask = emphases.model.mask_from_lengths(frame_lengths)

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
