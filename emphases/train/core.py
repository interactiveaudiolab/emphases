import contextlib
import functools
import os

import torch
import tqdm

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
    train_loader, valid_loader = emphases.data.loaders(dataset, 'train', 'valid', gpu)

    #################
    # Create models #
    #################

    if emphases.METHOD == 'wordwise':
        model = emphases.model.BaselineModel(device=device).to(device)
    elif emphases.METHOD == 'framewise':
        model = emphases.model.FramewiseModel().to(device)
    else:
        raise ValueError(f'Method {emphases.METHOD} is not defined')

    ##################################################
    # Maybe setup distributed data parallelism (DDP) #
    ##################################################

    if rank is not None:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank])

    ####################
    # Create optimizer #
    ####################

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=[.80, .99],
        eps=1e-9)

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

    #########
    # Train #
    #########

    # Automatic mixed precision (amp) gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    # Get total number of steps
    steps = emphases.NUM_STEPS

    # Setup progress bar
    if not rank:
        progress = tqdm.tqdm(
            initial=step,
            total=steps,
            dynamic_ncols=True,
            desc=f'Training {emphases.CONFIG}')
    while step < steps:

        # Seed sampler
        train_loader.batch_sampler.set_epoch(step // len(train_loader.dataset))

        for batch in train_loader:

            (
            audio,
            features,
            prominence,
            interpolated_prominence,
            word_bounds,
            word_lengths,
            frame_lengths,
            interpolated_prom_lengths
            ) = (item.to(device) if torch.is_tensor(item) else item for item in batch)

            with torch.cuda.amp.autocast():

                # Forward pass
                scores = model(features, word_bounds)

                # compute losses
                loss_fn =  torch.nn.MSELoss()

                if emphases.METHOD == 'wordwise':

                    # TODO - no for loop
                    losses = 0
                    for idx in range(len(scores)):
                        # masking the loss
                        losses += loss_fn(
                            scores.squeeze()[idx, :word_lengths[idx]],
                            prominence.squeeze()[idx, :word_lengths[idx]]
                            )

                elif emphases.METHOD == 'framewise':

                    # TODO - no for loop
                    losses = 0
                    for idx in range(len(scores)):
                        # masking the loss
                        losses += loss_fn(
                            scores.squeeze()[idx, :interpolated_prom_lengths[idx]],
                            interpolated_prominence.squeeze()[idx, :interpolated_prom_lengths[idx]]
                            )

            ######################
            # Optimize model #
            ######################

            optimizer.zero_grad()

            # Backward pass
            scaler.scale(losses).backward()

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
            if step >= steps:
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

        for i, batch in enumerate(loader):

            # Unpack batch
            (
                audio,
                features,
                prominence,
                interpolated_prominence,
                word_bounds,
                word_lengths,
                frame_lengths,
                interpolated_prom_lengths
            ) = (item.to(device) if torch.is_tensor(item) else item for item in batch)

            # Forward pass
            scores = model(features, word_bounds)

            if emphases.METHOD == 'wordwise':
                val_sim = []
                for idx in range(len(scores)):
                    val_sim.append(
                        cosine(
                        scores.squeeze()[idx, 0:word_lengths[idx]].reshape(1, -1),
                        prominence.squeeze()[idx, 0:word_lengths[idx]].reshape(1, -1)
                        )
                        )

                losses = 0
                for idx in range(len(scores)):
                    # masking the loss
                    losses += loss_fn(
                        scores.squeeze()[idx, 0:word_lengths[idx]],
                        prominence.squeeze()[idx, 0:word_lengths[idx]]
                        )

            elif emphases.METHOD == 'framewise':
                val_sim = []
                for idx in range(len(scores)):
                    val_sim.append(
                        cosine(
                        scores.squeeze()[idx, 0:interpolated_prom_lengths[idx]].reshape(1, -1),
                        interpolated_prominence.squeeze()[idx, 0:interpolated_prom_lengths[idx]].reshape(1, -1)
                        )
                        )

                losses = 0
                for idx in range(len(scores)):
                    # masking the loss
                    losses += loss_fn(
                        scores.squeeze()[idx, 0:interpolated_prom_lengths[idx]],
                        interpolated_prominence.squeeze()[idx, 0:interpolated_prom_lengths[idx]]
                        )

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
    # TODO
    pass
