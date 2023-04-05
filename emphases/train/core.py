import contextlib
import functools
import os
import json

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
        model, optimizer, step = emphases.checkpoint.load(
            path,
            model,
            optimizer)

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

    ##################################################
    # Save target statistics #
    ##################################################
    
    emphases.dataset = dataset
    init_target_stats('train', dataset)
    init_target_stats('valid', dataset)

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
        train_loader.batch_sampler.set_epoch(step // len(train_loader.dataset))

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
                scores, mask = model(
                    features,
                    frame_lengths,
                    word_bounds,
                    word_lengths)

                # Compute loss
                train_loss = loss(scores, targets, word_bounds, mask)
                
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

                if step % emphases.LOG_INTERVAL == 0:
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

    # Load target and prediction stats
    stats = load_target_stats(condition, 'annotate')

    # Setup evaluation metrics
    metrics = emphases.evaluate.Metrics(stats)

    prediction_collection = torch.tensor([]).to(device)

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
                _,           # alignment
                _,           # audio
                _            # stems
            ) = batch

            # Copy to GPU
            features = features.to(device)
            frame_lengths = frame_lengths.to(device)
            word_bounds = word_bounds.to(device)
            word_lengths = word_lengths.to(device)
            targets = targets.to(device)
            
            # Forward pass
            scores, mask = model(
                features,
                frame_lengths,
                word_bounds,
                word_lengths)

            # Downsample to word resolution for evaluation
            if emphases.DOWNSAMPLE_LOCATION in ['loss', 'inference']:
                scores = emphases.downsample(scores, word_lengths, word_bounds)

            prediction_collection = torch.cat((prediction_collection, scores.flatten()))
            # TODO - update the mean and std without the collection method
            
            # Update metrics
            metrics.update(scores, targets, word_bounds, mask)

            # Stop when we exceed some number of batches
            if i + 1 == emphases.LOG_STEPS:
                break

    # Format results
    scalars = {
        f'{key}/{condition}': value for key, value in metrics().items()}

    # Write to tensorboard
    emphases.write.scalars(directory, step, scalars)

    # Update prediction stats
    stats['prediction_mean'] = torch.mean(prediction_collection).item()
    stats['prediction_std'] = torch.std(prediction_collection).item()

    stats_file = emphases.ASSETS_DIR / 'statistics' / f'{emphases.dataset}_{condition}_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=True)

###############################################################################
# Partition specific statistics
###############################################################################

def init_target_stats(partition, dataset):
    
    partition_data = json.loads(open(emphases.PARTITION_DIR / f'{dataset}.json', 'r').read())
    targets_dir = emphases.CACHE_DIR / dataset / 'scores'
    partition_files = [file for file in targets_dir.glob('*.pt') if file.stem in partition_data[partition]]

    scores = torch.tensor([])
    for file in partition_files:
        scores = torch.cat((torch.load(file), scores))
        
    stats = {
        "target_mean": torch.mean(scores).item(),
        "target_std": torch.std(scores).item(),
        "prediction_mean": torch.mean(scores).item(),
        "prediction_std": torch.std(scores).item()
        }
    
    # Save the statistics
    (emphases.ASSETS_DIR / 'statistics').mkdir(exist_ok=True)
    stats_file = emphases.ASSETS_DIR / 'statistics' / f'{dataset}_{partition}_stats.json'

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=True)

def load_target_stats(partition, dataset):    
    stats_file = emphases.ASSETS_DIR / 'statistics' / f'{dataset}_{partition}_stats.json'
    return json.loads(open(stats_file, 'r').read())

###############################################################################
# Loss function
###############################################################################


def loss(scores, targets, word_bounds, mask):
    """Compute masked loss"""
    # If we are not downsampling the network output before the loss, we must
    # upsample the targets
    if emphases.DOWNSAMPLE_LOCATION == 'inference':

        # Interpolate
        targets = emphases.upsample(
            targets,
            scores.shape[-1],
            word_bounds)

        # Linear interpolation can cause out-of-range
        if emphases.UPSAMPLE_METHOD == 'linear':
            targets = torch.clamp(targets, min=0., max=1.)

    return emphases.LOSS(scores * mask, targets * mask)


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
