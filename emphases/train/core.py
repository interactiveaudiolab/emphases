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
    gpus=None,
    task_type='word_wise'):
    """Run model training"""
    # Distributed data parallelism
    if gpus and len(gpus) > 1:
        args = (
            dataset,
            checkpoint_directory,
            output_directory,
            log_directory,
            gpus,
            task_type)
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
            None if gpus is None else gpus[0],
            task_type)

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
    gpu=None,
    task_type='word_wise'):
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

    # model = emphases.model.Model().to(device)
    if task_type=='word_wise':
        model = emphases.model.BaselineModel(device=device).to(device)
    if task_type=='frame_wise':
        model = emphases.model.FramewiseModel().to(device)

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

    #####################
    # Create schedulers #
    #####################

    scheduler_fn = functools.partial(
        torch.optim.lr_scheduler.ExponentialLR,
        gamma=emphases.LEARNING_RATE_DECAY,
        last_epoch=step // len(train_loader.dataset) if step else -1)
    scheduler = scheduler_fn(optimizer)

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

        model.train()
        for batch in train_loader:

            (
            padded_audio,
            padded_mel_spectrogram,
            padded_prominence,
            padded_interpolated_prominence,
            word_bounds,
            word_lengths,
            frame_lengths,
            interpolated_prom_lengths
            ) = (item.to(device) if torch.is_tensor(item) else item for item in batch)

            if task_type=='word_wise':
                # Bundle training input
                model_input = (padded_mel_spectrogram, word_bounds, padded_prominence)
            
            if task_type=='frame_wise':
                model_input = padded_mel_spectrogram

            with torch.cuda.amp.autocast():

                # Forward pass
                # print("forward pass", model_input[0].shape, model_input[0].device, device)

                (
                    outputs
                ) = model(model_input)

                # compute losses
                loss_fn =  torch.nn.MSELoss()
                outputs = outputs.to(device)

                if task_type=='word_wise':
                    losses = 0
                    for idx in range(len(outputs)):
                        # masking the loss
                        losses += loss_fn(
                            outputs.squeeze()[idx, 0:word_lengths[idx]], 
                            padded_prominence.squeeze()[idx, 0:word_lengths[idx]]
                            )

                if task_type=='frame_wise':
                    losses = 0
                    for idx in range(len(outputs)):
                        # masking the loss
                        losses += loss_fn(
                            outputs.squeeze()[idx, 0:interpolated_prom_lengths[idx]], 
                            padded_interpolated_prominence.squeeze()[idx, 0:interpolated_prom_lengths[idx]]
                            )

                print("training loss:", losses)

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

                if step % emphases.LOG_INTERVAL == 0:

                    # Log losses
                    scalars = {
                        'loss/total': losses,
                        'learning_rate': optimizer.param_groups[0]['lr']}
                    emphases.write.scalars(log_directory, step, scalars)

                ############
                # Evaluate #
                ############

                if step % emphases.EVALUATION_INTERVAL == 0:
                    print('>>>>> Calling evaluation')
                    evaluate(
                        log_directory,
                        step,
                        model,
                        valid_loader,
                        gpu,
                        task_type)

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

        # Update learning rate every epoch
        scheduler.step()

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


def evaluate(directory, step, model, valid_loader, gpu, task_type):
    """Perform model evaluation"""
    # eval batchsize is set to 1

    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    # Prepare model for evaluation
    model.eval()
    loss_fn =  torch.nn.MSELoss()

    # Turn off gradient computation
    with torch.no_grad():
        for batch in valid_loader:
            (
            padded_audio,
            padded_mel_spectrogram,
            padded_prominence,
            padded_interpolated_prominence,
            word_bounds,
            word_lengths,
            frame_lengths,
            interpolated_prom_lengths
            ) = (item.to(device) if torch.is_tensor(item) else item for item in batch)

            cosine = torch.nn.CosineSimilarity()

            if task_type=='word_wise':
                # Bundle training input
                model_input = (padded_mel_spectrogram, word_bounds, padded_prominence)

                valid_outputs = model(model_input)
                valid_outputs = valid_outputs.to(device)

                val_sim = []
                for idx in range(len(valid_outputs)):
                    val_sim.append(
                        cosine(
                        valid_outputs.squeeze()[idx, 0:word_lengths[idx]].reshape(1, -1), 
                        padded_prominence.squeeze()[idx, 0:word_lengths[idx]].reshape(1, -1)
                        )
                        )
                
                losses = 0
                for idx in range(len(valid_outputs)):
                    # masking the loss
                    losses += loss_fn(
                        valid_outputs.squeeze()[idx, 0:word_lengths[idx]], 
                        padded_prominence.squeeze()[idx, 0:word_lengths[idx]]
                        )

            
            if task_type=='frame_wise':
                # Bundle training input
                model_input = padded_mel_spectrogram

                valid_outputs = model(model_input)
                valid_outputs = valid_outputs.to(device)

                val_sim = []
                for idx in range(len(valid_outputs)):
                    val_sim.append(
                        cosine(
                        valid_outputs.squeeze()[idx, 0:interpolated_prom_lengths[idx]].reshape(1, -1), 
                        padded_interpolated_prominence.squeeze()[idx, 0:interpolated_prom_lengths[idx]].reshape(1, -1)
                        )
                        )
                
                losses = 0
                for idx in range(len(valid_outputs)):
                    # masking the loss
                    losses += loss_fn(
                        valid_outputs.squeeze()[idx, 0:interpolated_prom_lengths[idx]], 
                        padded_interpolated_prominence.squeeze()[idx, 0:interpolated_prom_lengths[idx]]
                        )

            print('>>>>> cosine similarity values for validation set:', 
                    val_sim, '\n>>>>> mean cosine similarity:', torch.tensor(val_sim).mean())

            ######################
            # Logging Evaluation #
            ######################
            # Log losses
            scalars_eval = {
                'eval_loss/total': losses,
                'mean_cosine_score': torch.tensor(val_sim).mean()}

            emphases.write.scalars(directory, step, scalars_eval)

    # Prepare model for training
    model.train()


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
