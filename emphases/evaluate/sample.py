def evaluate(directory, step, model, gpu, condition, loader):
    """Perform model evaluation"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Setup evaluation metrics
    metrics = penne.evaluate.Metrics()

    # Prepare model for inference
    with penne.inference_context(model, device.type) as model:

        # Unpack batch
        for i, (audio, bins, pitch, voiced, *_) in enumerate(loader):

            # Forward pass
            logits = model(audio.to(device))

            # Update metrics
            metrics.update(
                logits,
                bins.to(device),
                pitch.to(device),
                voiced.to(device))

            # Stop when we exceed some number of batches
            if i + 1 == penne.LOG_STEPS:
                break

    # Format results
    scalars = {
        f'{key}/{condition}': value for key, value in metrics().items()}

    # Write to tensorboard
    penne.write.scalars(directory, step, scalars)

    return scalars[f'accuracy/{condition}']

# # Evaluate
# if step % penne.LOG_INTERVAL == 0:
#     evaluate_fn = functools.partial(
#         evaluate,
#         log_directory,
#         step,
#         model,
#         gpu)
#     evaluate_fn('train', train_loader)
#     valid_accuracy = evaluate_fn('valid', valid_loader)


##########################################
# AMP 
##########################################

@contextlib.contextmanager
def inference_context(model, device_type):
    # Prepare model for evaluation
    model.eval()

    # Turn off gradient computation
    with torch.no_grad():

        # Automatic mixed precision on GPU
        if device_type == 'cuda':
            with torch.autocast(device_type):
                yield model

        else:
            yield model

    # Prepare model for training
    model.train()