import torch


###############################################################################
# Checkpoint utilities
###############################################################################


def best_path(directory, regex='*.pt'):
    """Retrieve the path to the best checkpoint"""
    # Retrieve checkpoint filenames
    files = list(directory.glob(regex))

    # If no matching checkpoint files, no training has occurred
    if not files:
        return

    # Retrieve best checkpoint
    best_file, best_score = None, 0.
    for file in files:
        score = torch.load(file, map_location='cpu')['score']
        if score > best_score:
            best_score = score
            best_file = file

    return best_file, best_score


def latest_path(directory, regex='*.pt'):
    """Retrieve the path to the most recent checkpoint"""
    # Retrieve checkpoint filenames
    files = list(directory.glob(regex))

    # If no matching checkpoint files, no training has occurred
    if not files:
        return

    # Retrieve latest checkpoint
    files.sort(key=lambda file: int(''.join(filter(str.isdigit, file.stem))))
    return files[-1]


def load(checkpoint_path, model, optimizer=None):
    """Load model checkpoint from file"""
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    # Restore model
    model.load_state_dict(checkpoint_dict['model'])

    # Restore optimizer
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])

    # Restore training state
    epoch = checkpoint_dict['epoch']
    step = checkpoint_dict['step']
    score = checkpoint_dict['score']
    best = checkpoint_dict['best']

    return model, optimizer, epoch, step, score, best


def save(model, optimizer, epoch, step, score, best, file):
    """Save training checkpoint to disk"""
    # Maybe unpack DDP
    if torch.distributed.is_initialized():
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    # Save
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'score': score,
        'best': best,
        'model': model_state_dict,
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, file)
