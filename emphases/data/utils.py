import torch
import numpy as np
import emphases

def nearest_neighbour_interpolation(audio, word_bounds, prominence):
    """
    Performs nearest neighbour interpolation to estimate prominence values for every frame
    """
    wb_prom_pairs = []
    audio_len = audio.shape[-1]

    if word_bounds[0][0]!=0:
        wb_prom_pairs.append([(0, word_bounds[0][0]), 0])

    for idx in range(len(word_bounds)):
        wb_prom_pairs.append([word_bounds[idx], prominence[idx].item()])
        if idx+1<len(word_bounds):
            if word_bounds[idx][-1]!=word_bounds[idx+1][0]:
                start = word_bounds[idx][-1]
                end = word_bounds[idx+1][0]
                wb_prom_pairs.append([(start, end), 0])
                
    prom_extended = []

    for wb in wb_prom_pairs:
        start, end = wb[0][0], wb[0][1]
        prom_extended.extend([wb[-1]]*(end-start))

    if word_bounds[-1][-1]!=(audio_len//emphases.HOPSIZE + 1):
        pad_len = (audio_len//emphases.HOPSIZE + 1) - len(prom_extended)
        prom_extended.extend([0]*pad_len)

    prom_extended = torch.tensor(prom_extended)
    
    return prom_extended

def constant(tensor, ratio):
    """Create a grid for constant-ratio time-stretching"""
    return torch.linspace(
        0.,
        tensor.shape[-1] - 1,
        # round((tensor.shape[-1]) / ratio + 1e-4),
        round((tensor.shape[-1]) / (ratio + 0.4)),
        dtype=torch.float,
        device=tensor.device)

def grid_sample(sequence, grid, method='linear'):
    """Perform 1D grid-based sampling"""
    # Require interpolation method to be defined
    if method not in ['linear', 'nearest']:
        raise ValueError(
            f'Interpolation mode {emphases.PPG_INTERP_METHOD} is not defined')

    # Setup grid parameters
    x = grid
    fp = sequence

    # Linear grid interpolation
    if method == 'linear':
        xp = torch.arange(sequence.shape[-1], device=sequence.device)
        i = torch.clip(torch.searchsorted(xp, x, right=True), 1, len(xp) - 1)
        return (
            (fp[..., i - 1] * (xp[i] - x) + fp[..., i] * (x - xp[i - 1])) /
            (xp[i] - xp[i - 1]))

    # Nearest neighbors grid interpolation
    elif method == 'nearest':
        return fp[..., torch.round(x).to(torch.long)]

    else:
        raise ValueError(f'Grid sampling method {method} is not defined')

def interpolate_numpy(sequence, grid):
    xp = torch.arange(sequence.shape[-1], device=sequence.device)
    fp = sequence
    return torch.tensor(np.interp(grid, xp, fp))