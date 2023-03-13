import emphases

import torch

def frames_to_words(input, dim=None):
    method = emphases.FRAMES_TO_WORDS_RESAMPLE
    if len(input) == 0: return 0
    if method == 'max':
        if dim is not None:
            max_out = input.max(dim=dim)
            return max_out.values
        return input.max()
    elif method == 'avg':
        return input.mean(dim=dim)
    elif method == 'center':
        if dim is None:
            return input[len(input) // 2]
        else:
            center_index = torch.Tensor([input.shape[dim] // 2]).int().to(device=input.device)
            return torch.index_select(input, dim, center_index).squeeze()
    else:
        raise ValueError(f'Interpolation method {method} is not defined')