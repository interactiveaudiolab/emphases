from .convolution import Convolution
from .transformer import Transformer

import emphases


def Layers(**kwargs):
    if emphases.ARCHITECTURE == 'convolution':
        return Convolution(**kwargs)
    elif emphases.ARCHITECTURE == 'transformer':
        return Transformer()
    else:
        raise ValueError(
            f'Network layer {emphases.ARCHITECTURE} is not defined')
