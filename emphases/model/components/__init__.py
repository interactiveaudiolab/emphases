from .conv import Convolution
from .attention import Encoder

import emphases

def Component(**kwargs):
    if emphases.ENCODING_STACK == 'conv':
        return Convolution(**kwargs)
    elif emphases.ENCODING_STACK == 'transformer':
        return Encoder(**kwargs)
    else:
        raise ValueError(f'Encoder definition {emphases.ENCODING_STACK} is not defined')