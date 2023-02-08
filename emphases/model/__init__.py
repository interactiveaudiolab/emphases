from .framewise import Framewise
from .wordwise import Wordwise
from .attention import Encoder

import emphases


def Model():
    """Create a model"""
    if emphases.METHOD == 'framewise':
        return Framewise()
    elif emphases.METHOD == 'wordwise':
        return Wordwise()
    elif emphases.METHOD == 'attention':
        return Encoder()
    else:
        raise ValueError(f'Model {emphases.METHOD} is not defined')
