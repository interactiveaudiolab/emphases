from .framewise import Framewise
from .wordwise import Wordwise
from .components import Component

import emphases


def Model():
    """Create a model"""
    if emphases.METHOD == 'framewise':
        return Framewise()
    elif emphases.METHOD == 'wordwise':
        return Wordwise()
    else:
        raise ValueError(f'Model {emphases.METHOD} is not defined')
