import numpy as np
import torch

from typing import NewType, List, Union

__all__ = [
    'Tensor',
    'Array',
]

Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)