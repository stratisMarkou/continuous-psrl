import os
import sys
import random
from typing import Tuple, Sequence, Generator, Union, Optional

import numpy as np
import tensorflow as tf

from cpsrl.errors import ShapeError

ArrayType = Union[np.ndarray, Sequence[np.ndarray]]
ShapeType = Union[Tuple, Sequence[Tuple]]


def check_shape(arrays: ArrayType,
                shapes: ShapeType,
                shape_dict: Optional[dict] = None,
                keep_dict: bool = False) -> Union[ArrayType, Tuple[ArrayType, dict]]:
    
    if (type(arrays) in [list, tuple]) and \
       (type(shapes) in [list, tuple]):
        
        shape_dict = {} if shape_dict is None else shape_dict
        
        if len(arrays) != len(shapes):
            raise ShapeError(f"Got number of tesors/arrays {len(arrays)}, "
                             f"and number of shapes {len(shapes)}.")
            
        for array, shape in zip(arrays, shapes):
            array, shape_dict = _check_shape(array,
                                             shape,
                                             shape_dict=shape_dict)
            
        if keep_dict:
            return arrays, shape_dict
        
        else:
            return arrays

    else:
        return _check_shape(arrays, shapes)[0]


def _check_shape(array: np.ndarray,
                 shape: Tuple,
                 shape_dict: Optional[dict] = None
                 ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    
    array_shape = array.shape
    check_string_names = shape_dict is not None
    
    # Check if array shape and shape have same length (i.e. are comparable)
    if len(array_shape) != len(shape):
        raise ShapeError(f"Tensor/Array shape {array_shape}, "
                         f"check shape {shape}")
    
    # Check if shapes are compatible
    for s1, s2 in zip(array.shape, shape):
        
        # Try to convert s2 to int
        try:
            # If s2 == '-1', any shape passes test
            if int(s2) == -1:
                continue

            elif s1 != int(s2):

                raise ShapeError(f"Tensor/Array shape {array_shape}, "
                                 f"check shape {shape}")
                
        # If s2 string found, try to match against dict
        except ValueError:
            
            if not (s2 in shape_dict):
                shape_dict[s2] = s1
                
            elif shape_dict[s2] != s1:
                raise ShapeError(f"Tensor/Array shape had {s2} axis size {s1}, "
                                 f"expected axis size {shape_dict[s2]}.")
            
    if check_string_names:
        return array, shape_dict
    
    else:
        return array


def set_seed(seed: int) -> Generator:
    """
    Sets the global random seed and returns a generator for instantiating
    new RNGs. Useful to provide policies, envs, etc. with their own RNG.
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    return RNG(seed)


def RNG(seed: int):
    """Generates independent numpy RNGs."""
    sub_seed = 0
    while True:
        sub_seed += 1
        yield np.random.Generator(np.random.Philox(key=seed + sub_seed))


class Logger(object):
    """A simple logger."""
    def __init__(self, directory: str, exp_name: str):
        self.terminal = sys.stdout
        self.directory = directory
        self.exp_name = exp_name

    def write(self, message):
        self.terminal.write(message)
        log_filename = os.path.join(self.directory, f"{self.exp_name}.txt")
        with open(log_filename, "a") as f:
            f.write(message)

    def flush(self):
        pass
