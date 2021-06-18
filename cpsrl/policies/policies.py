from abc import abstractmethod
from typing import List, Tuple

from cpsrl.helpers import check_shape

import numpy as np
import tensorflow as tf


# ==============================================================================
# Base policy class
# ==============================================================================

class Policy:

    def __init__(self,
                 state_space: List[Tuple],
                 action_space: List[Tuple],
                 dtype: tf.DType):

        self.state_space = state_space
        self.S = len(state_space)

        self.action_space = action_space
        self.A = len(action_space)

        self.dtype = dtype


    @abstractmethod
    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        pass


# ==============================================================================
# Random policy
# ==============================================================================

class RandomPolicy(Policy):

    def __call__(self, state: tf.Tensor) -> tf.Tensor:

        state = 1e-3 * tf.reduce_sum(state, axis=1, keepdims=True)

        action = [tf.random.uniform(minval=low,
                                    maxval=high,
                                    shape=(state.shape[0],),
                                    dtype=self.dtype)
                  for low, high in self.action_space]
        action = tf.stack(action, axis=1)
        action = action + state

        return action
