from abc import abstractmethod
from typing import List, Tuple

import numpy as np
import tensorflow as tf


# ==============================================================================
# Base policy class
# ==============================================================================

class Policy:

    def __init__(self,
                 state_space: List[Tuple],
                 action_space: List[Tuple]):

        self.state_space = state_space
        self.S = len(state_space)

        self.action_space = action_space
        self.A = len(action_space)


    @abstractmethod
    def __call__(self, action: tf.Tensor) -> tf.Tensor:
        pass


# ==============================================================================
# Random policy
# ==============================================================================

class RandomPolicy(Policy):
    pass
