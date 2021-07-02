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
                 dtype: tf.DType = None):

        self.state_space = state_space
        self.S = len(state_space)

        self.action_space = action_space
        self.A = len(action_space)

        if dtype is not None:
            self.dtype = dtype

    @abstractmethod
    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def reset(self):
        pass

# ==============================================================================
# Random policy
# ==============================================================================

class RandomPolicy(Policy):

    def __call__(self, state: tf.Tensor) -> tf.Tensor:

        action = [tf.random.uniform(minval=low,
                                    maxval=high,
                                    shape=(state.shape[0],),
                                    dtype=self.dtype)
                  for low, high in self.action_space]
        action = tf.stack(action, axis=1)

        return action

    def reset(self):
        pass


# ==============================================================================
# Fully connected neural network policy
# ==============================================================================

class FCNPolicy(Policy, tf.keras.Model):

    def __init__(self,
                 hidden_sizes: List[int],
                 state_space: List[Tuple],
                 action_space: List[Tuple],
                 trainable: bool,
                 dtype: tf.DType,
                 name='fcn_policy',
                 **kwargs):

        # Policy superclass initialisation
        Policy.__init__(self,
                        state_space=state_space,
                        action_space=action_space)

        # Keras superclass initialisation
        tf.keras.Model.__init__(self, dtype=dtype, name=name, **kwargs)

        # Initialise the layers
        self.S = len(state_space)
        self.A = len(action_space)

        # Set hidden layer sizes
        self.sizes = [self.S] + hidden_sizes + [self.A]
        self.sizes = list(zip(self.sizes[:-1], self.sizes[1:]))

        # Specify whether policy is trainable
        self.trainable = trainable

        # Create weight and bias tensors
        self.W = [tf.random.normal(shape=(s1, s2), dtype=self.dtype) / s1 ** 0.5
                  for s1, s2, in self.sizes]
        self.W = [tf.Variable(W, trainable=self.trainable) for W in self.W]

        self.b = [tf.zeros(shape=(1, s2), dtype=self.dtype)
                  for s1, s2, in self.sizes]
        self.b = [tf.Variable(b, trainable=self.trainable) for b in self.b]

        # Tensors for scaling the final action, outputed by the policy
        self.action_ranges = [(a2 - a1) / 2 for a1, a2 in self.action_space]
        self.action_ranges = tf.convert_to_tensor(self.action_ranges)
        self.action_ranges = tf.cast(self.action_ranges, dtype=dtype)

        self.action_centers = [0.5 * (a1 + a2) for a1, a2 in self.action_space]
        self.action_centers = tf.convert_to_tensor(self.action_centers)
        self.action_centers = tf.cast(self.action_centers, dtype=dtype)


    def reset(self):

        # Create and assign weight and bias tensors
        for i, (s1, s2) in enumerate(self.sizes):

            W = tf.random.normal(shape=(s1, s2), dtype=self.dtype) / s1 ** 0.5
            self.W[i].assign(W)

            b = tf.zeros(shape=(1, s2), dtype=self.dtype)
            self.b[i].assign(b)

    def __call__(self, tensor: tf.Tensor) -> tf.Tensor:

        # Check input shape is valid
        check_shape(tensor, (-1, self.S))

        for i, (W, b) in enumerate(zip(self.W, self.b)):

            # Multiply by weights and add bias
            tensor = tensor @ W + b

            # If not at the last layer, apply relu
            if i < len(self.sizes) - 1:
                tensor = tf.nn.relu(tensor)

            # If  at last layer, apply sine and scale to action range
            else:
                tensor = tf.math.tanh(tensor)
                tensor = tensor * self.action_ranges[None, :]
                tensor = tensor + self.action_centers

        return tensor

