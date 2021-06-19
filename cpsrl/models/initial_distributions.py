from abc import ABC
from typing import List

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from cpsrl.helpers import check_shape


# ==============================================================================
# Base initial distribution class
# ==============================================================================

class InitialStateDistribution(ABC):

    def __init__(self,
                 state_space: List[float],
                 distribution_kind: tfd.distribution._DistributionMeta,
                 dtype: tf.DType):

        # Set state space and distribution, state space constraints not enforced
        self.state_space = state_space
        self.distribution_kind = distribution_kind

        # Set training data and data type
        self.x_train = tf.zeros(shape=(0, len(state_space)), dtype=dtype)
        self.dtype = dtype

    @property
    def distribution(self) -> tfd.Distribution:
        raise NotImplementedError

    def sample(self, num_samples) -> tf.Tensor:
        return self.distribution.sample(sample_shape=(num_samples,))

    def add_training_data(self, x_train: tf.Tensor):

        check_shape(x_train, ('N', len(self.state_space)))

        self.x_train = tf.concat([self.x_train, x_train], axis=0)


# ==============================================================================
# Gaussian initial distribution
# ==============================================================================

class IndependentGaussian(ABC):

    def __init__(self,
                 state_space: List[float],
                 distribution: tfd.Distribution,
                 mean: tf.Tensor,
                 variances: tf.Tensor):

        super().__init__(state_space=state_space,
                         distribution=distribution)

        self.mean = mean
        self.variances = variances

    def sample(self) -> tf.Tensor:
        pass

    def add_training_data(self, x_train: tf.Tensor):
        pass

    def update(self):
        pass
