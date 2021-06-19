from abc import ABC
from typing import List
import warnings

from cpsrl.helpers import check_shape

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


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
        self.S  = len(state_space)

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

class IndependentGaussian(InitialStateDistribution):

    def __init__(self,
                 state_space: List[float],
                 mean: tf.Tensor,
                 scales: tf.Tensor,
                 trainable: bool,
                 dtype: tf.DType):

        super().__init__(state_space=state_space,
                         distribution_kind=tfd.MultivariateNormalDiag,
                         dtype=dtype)

        self.mean = mean
        self.scales = scales
        self.trainable = trainable

    def distribution(self) -> tfd.Distribution:
        return self.distribution_kind(loc=self.mean, scale_diag=self.scales)

    def update(self):

        if self.trainable:

            # Check that at least one datapoint is present
            assert self.x_train.shape[0] > 0

            self.mean = tf.reduce_mean(self.x_train, axis=0)
            self.scales = tf.math.reduce_std(self.x_train, axis=0)

        else:
            warnings.warn("Attempted to update non-trainable initial "
                          "distribution.")
