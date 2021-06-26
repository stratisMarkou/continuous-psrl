from abc import ABC, abstractmethod
from typing import List, Tuple
import warnings

from cpsrl.helpers import check_shape
from cpsrl.errors import InitialDistributionError

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


# ==============================================================================
# Base initial distribution class
# ==============================================================================

class InitialStateDistribution(ABC):

    def __init__(self,
                 state_space: List[Tuple[float, float]],
                 dtype: tf.DType):

        # Set state space and distribution, state space constraints not enforced
        self.state_space = state_space
        self.S = len(state_space)

        # Set training data and data type
        self.x_train = tf.zeros(shape=(0, len(state_space)), dtype=dtype)
        self.dtype = dtype

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def posterior_sample(self) -> tfd.Distribution:
        pass

    def add_training_data(self, init_states: tf.Tensor):

        check_shape(init_states, (-1, self.S))

        self.x_train = tf.concat([self.x_train, init_states], axis=0)


# ==============================================================================
# Gaussian initial distribution with Normal Gamma prior
# ==============================================================================

class IndependentGaussian(InitialStateDistribution):

    def __init__(self,
                 state_space: List[Tuple[float, float]],
                 mu: tf.Tensor,
                 kappa: tf.Tensor,
                 alpha: tf.Tensor,
                 beta: tf.Tensor,
                 trainable: bool,
                 dtype: tf.DType):

        super().__init__(state_space=state_space, dtype=dtype)

        # Check shapes of the NG prior parameters
        check_shape([mu, kappa, alpha, beta],
                    [('S',), ('S',), ('S',), ('S',)])

        # Set NG prior parameters
        self.mu0 = mu
        self.kappa0 = kappa
        self.alpha0 = alpha
        self.beta0 = beta

        # Set NG posterior parameters equal to prior
        self.mu = mu
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta

        self.trainable = trainable

    def update(self):
        """
        Updates the posterior parameters of the Normal-Gamma distribution,
        using the standard expressions given in

            https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

        :return:
        """

        if self.x_train.shape[0]:
            raise InitialDistributionError("Attempted to update initial "
                                           "distribution with no training "
                                           "data.")

        # Number of training points
        N = self.x_train.shape[0]

        # Mean, sum and variance of initial states (entry-wise)
        x_mean = tf.reduce_mean(self.x_train, axis=0)
        x_sum = N * x_mean
        x_var = tf.math.reduce_std(self.x_train, axis=0) ** 2

        # Check shape of x_sum, x_var and prior parameters
        check_shape([x_sum, self.mu0, self.kappa0, self.alpha0, self.beta0],
                    [("S",), ("S",), ("S",), ("S",), ("S",)])

        # Update posterior mu parameter
        self.mu = (self.kappa0 * self.mu0 + x_sum) / (self.kappa0 + N)

        # Update posterior kappa parameter
        self.kappa = self.kappa0 + N

        # Update posterior alpha parameter
        self.alpha = self.alpha0 + 0.5 * N

        # Update posterior beta parameter
        diff = x_mean - self.mu0
        self.beta = self.beta0 + 0.5 * x_var + \
                    0.5 * N * self.kappa0 * diff ** 2 / (self.kappa0 + N)

    def posterior_sample(self) -> tfd.Distribution:
        """
        Samples an initial distribution from the posterior over initial
        distributions, returning the result as a tfd.Distribution, which can
        itself be sampled and used for initialising a rollout.

        :return:
        """

        # Check shape of alpha and beta posterior parameters
        check_shape([self.alpha, self.beta], [(self.S,), (self.S,)])

        # Sample precision from Gamma distribution
        gamma_dist = tfd.Gamma(concentration=self.alpha, rate=self.beta)
        precision = gamma_dist.sample()

        # Check shape of precision and kappa/mu posterior parameters
        check_shape([precision, self.kappa, self.mu],
                    [(self.S,), (self.S,), (self.S,)])

        # Compute scale of mean of initial distribution
        mean_scale = (self.kappa * self.precision) ** -0.5

        # Sample precision from Normal distribution
        normal_dist = tfd.MultivariateNormalDiag(mean=self.mu,
                                                 scale_diag=mean_scale)
        mean = normal_dist.sample()

        # Create initial distribution and return
        post_sample = tfd.MultivariateNormalDiag(mean=mean,
                                                 scale_diag=precision**-1)

        return post_sample
