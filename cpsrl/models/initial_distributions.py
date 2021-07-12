from abc import ABC, abstractmethod
from typing import List, Tuple
import warnings

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from cpsrl.helpers import check_shape
from cpsrl.errors import InitialDistributionError


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
    def sample_posterior(self) -> tfd.Distribution:
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
                 mu0: tf.Tensor,
                 kappa0: tf.Tensor,
                 alpha0: tf.Tensor,
                 beta0: tf.Tensor,
                 trainable: bool,
                 dtype: tf.DType):

        super().__init__(state_space=state_space, dtype=dtype)

        # Check shapes of the NG prior parameters
        check_shape([mu0, kappa0, alpha0, beta0],
                    [('S',), ('S',), ('S',), ('S',)])

        # Set NG prior parameters
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

        # Set NG posterior parameters equal to prior
        self.mu = mu0
        self.kappa = kappa0
        self.alpha = alpha0
        self.beta = beta0

        self.trainable = trainable

    def update(self):
        """
        Updates the posterior parameters of the Normal-Gamma distribution,
        using the standard expressions given in

            https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

        :return:
        """

        if not self.trainable:
            raise InitialDistributionError("Attempted to update non-trainable "
                                           "initial distribution.")

        if self.x_train.shape[0] == 0:
            warnings.warn("Attempted to update initial "
                          "distribution with no training data.")
            return

        # Number of training points
        N = self.x_train.shape[0]

        # Mean, sum and variance of initial states (entry-wise)
        x_mean = tf.reduce_mean(self.x_train, axis=0)
        x_var = tf.math.reduce_std(self.x_train, axis=0) ** 2

        # Check shape of x_sum, x_var and prior parameters
        check_shape([self.mu0,
                     self.kappa0,
                     self.alpha0,
                     self.beta0,
                     x_mean,
                     x_var],
                    [("S",), ("S",), ("S",), ("S",), ("S",), ("S",)])

        # Update posterior mu parameter
        self.mu = (self.kappa0 * self.mu0 + N * x_mean) / (self.kappa0 + N)

        # Update posterior kappa parameter
        self.kappa = self.kappa0 + N

        # Update posterior alpha parameter
        self.alpha = self.alpha0 + 0.5 * N

        # Update posterior beta parameter
        diff = x_mean - self.mu0
        self.beta = self.beta0 + 0.5 * N * x_var + \
                    0.5 * N * self.kappa0 * diff ** 2 / (self.kappa0 + N)

    def sample_posterior(self) -> tfd.Distribution:
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
        mean_scale = (self.kappa * precision) ** -0.5

        # Sample precision from Normal distribution
        normal_dist = tfd.MultivariateNormalDiag(loc=self.mu,
                                                 scale_diag=mean_scale)
        mean = normal_dist.sample()

        # Create initial distribution and return
        post_sample = tfd.MultivariateNormalDiag(loc=mean,
                                                 scale_diag=precision**-0.5)

        return post_sample


# ==============================================================================
# Gaussian initial distribution
# ==============================================================================

class IndependentGaussianMAPMean(IndependentGaussian):

    def __init__(self,
                 state_space: List[Tuple[float, float]],
                 mu0: tf.Tensor,
                 alpha0: tf.Tensor,
                 beta0: tf.Tensor,
                 trainable: bool,
                 dtype: tf.DType):

        # For MAP estimate of mean, kappa is set to 0
        kappa0 = tf.zeros_like(mu0, dtype=dtype)

        super().__init__(state_space=state_space,
                         mu0=mu0,
                         kappa0=kappa0,
                         alpha0=alpha0,
                         beta0=beta0,
                         trainable=trainable,
                         dtype=dtype)

    def sample_posterior(self) -> tfd.Distribution:
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

        # Set mean to MAP
        mean = self.mu0 if np.any(self.kappa.numpy() < 1.) else self.mu

        # Create initial distribution and return
        post_sample = tfd.MultivariateNormalDiag(loc=mean,
                                                 scale_diag=precision**-0.5)

        print(f"\nInitial distribution: mean {mean}, std {precision**-0.5}\n")

        return post_sample
