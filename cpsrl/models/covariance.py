from abc import abstractmethod
from typing import List, Callable

from cpsrl.helpers import check_shape, VariableOrTensor

import numpy as np
import tensorflow as tf


# ==============================================================================
# Base covariance class
# ==============================================================================

class Covariance(tf.keras.Model):

    def __init__(self, dtype: tf.DType, name: str = 'eq', **kwargs):

        super().__init__(name=name, dtype=dtype, **kwargs)

    @abstractmethod
    def __call__(self,
                 x1: VariableOrTensor,
                 x2: VariableOrTensor,
                 diag: bool = False,
                 epsilon: float = None) -> tf.Tensor:
        pass

    @abstractmethod
    def sample_rff(self, num_features: int) -> Callable:
        pass


# ==============================================================================
# EQ covariance
# ==============================================================================

class EQ(Covariance):

    def __init__(self, 
                 log_coeff: float,
                 log_scales: List[float],
                 trainable: bool,
                 dtype: tf.DType,
                 name: str = 'eq',
                 **kwargs):
        
        super().__init__(name=name, dtype=dtype, **kwargs)
    
        # Convert parameters to tensors
        log_coeff = tf.convert_to_tensor(log_coeff, dtype=dtype)
        log_scales = tf.convert_to_tensor(log_scales, dtype=dtype)

        # Reshape parameter tensors
        log_coeff = tf.squeeze(log_coeff)
        log_scales = tf.reshape(log_scales, (-1,))
        
        # Set input dimensionality
        self.input_dim = log_scales.shape[0]
        
        # Set EQ coefficient and lengthscales
        self.log_coeff = tf.Variable(log_coeff, trainable=trainable)
        self.log_scales = tf.Variable(log_scales, trainable=trainable)

    def __call__(self,
                 x1: tf.Tensor,
                 x2: tf.Tensor,
                 diag: bool = False,
                 epsilon: float = None) -> tf.Tensor:

        # Check input shapes
        check_shape([x1, x2, self.log_scales], [(-1, 'D'), (-1, 'D'), ('D',)])

        # Add dimensions to broadcast
        x1 = x1[:, None, :]
        x2 = x2[None, :, :]

        # Compute quadratic, exponentiate and multiply by coefficient
        quad = - 0.5 * ((x1 - x2) / self.scales[None, None, :]) ** 2
        quad = tf.reduce_sum(quad, axis=-1)
        eq_cov = self.coeff ** 2 * tf.exp(quad)

        # Add jitter for invertibility
        if epsilon is not None:
            eq_cov = eq_cov + epsilon * tf.eye(eq_cov.shape[0], 
                                               dtype=self.dtype)

        eq_cov = tf.linalg.diag_part(eq_cov) if diag else eq_cov

        return eq_cov

    @property
    def scales(self) -> tf.Tensor:
        return tf.math.exp(self.log_scales)
    
    @property
    def coeff(self) -> tf.Tensor:
        return tf.math.exp(self.log_coeff)

    def sample_rff(self, num_features: int) -> Callable:

        omega = tf.random.normal(shape=(num_features, self.input_dim),
                                 dtype=self.dtype)

        # Scale omegas by lengthscale
        omega = omega / self.scales[None, :]

        # Draw normally distributed RFF weights
        weights = tf.random.normal(mean=0.,
                                   stddev=1.,
                                   shape=(num_features,),
                                   dtype=self.dtype)

        phi = tf.random.uniform(minval=0.,
                                maxval=(2 * np.pi),
                                shape=(num_features, 1),
                                dtype=self.dtype)

        def rff(x):

            check_shape(x, (-1, self.input_dim))
            
            features = tf.cos(tf.einsum('fd, nd -> fn', omega, x) + phi)
            features = (2 / num_features) ** 0.5 * self.coeff * features

            return tf.einsum('f, fn -> n', weights, features)

        return rff
