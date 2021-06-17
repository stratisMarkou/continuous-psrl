from abc import abstractmethod

import tensorflow as tf

from cpsrl.helpers import check_shape


# ==============================================================================
# Base mean class
# ==============================================================================

class Mean(tf.keras.Model):

    def __init__(self, dtype: tf.DType, name='mean'):

        super().__init__(name=name, dtype=dtype)

    @abstractmethod
    def __call__(self, x: tf.Tensor):
        pass


# ==============================================================================
# Constant mean class
# ==============================================================================

class ConstantMean(Mean):

    def __init__(self,
                 trainable: bool,
                 dtype: tf.DType,
                 name='constant_mean'):
        
        super().__init__(name=name, dtype=dtype)
        
        self.constant = tf.Variable(tf.constant(0., dtype=dtype),
                                    trainable=trainable)
        
    def __call__(self, x: tf.Tensor):

        check_shape(x, (-1, -1))

        return self.constant * tf.ones(x.shape[0], dtype=self.dtype)


# ==============================================================================
# Linear mean class
# ==============================================================================


class LinearMean(Mean):

    def __init__(self,
                 input_dim: int,
                 trainable: bool,
                 dtype: tf.DType,
                 name='linear_mean'):

        super().__init__(name=name, dtype=dtype)

        self.input_dim = input_dim
        self.coefficients = tf.Variable(tf.zeros(shape=(input_dim,),
                                                 dtype=dtype),
                                        trainable=trainable)

    def __call__(self, x: tf.Tensor):

        check_shape(x, (-1, self.input_dim))

        return tf.einsum('d, nd -> n', self.coefficients, x)
