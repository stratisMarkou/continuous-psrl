from abc import ABC, abstractmethod

import tensorflow as tf

from cpsrl.helpers import check_shape, VariableOrTensor


# ==============================================================================
# Base mean class
# ==============================================================================

class Mean(ABC, tf.keras.Model):

    def __init__(self, dtype: tf.DType, name="mean"):

        super().__init__(name=name, dtype=dtype)

    @abstractmethod
    def __call__(self, x: VariableOrTensor) -> tf.Tensor:
        pass

    @abstractmethod
    def parameter_summary(self) -> str:
        pass

# ==============================================================================
# Constant mean class
# ==============================================================================


class ConstantMean(Mean):

    def __init__(self,
                 input_dim: int,
                 trainable: bool,
                 dtype: tf.DType,
                 name='constant_mean'):
        
        super().__init__(name=name, dtype=dtype)

        self.input_dim = input_dim
        self.constant = tf.Variable(tf.constant(0., dtype=dtype),
                                    trainable=trainable)
        
    def __call__(self, x: VariableOrTensor) -> tf.Tensor:

        check_shape(x, (-1, self.input_dim))

        return self.constant * tf.ones((x.shape[0], 1), dtype=self.dtype)

    def parameter_summary(self) -> str:
        return f"Constant mean \n" \
               f"\t constant: {self.constant.numpy()}"

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

        self.coefficients = tf.Variable(tf.zeros(shape=(input_dim, 1),
                                                 dtype=dtype),
                                        trainable=trainable)

        self.constant = tf.Variable(tf.zeros(shape=(),
                                             dtype=dtype),
                                    trainable=trainable)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:

        check_shape(x, (-1, self.input_dim))

        return x @ self.coefficients + self.constant

    def parameter_summary(self) -> str:
        return f"Linear mean\n" \
               f"\tconstant: {self.constant.numpy()}\n" \
               f"\tcoefficients: {self.coefficients.numpy()[:, 0]}"
