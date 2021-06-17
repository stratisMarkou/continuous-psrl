import tensorflow as tf

from cpsrl.helpers import check_shape


# ==============================================================================
# Constant mean class
# ==============================================================================

class ConstantMean(tf.keras.Model):

    def __init__(self,
                 trainable: bool,
                 dtype: tf.dtype,
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


class LinearMean(tf.keras.Model):

    def __init__(self,
                 input_dim: int,
                 trainable: bool,
                 dtype: tf.dtype,
                 name='linear_mean'):

        super().__init__(name=name, dtype=dtype)

        self.coefficients = tf.Variable(tf.zeros(shape=(input_dim,),
                                                 dtype=dtype),
                                        trainable=trainable)

    def __call__(self, x: tf.Tensor):

        check_shape(x, (-1, -1))

        return tf.einsum('d, nd -> n', self.coefficients, x)
