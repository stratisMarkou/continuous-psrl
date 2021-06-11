import tensorflow as tf
import numpy as np
    
    
class EQ(tf.keras.Model):

    def __init__(self, 
                 log_coeff,
                 log_scales,
                 trainable,
                 dtype,
                 name='eq',
                 **kwargs):
        
        super().__init__(name=name, dtype=dtype, **kwargs)
    
        # Convert parameters to tensors
        log_coeff = tf.convert_to_tensor(log_coeff, dtype=dtype)
        log_scales = tf.convert_to_tensor(log_scales, dtype=dtype)

        # Reshape parameter tensors
        log_coeff = tf.squeeze(log_coeff)
        log_scales = tf.reshape(log_scales, (-1,))
        
        # Set input dimensionality
        self.dim = log_scales.shape[0]
        
        # Set EQ parameters
        self.log_scales = tf.Variable(log_scales, trainable=trainable)
        self.log_coeff = tf.Variable(log_coeff, trainable=trainable)
        
        
    def __call__(self,
                 x1,
                 x2,
                 diag=False,
                 epsilon=None):
        
        # Convert to tensors
        x1 = tf.convert_to_tensor(x1, dtype=self.dtype)
        x2 = tf.convert_to_tensor(x2, dtype=self.dtype)

        # Get vector of lengthscales
        scales = self.scales
        
        # If calculating full covariance, add dimensions to broadcast
        if not diag:

            x1 = x1[:, None, :]
            x2 = x2[None, :, :]

            scales = self.scales[None, None, :] ** 2

        # Compute quadratic, exponentiate and multiply by coefficient
        quad = - 0.5 * (x1 - x2) ** 2 / scales
        quad = tf.reduce_sum(quad, axis=-1)
        eq_cov = self.coeff ** 2 * tf.exp(quad)
        
        # Add jitter for invertibility
        if epsilon is not None:
            eq_cov = eq_cov + epsilon * tf.eye(eq_cov.shape[0], 
                                               dtype=self.dtype)

        return eq_cov
        
    
    @property
    def scales(self):
        return tf.math.exp(self.log_scales)
    
    
    @property
    def coeff(self):
        return tf.math.exp(self.log_coeff)
    
    
    def sample_rff(self, num_features):

        # Dimension of data space
        x_dim = self.scales.shape[0]
        omega_shape = (num_features, x_dim)

        omega = tf.random.normal(shape=(num_features, x_dim), dtype=self.dtype)

        # Scale omegas by lengthscale
        omega = omega / self.scales[None, :]

        # Draw normally distributed RFF weights
        weights = tf.random.normal(mean=0.,
                                   stddev=1.,
                                   shape=(num_features,),
                                   dtype=self.dtype)

        phi = tf.random.uniform(minval=0.,
                                maxval=(2 * np.pi),
                                shape=(num_features, x_dim),
                                dtype=self.dtype)

        def rff(x):
        
            features = tf.cos(tf.einsum('fd, nd -> fn', omega, x) + phi)
            features = (2 / num_features) ** 0.5 * features * self.coeff

            return tf.einsum('f, fn -> n', weights, features)

        return rff