import tensorflow as tf

class ConstantMean(tf.keras.Model):

    def __init__(self,
                 dtype,
                 name='constant_mean'):
        
        super().__init__(name=name, dtype=dtype)
        
        self.constant = tf.Variable(tf.constant(0., dtype=dtype))
        
        
    def __call__(self, x):
        return self.constant * tf.ones(x.shape[0], dtype=self.dtype)