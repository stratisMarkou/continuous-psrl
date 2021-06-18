# %%

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from cpsrl.models.mean import ConstantMean, LinearMean
from cpsrl.models.covariance import EQ
from cpsrl.models.gp import VFEGP
from cpsrl.models.gp import VFEGPStack

# %%

# Check constant mean works
constant_mean = ConstantMean(input_dim=3,
                             trainable=True,
                             dtype=tf.float64)

ones = tf.ones(shape=(2, 3), dtype=tf.float64)
constant_mean(ones)

# Check linear mean works
linear_mean = LinearMean(input_dim=3,
                         trainable=True,
                         dtype=tf.float64)
linear_mean(ones)

# %%

# Check EQ covariance works
log_coeff = 0.
log_scales = [-1., -1.]

# Check EQ kernel works
eq = EQ(log_coeff=log_coeff,
        log_scales=log_scales,
        trainable=True,
        dtype=tf.float64)

x1 = tf.random.uniform(shape=(4, 2), dtype=tf.float64)
x2 = tf.random.uniform(shape=(3, 2), dtype=tf.float64)

eq(x1, x2)

# %%

# Check VFEGP works
dtype = tf.float64
trainable_mean = True
trainable_cov = True
trainable_noise = True
trainable_inducing = True
log_coeff = -1.
log_scales = [-1., -1., -1.]
log_noise = -4.

num_data = 10
state_dim = 2
action_dim = 1
num_ind = 10

x_train = tf.random.uniform(shape=(num_data, state_dim + action_dim),
                            dtype=dtype)
y_train = tf.random.uniform(shape=(num_data, state_dim),
                            dtype=dtype)

mean = LinearMean(input_dim=(state_dim + action_dim),
                  trainable=trainable_mean,
                  dtype=dtype)

cov = EQ(log_coeff=log_coeff,
         log_scales=log_scales,
         trainable=trainable_cov,
         dtype=dtype)

vfe_gp = VFEGP(mean=mean,
               cov=cov,
               x_train=x_train,
               y_train=y_train[:, :1],
               x_ind=None,
               num_ind=num_ind,
               trainable_inducing=trainable_inducing,
               log_noise=log_noise,
               trainable_noise=trainable_noise,
               dtype=dtype)

vfe_gp.free_energy()

# %%

# Check VFEGP works
dtype = tf.float64
trainable_mean = True
trainable_cov = True
trainable_noise = True
trainable_inducing = True
log_coeff = -1.
log_scales = [-1., -1., -1.]
log_noise = -4.

num_data = 10
state_dim = 2
action_dim = 1
num_ind = 10

x_train = tf.random.uniform(shape=(num_data, state_dim + action_dim),
                            dtype=dtype)
y_train = tf.random.uniform(shape=(num_data, state_dim),
                            dtype=dtype)

means = [LinearMean(input_dim=(state_dim + action_dim),
                    trainable=trainable_mean,
                    dtype=dtype) for i in range(state_dim)]

covs = [EQ(log_coeff=log_coeff,
           log_scales=log_scales,
           trainable=trainable_cov,
           dtype=dtype) for i in range(state_dim)]

vfe_gps = [VFEGP(mean=means[i],
                 cov=covs[i],
                 x_train=x_train,
                 y_train=y_train[:, i:i + 1],
                 x_ind=None,
                 num_ind=num_ind,
                 trainable_inducing=trainable_inducing,
                 log_noise=log_noise,
                 trainable_noise=trainable_noise,
                 dtype=dtype)
           for i in range(state_dim)]

vfe_gp_stack = VFEGPStack(vfe_gps=vfe_gps,
                          dtype=dtype)

vfe_gp_stack.free_energy()

# %%

num_steps = 10
optimizer = tf.keras.optimizers.Adam(1e-1)

for step in range(num_steps + 1):
    with tf.GradientTape() as tape:
        tape.watch(vfe_gp_stack.trainable_variables)

        free_energy = vfe_gp_stack.free_energy()
        loss = - free_energy

    gradients = tape.gradient(loss, vfe_gp_stack.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vfe_gp_stack.trainable_variables))

# %%

num_features = 100
x = tf.reshape(tf.range(15, dtype=dtype), (5, 3))

post_sample = vfe_gp_stack.sample_posterior(num_features=num_features)

# %%

post_sample(x, add_noise=True)

# %%

vfe_gp_stack.add_training_data(x_train=x_train,
                               y_train=y_train)

# %%

tf.debugging.assert_near(vfe_gp.noise, 0.01831, atol=1e-2)

# %%

DTYPE = tf.float64

# Set random seed to run the same test always
tf.random.set_seed(0)

# Number of datapoints and input dimension
N = 200
D = 1

# Set all parameters to trainable
trainable_mean = True
trainable_cov = False
trainable_noise = True
trainable_inducing = False

log_coeff = -6.
log_scales = D * [2.]
log_noise = 0.
num_ind = N

# Draw random data
x_train = 2 * tf.random.uniform(shape=(N, D),
                                dtype=DTYPE)

y_train = tf.random.normal(mean=0.,
                           stddev=1.,
                           shape=(N, 1),
                           dtype=DTYPE)

# Initialise mean and covariance
mean = LinearMean(input_dim=D,
                  trainable=trainable_mean,
                  dtype=DTYPE)

cov = EQ(log_coeff=log_coeff,
         log_scales=log_scales,
         trainable=trainable_cov,
         dtype=DTYPE)

# Initialise Variational Free Energy GP
vfe_gp = VFEGP(mean=mean,
               cov=cov,
               x_train=x_train,
               y_train=y_train,
               x_ind=None,
               num_ind=num_ind,
               trainable_inducing=trainable_inducing,
               log_noise=log_noise,
               trainable_noise=trainable_noise,
               dtype=DTYPE)

# Check optimisation works without error
num_steps = 100
optimizer = tf.keras.optimizers.Adam(1e-1)

normal = tfd.MultivariateNormalFullCovariance

for step in range(num_steps + 1):
    with tf.GradientTape() as tape:
        tape.watch(vfe_gp.trainable_variables)

        free_energy = vfe_gp.free_energy()
        loss = - free_energy

    gradients = tape.gradient(loss, vfe_gp.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vfe_gp.trainable_variables))

post_sample = vfe_gp.sample_posterior(num_features=100)
print(vfe_gp.mean.coefficients)
print(post_sample(x_train, add_noise=False).shape)

# %%