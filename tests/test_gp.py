import tensorflow as tf
import numpy as np

from cpsrl.models.mean import ConstantMean, LinearMean
from cpsrl.models.covariance import EQ
from cpsrl.models.gp import VFEGP
from cpsrl.helpers import check_shape
from cpsrl.errors import ModelError

DTYPE = tf.float64


# ==============================================================================
# Test forward pass through mean
# ==============================================================================


def test_constant_mean():
    """
    Initialises a constant mean and passes some randomly sampled data
    through it, and checks if the output mean has the correct shape.
    :return:
    """

    # Set random seeds
    np.random.seed(0)
    tf.random.set_seed(0)

    max_points = 100

    for D in range(1, 10):

        N = np.random.randint(1, max_points)

        x_rand = tf.random.uniform(shape=(N, D), dtype=DTYPE)

        constant_mean = ConstantMean(input_dim=D,
                                     trainable=False,
                                     dtype=DTYPE)

        output = constant_mean(x_rand)

        check_shape([x_rand, output], [(N, D), (N, 1)])


def test_linear_mean():
    """
    Initialises a linear mean and passes some randomly sampled data
    through it, and checks if the output mean has the correct shape.
    :return:
    """

    # Set random seeds
    np.random.seed(0)
    tf.random.set_seed(0)

    max_points = 100

    for D in range(1, 10):

        N = np.random.randint(1, max_points)

        x_rand = tf.random.uniform(shape=(N, D), dtype=DTYPE)

        linear_mean = LinearMean(input_dim=D,
                                 trainable=False,
                                 dtype=DTYPE)

        output = linear_mean(x_rand)

        check_shape([x_rand, output], [(N, D), (N, 1)])


# ==============================================================================
# Test forward pass through covariance
# ==============================================================================

def test_eq_cov():
    """
    Initialises an EQ covariance and passes some randomly sampled data
    through it, and checks if the output covariance has the correct shape.
    :return:
    """

    # Set random seeds
    np.random.seed(0)
    tf.random.set_seed(0)

    max_points = 100

    for D in range(1, 10):

        N1 = np.random.randint(1, max_points)
        N2 = np.random.randint(1, max_points)

        x1 = tf.random.uniform(shape=(N1, D), dtype=DTYPE)
        x2 = tf.random.uniform(shape=(N2, D), dtype=DTYPE)

        # Check EQ covariance works
        log_coeff = 0.
        log_scales = D * [0.]

        # Check EQ kernel works
        eq = EQ(log_coeff=log_coeff,
                log_scales=log_scales,
                trainable=False,
                dtype=DTYPE)

        cov = eq(x1, x2)
        cov_diag = eq(x1, x1, diag=True)

        check_shape([x1, x2, cov, cov_diag],
                    [(N1, D), (N2, D), (N1, N2), (N1,)])


# ==============================================================================
# Test VFEGP trains and handles the zero data case
# ==============================================================================

def test_vfe_gp_no_data():
    """
    Tests whether initialising with invalid parameters will throw the
    appropriate errors. The cases tested are:

        - Initialising with x_train data only
        - Initialising with y_train data only
        - Initialising with no training data and num_ind is not None.
        - Initialising with no training data and calling VFEGP.free_energy()

    :return:
    """

    # Set random seed to run the same test always
    tf.random.set_seed(0)

    # Number of datapoints and input dimension
    N = 1000
    D = 1

    # Set all parameters to trainable
    trainable_mean = False
    trainable_cov = False
    trainable_noise = False
    trainable_inducing = False

    log_coeff = -6.
    log_scales = D * [2.]
    log_noise = 0.
    num_ind = 10
    x_ind = tf.random.uniform((num_ind, D), dtype=DTYPE)

    # Draw random data
    x_train = tf.random.uniform(shape=(N, D),
                                dtype=DTYPE)

    y_train = tf.random.normal(mean=0.,
                               stddev=1.,
                               shape=(N, 1),
                               dtype=DTYPE)

    # Initialise mean and covariance
    mean = ConstantMean(input_dim=D,
                        trainable=trainable_mean,
                        dtype=DTYPE)

    cov = EQ(log_coeff=log_coeff,
             log_scales=log_scales,
             trainable=trainable_cov,
             dtype=DTYPE)

    # Initialising using only x_train should throw error
    try:
        VFEGP(mean=mean,
              cov=cov,
              input_dim=D,
              x_train=x_train,
              y_train=None,
              x_ind=x_ind,
              num_ind=None,
              trainable_inducing=trainable_inducing,
              log_noise=log_noise,
              trainable_noise=trainable_noise,
              dtype=DTYPE)

        raise Exception

    except ModelError:
        pass

    # Initialising using only y_train should throw error
    try:
        VFEGP(mean=mean,
              cov=cov,
              input_dim=D,
              x_train=None,
              y_train=y_train,
              x_ind=x_ind,
              num_ind=None,
              trainable_inducing=trainable_inducing,
              log_noise=log_noise,
              trainable_noise=trainable_noise,
              dtype=DTYPE)

        raise Exception

    except ModelError:
        pass

    # Initialising with no data using num_ind should raise error
    try:

        VFEGP(mean=mean,
              cov=cov,
              input_dim=D,
              x_train=None,
              y_train=None,
              x_ind=None,
              num_ind=num_ind,
              trainable_inducing=trainable_inducing,
              log_noise=log_noise,
              trainable_noise=trainable_noise,
              dtype=DTYPE)

        raise Exception

    except ModelError:
        pass

    # Evaluating the free energy with no training data should throw error
    try:

        vfe_gp = VFEGP(mean=mean,
                       cov=cov,
                       input_dim=D,
                       x_train=None,
                       y_train=None,
                       x_ind=x_ind,
                       num_ind=None,
                       trainable_inducing=trainable_inducing,
                       log_noise=log_noise,
                       trainable_noise=trainable_noise,
                       dtype=DTYPE)

        vfe_gp.free_energy()

        raise Exception

    except ModelError:
        pass

# ==============================================================================
# Test VFEGP recovers correct constant mean and noise level
# ==============================================================================

def test_vfe_gp_noise():
    """
    Trains a VFEGP on pure-noise iid data, and checks if the model recovers the
    correct noise level.
    :return:
    """

    # Set random seed to run the same test always
    tf.random.set_seed(0)

    # Number of datapoints and input dimension
    N = 1000
    D = 3

    # Set all parameters to trainable
    trainable_mean = True
    trainable_cov = False
    trainable_noise = True
    trainable_inducing = False

    log_coeff = -6.
    log_scales = D * [5.]
    log_noise = 0.
    num_ind = 20

    # Ground truth mean and noise std
    data_constant = 1.
    data_std = 1e-1

    # Draw random data
    x_train = tf.random.uniform(shape=(N, D),
                                dtype=DTYPE)

    y_train = tf.random.normal(mean=data_constant,
                               stddev=data_std,
                               shape=(N, 1),
                               dtype=DTYPE)

    # Initialise mean and covariance
    mean = ConstantMean(input_dim=D,
                        trainable=trainable_mean,
                        dtype=DTYPE)

    cov = EQ(log_coeff=log_coeff,
             log_scales=log_scales,
             trainable=trainable_cov,
             dtype=DTYPE)

    # Initialise Variational Free Energy GP
    vfe_gp = VFEGP(mean=mean,
                   cov=cov,
                   input_dim=D,
                   x_train=x_train,
                   y_train=y_train,
                   x_ind=None,
                   num_ind=num_ind,
                   trainable_inducing=trainable_inducing,
                   log_noise=log_noise,
                   trainable_noise=trainable_noise,
                   dtype=DTYPE)

    # Check optimisation works without error
    num_steps = 200
    optimizer = tf.keras.optimizers.Adam(1e-1)

    for step in range(num_steps + 1):
        with tf.GradientTape() as tape:
            tape.watch(vfe_gp.trainable_variables)

            free_energy = vfe_gp.free_energy()
            loss = - free_energy

        gradients = tape.gradient(loss, vfe_gp.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vfe_gp.trainable_variables))

    print(vfe_gp.mean.constant, data_constant)
    print(vfe_gp.noise)

    # Assert model has learnt the correct noise level
    tf.debugging.assert_near(vfe_gp.mean.constant, data_constant, rtol=1e-1)
    tf.debugging.assert_near(vfe_gp.noise, data_std, rtol=1e-1)


# ==============================================================================
# Test VFEGP recovers correct linear mean and noise level
# ==============================================================================

def test_vfe_gp_linear():
    """
    Trains a VFEGP on pure-noise iid data, and checks if the model recovers the
    correct noise level.
    :return:
    """

    # Set random seed to run the same test always
    tf.random.set_seed(0)

    # Number of datapoints and input dimension
    N = 1000
    D = 3

    # Set all parameters to trainable
    trainable_mean = True
    trainable_cov = False
    trainable_noise = True
    trainable_inducing = False

    log_coeff = -6.
    log_scales = D * [5.]
    log_noise = 0.
    num_ind = 20

    # Ground truth mean and noise std
    data_linear_weights = tf.range(D, dtype=DTYPE) + 1.
    data_constant = 1.
    data_std = 1e-1

    # Draw random data
    x_train = tf.random.uniform(shape=(N, D),
                                dtype=DTYPE)

    y_train = tf.random.normal(mean=data_constant,
                               stddev=data_std,
                               shape=(N, 1),
                               dtype=DTYPE)

    # Add linear mean to data
    y_train = y_train + tf.einsum('ij, j -> i',
                                  x_train,
                                  data_linear_weights)[:, None]

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
                   input_dim=D,
                   x_train=x_train,
                   y_train=y_train,
                   x_ind=None,
                   num_ind=num_ind,
                   trainable_inducing=trainable_inducing,
                   log_noise=log_noise,
                   trainable_noise=trainable_noise,
                   dtype=DTYPE)

    # Check optimisation works without error
    num_steps = 200
    optimizer = tf.keras.optimizers.Adam(1e-1)

    for step in range(num_steps + 1):
        with tf.GradientTape() as tape:
            tape.watch(vfe_gp.trainable_variables)

            free_energy = vfe_gp.free_energy()
            loss = - free_energy

        gradients = tape.gradient(loss, vfe_gp.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vfe_gp.trainable_variables))

    # Assert model has learnt the ground truth parameters
    tf.debugging.assert_near(vfe_gp.mean.constant,
                             data_constant,
                             rtol=1e-2)

    tf.debugging.assert_near(vfe_gp.mean.coefficients[:, 0],
                             data_linear_weights,
                             rtol=1e-2)

    tf.debugging.assert_near(vfe_gp.noise,
                             data_std,
                             rtol=1e-1)
