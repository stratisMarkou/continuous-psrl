from typing import List, Callable, Optional

import tensorflow as tf
import numpy as np

from cpsrl.models.mean import Mean
from cpsrl.models.covariance import Covariance
from cpsrl.helpers import check_shape


# ==============================================================================
# Variational Sparse Gaussian Process
# ==============================================================================


class VFEGPStack(tf.keras.Model):
    
    def __init__(self,
                 vfe_gps: List,
                 dtype: tf.DType,
                 name: str = 'gp_stack',
                 **kwargs):
        """
        Creates a stack of single-output VFE GP models, for multi-output
        regression, where the models model the different outputs independently.

        :param vfe_gps: VFE GP models to use in the stack
        :param dtype: dtype of the stack
        :param name: name for the model
        :param kwargs:
        """
        
        super().__init__(name=name, dtype=dtype, **kwargs)
        
        self.vfe_gps = []
        
        for vfe_gp in vfe_gps:
            self.vfe_gps.append(vfe_gp)
    
    def add_training_data(self, x_train: tf.Tensor, y_train: tf.Tensor):
        """
        Adds training data to the model, giving each VFE GP in the stack an
        identical copy of the data.

        :param x_train: tensor of shape (N, D1)
        :param y_train: tensor of shape (N, D2)
        :return:
        """
        
        check_shape([x_train, y_train], [('N', '-1'), ('N', '-1')])
        
        for i, vfe_gp in enumerate(self.vfe_gps):
            vfe_gp.add_training_data(x_train, y_train[:, i:i+1])
    
    def sample_posterior(self, num_features: int) -> Callable:
        """
        Produces a posterior sample, using *num_features* random fourier
        features, returning the sample in the form of a Callable with signature:

            post_sample(x: tf.Tensor, add_noise: bool).

        The *x* argument is assumed to be a tf.Tensor of shape (N, D),
        where D is the input dimension of the VFEGP. If *add_noise* is True,
        calling the posterior sample adds noise to its output.

        :param num_features:
        :return: posterior sample as Callable
        """
        
        post_samples = [vfe_gp.sample_posterior(num_features=num_features) \
                        for vfe_gp in self.vfe_gps]
        
        def post_sample(x: tf.Tensor, add_noise: bool) -> tf.Tensor:
            
            # Check shape of input against training
            check_shape([x, self.vfe_gps[0].x_train],
                        [('N1', 'D'), ('N2', 'D')])

            samples = [sample(x, add_noise) for sample in post_samples]
            
            return tf.stack(samples, axis=1)
            
        return post_sample
    
    def free_energy(self) -> tf.Tensor:
        return tf.reduce_sum([vfe_gp.free_energy() for vfe_gp in self.vfe_gps])
    

# ==============================================================================
# Variational Sparse Gaussian Process
# ==============================================================================


class VFEGP(tf.keras.Model):
    
    def __init__(self,
                 mean: Mean,
                 cov: Covariance,
                 x_train: tf.Tensor,
                 y_train: tf.Tensor,
                 trainable_inducing: bool,
                 log_noise: float,
                 trainable_noise: bool,
                 dtype: tf.DType,
                 x_ind: Optional[tf.Tensor] = None,
                 num_ind: Optional[int] = None,
                 name='vfegp',
                 **kwargs):
        """
        Gaussian Process model using the Variational Free Energy approximation
        of Titsias.

        Inducing points can either be initialised by specifiying their
        initial locations using *x_ind*, or by specifying an integer number
        *num_ind* of inducing points to use, in which case the points are
        initialised as a random subset of the training points.

        Exactly one of *x_ind* or *num_ind* should be specified, otherwise an
        error will be thrown.

        :param mean: mean function of the GP
        :param cov: covariance function of the GP
        :param x_train: training inputs, shape (N, D)
        :param y_train: training outputs, shape (N, 1)
        :param trainable_inducing: whether to allow inducing locations to train
        :param log_noise: log of noise of VFEGP
        :param trainable_noise: whether to allow the noise level to train
        :param dtype: data type of the GP model
        :param x_ind: optional, initial inducing point locations
        :param num_ind: optional, number of inducing points to use
        :param name:
        :param kwargs:
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        
        # Check x_train and y_train have compatible shapes
        check_shape([x_train, y_train], [('N', 'D'), ('N', '1')])
        
        # Set training data and inducing point initialisation
        self.x_train = tf.zeros(shape=(0, x_train.shape[1]), dtype=dtype)
        self.y_train = tf.zeros(shape=(0, 1), dtype=dtype)
        
        self.add_training_data(x_train, y_train)
        
        # Initialise inducing points
        self.x_ind = self.init_inducing(x_ind, num_ind)
        self.x_ind = tf.Variable(self.x_ind, trainable=trainable_inducing)
        
        # Set mean and covariance functions
        self.mean = mean
        self.cov = cov
    
        # Set log of noise parameter
        self.log_noise = tf.convert_to_tensor(log_noise, dtype=dtype)
        self.log_noise = tf.Variable(self.log_noise, trainable=trainable_noise)
        
    def init_inducing(self, x_ind: tf.Tensor, num_ind: int) -> tf.Tensor:
        """
        Creates a tensor containing the initial inducing point locations.
        Assumes exactly one of *x_ind* or *num_ind* is specified,
        and otherwise throws an error.

        If *x_ind* is specified, this tensor is used to specify the inducing
        point locations. If *num_ind* is specified, then the model uses a
        random subset of the training data of size *num_ind* as the initial
        inducing point locations.

        :param x_ind: optional, initial inducing point locations, shape (N, D)
        :param num_ind: optional, number of inducing points to use
        :return:
        """
        
        assert ((x_ind is not None) and (num_ind is None)) or \
               ((x_ind is None) and (num_ind is not None))
        
        # Set inducing points either to initial locations or on training data
        if x_ind is not None:

            # Check the inducing point shape
            check_shape([self.x_train, x_ind], [('N', 'D'), ('M', 'D')])

            x_ind = tf.convert_to_tensor(x_ind, dtype=self.dtype)
            
        else:
            ind_idx = np.random.choice(np.arange(self.x_train.shape[0]),
                                       size=(num_ind,),
                                       replace=False)

            x_ind = tf.convert_to_tensor(self.x_train.numpy()[ind_idx],
                                         dtype=self.dtype)
            
        return x_ind
    
        
    def add_training_data(self, x_train: tf.Tensor, y_train: tf.Tensor):
        """
        Adds data to the model's training data.

        :param x_train: new training inputs, shape (N, D)
        :param y_train: new training outputs, shape (N, 1)
        :return:
        """
        
        # Check x_train and y_train have compatible shapes
        check_shape([self.x_train, x_train, self.y_train, y_train],
                    [('N1', 'D'), ('N2', 'D'), ('N1', '1'), ('N2', '1')])
        
        # Concatenate observed data and new data
        self.x_train = tf.concat([self.x_train, x_train], axis=0)
        self.y_train = tf.concat([self.y_train, y_train], axis=0)
    
    
    @property
    def noise(self) -> tf.Tensor:
        """
        The standard deviation of the Gaussian noise of the GP model.
        :return:
        """
        return tf.math.exp(self.log_noise)

    def post_pred(self, x_pred: tf.Tensor):
        
        # Number of training points
        N = self.y_train.shape[0]
        M = self.x_ind.shape[0]
        K = x_pred.shape[0]
        
        # Compute covariance terms
        K_ind_ind = self.cov(self.x_ind, self.x_ind, epsilon=1e-9)
        K_train_ind = self.cov(self.x_train, self.x_ind)
        K_ind_train = self.cov(self.x_ind, self.x_train)
        K_pred_ind = self.cov(x_pred, self.x_ind)
        K_ind_pred = self.cov(self.x_ind, x_pred)
        K_pred_pred = self.cov(x_pred, x_pred)
        
        # Compute intermediate matrices using Cholesky for numerical stability
        L, U, A, B, B_chol = self.compute_intermediate_matrices(K_ind_ind,
                                                                K_ind_train)
        
        # Compute mean
        diff = self.y_train # - self.mean(self.x_train)[:, None]
        beta = tf.linalg.cholesky_solve(B_chol, tf.matmul(U, diff))
        beta = tf.linalg.triangular_solve(tf.transpose(L, (1, 0)),
                                          beta,
                                          lower=False)
        mean = tf.matmul(K_pred_ind / self.noise ** 2, beta)[:, 0]
        
        # Compute covariance
        C = tf.linalg.triangular_solve(L, K_ind_pred)
        D = tf.linalg.triangular_solve(B_chol, C)
        
        cov = K_pred_pred + self.noise ** 2 * tf.eye(K, dtype=self.dtype)
        cov = cov - tf.matmul(C, C, transpose_a=True)
        cov = cov + tf.matmul(D, D, transpose_a=True)
        
        return mean, cov
        
    def free_energy(self):
        
        # Number of training points and inducing points
        N = self.y_train.shape[0]
        M = self.x_ind.shape[0]
        
        # Compute covariance terms
        K_ind_ind = self.cov(self.x_ind, self.x_ind, epsilon=1e-9)
        K_train_ind = self.cov(self.x_train, self.x_ind)
        K_ind_train = self.cov(self.x_ind, self.x_train)
        K_train_train_diag = self.cov(self.x_train, self.x_train, diag=True)
        
        # Compute intermediate matrices using Cholesky for numerical stability
        L, U, A, B, B_chol = self.compute_intermediate_matrices(K_ind_ind,
                                                                K_ind_train)
        
        # Compute log-normalising constant of the matrix
        log_pi = - N / 2 * tf.math.log(tf.constant(2 * np.pi, dtype=self.dtype))
        log_det_B = - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(B_chol)))
        log_det_noise = - N / 2 * tf.math.log(self.noise ** 2)
        
        # Log of determinant of normalising term
        log_det = log_pi + log_det_B + log_det_noise       
        
        # Compute quadratic form
        diff = self.y_train - self.mean(self.x_train)[:, None]
        c = tf.linalg.triangular_solve(B_chol, tf.matmul(A, diff), lower=True) / self.noise
        quad = - 0.5 * tf.reduce_sum((diff / self.noise) ** 2)
        quad = quad + 0.5 * tf.reduce_sum(c ** 2)
        
        # Compute trace term
        trace = - 0.5 * tf.reduce_sum(K_train_train_diag) / self.noise ** 2
        trace = trace + 0.5 * tf.linalg.trace(tf.matmul(A, A, transpose_b=True))
        
        free_energy = (log_det + quad + trace) / N
        
        return free_energy
        

    def sample_posterior(self, num_features):
        
        # Number of inducing points
        M = self.x_ind.shape[0]
        
        # Draw a sample function from the RFF prior - rff_prior is a function
        rff_prior = self.cov.sample_rff(num_features)
        
        K_ind_ind = self.cov(self.x_ind, self.x_ind, epsilon=1e-9)
        K_train_ind = self.cov(self.x_train, self.x_ind)
        K_ind_train = self.cov(self.x_ind, self.x_train)
        
        # Compute intermediate matrices using Cholesky for numerical stability
        L, U, A, B, B_chol = self.compute_intermediate_matrices(K_ind_ind,
                                                                K_ind_train)
        
        # Compute mean of VFE posterior over inducing values
        u_mean = self.noise ** -2 * \
                 L @ tf.linalg.cholesky_solve(B_chol, U @ self.y_train)
        
        # Compute Cholesky of covariance of VFE posterior over inducing values
        u_cov_chol = tf.linalg.triangular_solve(B_chol, tf.transpose(L, (1, 0)))
        
        rand = tf.random.normal((M, 1), dtype=self.dtype)
        
        u = u_mean[:, 0] + tf.matmul(u_cov_chol, rand, transpose_a=True)[:, 0]
        v = tf.linalg.cholesky_solve(L, (u - rff_prior(self.x_ind))[:, None])[:, 0]
        
        def post_sample(x, add_noise):
            
            # Check input shape
            check_shape([self.x_train, x], [(-1, 'D'), (-1, 'D')])
            
            # Covariance between inputs and inducing points
            K_x_ind = self.cov(x, self.x_ind)
            
            sample = rff_prior(x) + tf.linalg.matvec(K_x_ind, v)
                     
            if add_noise:
                sample = sample + tf.random.normal(mean=0.,
                                                   stddev=self.noise,
                                                   shape=sample.shape)
            return sample
        
        return post_sample
    
    
    def compute_intermediate_matrices(self, K_ind_ind, K_ind_train):
        
        # Number of inducing points
        M = self.x_ind.shape[0]
        
        # Compute the following matrices, in a numerically stable way
        # L = chol(K_ind_ind)
        # U = iL K_ind_train
        # A = U / noise
        # B = I + A A.T
        L = tf.linalg.cholesky(K_ind_ind)
        U = tf.linalg.triangular_solve(L, K_ind_train, lower=True)
        A = U / self.noise
        B = tf.eye(M, dtype=self.dtype) + tf.matmul(A, A, transpose_b=True)
        B_chol = tf.linalg.cholesky(B)
        
        return L, U, A, B, B_chol