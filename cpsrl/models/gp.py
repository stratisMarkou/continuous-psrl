import tensorflow as tf
import numpy as np

from cpsrl.errors import ModelError
from cpsrl.helpers import check_shape


# ==============================================================================
# Variational Sparse Gaussian Process
# ==============================================================================


class VFEGPStack(tf.keras.Model):
    
    def __init__(self, vfe_gps, dtype, name='gp_stack', **kwargs):
        
        super().__init__(name=name, dtype=dtype, **kwargs)
        
        self.vfe_gps = []
        
        for vfe_gp in vfe_gps:
            self.vfe_gps.append(vfe_gp)
    
    
    def add_training_data(self, x_train, y_train):
        
        check_shape([x_train, y_train], [('N', '-1'), ('N', '-1')])
        
        for i, vfe_gp in enumerate(self.vfe_gps):
            vfe_gp.add_training_data(x_train, y_train[:, i:i+1])
    
    
    def sample_posterior(self, num_features):
        
        post_samples = [vfe_gp.sample_posterior(num_features=num_features) \
                        for vfe_gp in self.vfe_gps]
        
        def post_sample(x, add_noise):
            
            # Check shape of input against training
            check_shape([x, self.vfe_gps[0].x_train],
                        [('N1', 'D'), ('N2', 'D')])
            
            samples = [sample(x, add_noise) for sample in post_samples]
            
            return tf.stack(samples, axis=1)
            
        return post_sample
    
    
    def free_energy(self):
        return tf.reduce_sum([vfe_gp.free_energy() for vfe_gp in self.vfe_gps])
    

# ==============================================================================
# Variational Sparse Gaussian Process
# ==============================================================================


class VFEGP(tf.keras.Model):
    
    def __init__(self,
                 mean,
                 cov,
                 x_train,
                 y_train,
                 x_ind,
                 ind_fraction,
                 trainable_inducing,
                 log_noise,
                 trainable_noise,
                 dtype,
                 name='vfegp',
                 **kwargs):
        
        """
        
        Params:
            
            mean (cpsrl.models.mean) : mean function for the GP
            cov (cpsrl.models.covariance) : covariance function of the GP
            x_train (tf.tensor, np.array) : training inputs (N, D)
            y_train (tf.tensor, np.array) : training outputs (N)
            x_ind (tf.tensor, np.array) : x_ind
        """
        
        super().__init__(name=name, dtype=dtype, **kwargs)
        
        # Check x_train and y_train have compatible shapes
        check_shape([x_train, y_train], [('N', 'D'), ('N', '1')])
        
        # Set training data and inducing point initialisation
        self.x_train = tf.zeros(shape=(0, x_train.shape[1]), dtype=dtype)
        self.y_train = tf.zeros(shape=(0, 1), dtype=dtype)
        
        self.add_training_data(x_train, y_train)
        
        # Initialise inducing points
        self.x_ind = self.init_inducing(x_ind, ind_fraction)
        self.x_ind = tf.Variable(self.x_ind, trainable=trainable_inducing)
        
        # Set mean and covariance functions
        self.mean = mean
        self.cov = cov
    
        # Set log of noise parameter
        self.log_noise = tf.convert_to_tensor(log_noise, dtype=dtype)
        self.log_noise = tf.Variable(self.log_noise, trainable=trainable_noise)
        
    
    def init_inducing(self, x_ind, ind_fraction):
        
        assert ((x_ind is not None) and (ind_fraction is None)) or \
               ((x_ind is None) and (ind_fraction is not None))
        
        # Set inducing points either to initial locations or on training data
        if x_ind is not None:
            x_ind = tf.convert_to_tensor(x_ind, dtype=self.dtype)
            
        else:
            num_train = self.x_train.shape[0]
            num_inducing = int(ind_fraction * num_train + 0.5)
            
            ind_idx = np.random.choice(np.arange(num_train),
                                       size=(num_inducing,),
                                       replace=False)
            
            x_ind = tf.convert_to_tensor(self.x_train.numpy()[ind_idx],
                                         dtype=self.dtype)
            
        return x_ind
    
        
    def add_training_data(self, x_train, y_train):
        
        # Check x_train and y_train have compatible shapes
        check_shape([self.x_train, x_train, self.y_train, y_train],
                    [('N1', 'D'), ('N2', 'D'), ('N1', '1'), ('N2', '1')])
        
        # Concatenate observed data and new data
        self.x_train = tf.concat([self.x_train, x_train], axis=0)
        self.y_train = tf.concat([self.y_train, y_train], axis=0)
    
    
    @property
    def noise(self):
        return tf.math.exp(self.log_noise)
        
        
    def post_pred(self, x_pred):
        
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