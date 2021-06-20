from cpsrl.models.mean import LinearMean
from cpsrl.models.covariance import EQ
from cpsrl.models.gp import VFEGP, VFEGPStack
from cpsrl.models.initial_distributions import IndependentGaussian
from cpsrl.agents.gppsrl import GPPSRLAgent
from cpsrl.policies.policies import FCNPolicy

import tensorflow as tf

dtype = tf.float64

# Environment constants
S = 2
A = 1
N = 500
R = 10
horizon = 20
gamma = 0.9

# Draw random data
s = tf.random.uniform(shape=(N, S), dtype=dtype)
a = tf.random.uniform(shape=(N, A), dtype=dtype)
s_ = tf.random.uniform(shape=(N, S), dtype=dtype)
r = tf.random.normal(shape=(N, 1), dtype=dtype)
sa = tf.concat([s, a], axis=1)
s0 = tf.random.uniform(shape=(R, S), dtype=dtype)

# Trainable settings
dyn_trainable_mean = True
dyn_trainable_cov = False
dyn_trainable_inducing = False
dyn_trainable_noise = True

# Covariance parameters
dyn_log_coeff = -2.
dyn_log_scales = (S + A) * [-2.]
dyn_log_noise = -2.
dyn_num_ind = 10

# Initialise means
dyn_means = [LinearMean(input_dim=S+A,
                        trainable=dyn_trainable_mean,
                        dtype=dtype)
             for i in range(S)]

dyn_covs = [EQ(log_coeff=dyn_log_coeff,
               log_scales=dyn_log_scales,
               trainable=dyn_trainable_cov,
               dtype=dtype)
            for _ in range(S)]

dyn_vfe_gps = [VFEGP(mean=dyn_means[i],
                     cov=dyn_covs[i],
                     input_dim=S+A,
                     x_train=sa,
                     y_train=s_[:, i:i+1],
                     trainable_inducing=dyn_trainable_inducing,
                     log_noise=dyn_log_noise,
                     trainable_noise=dyn_trainable_noise,
                     dtype=dtype,
                     x_ind=None,
                     num_ind=dyn_num_ind)
               for i in range(S)]

dyn_vfe_stack = VFEGPStack(vfe_gps=dyn_vfe_gps,
                           dtype=dtype)

# Trainable settings
rew_trainable_mean = True
rew_trainable_cov = False
rew_trainable_inducing = False
rew_trainable_noise = True

# Covariance parameters
rew_log_coeff = -2.
rew_log_scales = S * [-2.]
rew_log_noise = -2.
rew_num_ind = 10

# Initialise means
rew_mean = LinearMean(input_dim=S,
                      trainable=rew_trainable_mean,
                      dtype=dtype)

rew_cov = EQ(log_coeff=rew_log_coeff,
             log_scales=rew_log_scales,
             trainable=rew_trainable_cov,
             dtype=dtype)

rew_vfe_gp = VFEGP(mean=rew_mean,
                   cov=rew_cov,
                   input_dim=S,
                   x_train=s,
                   y_train=r,
                   trainable_inducing=rew_trainable_inducing,
                   log_noise=rew_log_noise,
                   trainable_noise=rew_trainable_noise,
                   dtype=dtype,
                   x_ind=None,
                   num_ind=rew_num_ind)

state_space = S * [(-2, 2)]
action_space = A * [(-2, 2)]
hidden_sizes = [64, 64]
trainable_policy = True

policy = FCNPolicy(hidden_sizes=hidden_sizes,
                   state_space=state_space,
                   action_space=action_space,
                   trainable=trainable_policy,
                   dtype=dtype)

# Initial distribution parameters
init_mean = -0.5 * tf.ones(shape=(S,), dtype=dtype)
init_scales = tf.ones(shape=(S,), dtype=dtype)
init_trainable = False

initial_distribution = IndependentGaussian(state_space=state_space,
                                           mean=init_mean,
                                           scales=init_scales,
                                           trainable=init_trainable,
                                           dtype=dtype)

update_params = {
    "num_steps_dyn" : 10,
    "learn_rate_dyn" : 1e-3,
    "num_steps_rew" : 10,
    "learn_rate_rew" : 1e-3,
    "num_rollouts" : 20,
    "num_features" : 200,
    "num_steps_policy" : 100,
    "learn_rate_policy" : 1e-3,
    "num_ind_dyn" : 2,
    "num_ind_rew" : 2
}

agent = GPPSRLAgent(action_space=action_space,
                    horizon=horizon,
                    gamma=gamma,
                    update_params=update_params,
                    initial_distribution=initial_distribution,
                    dynamics_model=dyn_vfe_stack,
                    rewards_model=rew_vfe_gp,
                    policy=policy,
                    dtype=dtype)

agent.update()

