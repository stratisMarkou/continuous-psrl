from cpsrl.models.mean import LinearMean
from cpsrl.models.covariance import EQ
from cpsrl.models.gp import VFEGP, VFEGPStack
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
dyn_trainable_inducing = True
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
rew_trainable_inducing = True
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

agent = GPPSRLAgent(dynamics_model=dyn_vfe_stack,
                    rewards_model=rew_vfe_gp,
                    policy=policy,
                    dtype=dtype)


num_steps = 10
optimiser = tf.optimizers.Adam(1e-1)

for i in range(num_steps):
    with tf.GradientTape() as tape:

        tape.watch(dyn_vfe_stack.trainable_variables)
        dyn_vfe = dyn_vfe_stack.free_energy()

        loss = - dyn_vfe
        print(loss)

    dyn_grads = tape.gradient(loss,
                              dyn_vfe_stack.trainable_variables)
    optimiser.apply_gradients(zip(dyn_grads,
                                  dyn_vfe_stack.trainable_variables))

print("=======================================================================")

num_steps = 10
optimiser = tf.optimizers.Adam(1e-1)

for i in range(num_steps):
    with tf.GradientTape() as tape:

        tape.watch(rew_vfe_gp.trainable_variables)
        rew_vfe = rew_vfe_gp.free_energy()

        loss = - rew_vfe
        print(loss)

    rew_grads = tape.gradient(loss,
                              rew_vfe_gp.trainable_variables)
    optimiser.apply_gradients(zip(rew_grads,
                                  rew_vfe_gp.trainable_variables))


num_features = 200

dyn_sample = dyn_vfe_stack.sample_posterior(num_features=num_features)
rew_sample = rew_vfe_gp.sample_posterior(num_features=num_features)


rollout = agent.rollout(dynamics_sample=dyn_sample,
                        rewards_sample=rew_sample,
                        horizon=horizon,
                        gamma=gamma,
                        s0=s0)

cumulative_reward, states, actions, next_states, rewards = rollout

print(cumulative_reward)

