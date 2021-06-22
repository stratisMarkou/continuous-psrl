import os
import json
import pickle
import argparse
import sys

import tensorflow as tf

from cpsrl.agents import RandomAgent, GPPSRLAgent
from cpsrl.models.mean import LinearMean
from cpsrl.models.covariance import EQ
from cpsrl.models.gp import VFEGP, VFEGPStack
from cpsrl.models.initial_distributions import IndependentGaussian
from cpsrl.policies.policies import FCNPolicy
from cpsrl.environments import MountainCar, CartPole
from cpsrl.train_utils import play_episode
from cpsrl.helpers import set_seed, Logger

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("env",
                    type=str,
                    choices=["MountainCar", "CartPole"],
                    default="MountainCar",
                    help="Environment name")

parser.add_argument("agent",
                    type=str,
                    choices=["GPPSRL", "Random"],
                    default="GPPSRL",
                    help="Agent name.")

parser.add_argument("num_episodes", type=int, help="Number of episodes to play for.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")

parser.add_argument("--log_dir",
                    type=str,
                    default="logs",
                    help="Directory for storing logs.")
parser.add_argument("--data_dir",
                    type=str,
                    default="data",
                    help="Directory for storing trajectory data.")
parser.add_argument("--results_dir",
                    type=str,
                    default="results",
                    help="Directory for storing results.")
parser.add_argument("--model_dir",
                    type=str,
                    default="models",
                    help="Directory for storing models.")

# Environment parameters (for dynamics and rewards)
parser.add_argument("--gamma", type=int, default=0.99, help="Discount factor.")

# # Agent parameters
# parser.add_argument("--agent_params")


# =============================================================================
# Setup
# =============================================================================

args = parser.parse_args()
dtype = tf.float64
rng_seq = set_seed(args.seed)
exp_name = f"{args.agent}_{args.env}_{args.seed}"

for dir in [args.log_dir, args.results_dir, args.data_dir, args.model_dir]:
    os.makedirs(dir, exist_ok=True)

logger = Logger(directory=args.log_dir, exp_name=exp_name)
sys.stdout, sys.stderr = logger, logger
print(vars(args))

# Set up env
env_rng = next(rng_seq)
if args.env == "MountainCar":
    env = MountainCar(rng=env_rng)
elif args.env == "CartPole":
    env = CartPole(rng=env_rng)
else:
    raise ValueError(f"Invalid environment: {args.env}")

# Set up models
S, A = len(env.state_space), len(env.action_space)
dyn_trainable_mean = True
dyn_trainable_cov = True
dyn_trainable_inducing = True
dyn_trainable_noise = True

dyn_log_coeff = 0.
dyn_log_scales = (S + A) * [0.]
dyn_log_noise = -1.

# 1. Dynamics models

dyn_means = [LinearMean(input_dim=S + A,
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
                     input_dim=S + A,
                     trainable_inducing=dyn_trainable_inducing,
                     log_noise=dyn_log_noise,
                     trainable_noise=dyn_trainable_noise,
                     dtype=dtype,
                     x_ind=tf.zeros((1, S + A), dtype=dtype),
                     num_ind=None)
               for i in range(S)]

dynamics_model = VFEGPStack(vfe_gps=dyn_vfe_gps, dtype=dtype)

# 2. Reward model
rew_trainable_mean = True
rew_trainable_cov = False
rew_trainable_inducing = False
rew_trainable_noise = True

rew_log_coeff = 0.
rew_log_scales = S * [0.]
rew_log_noise = -1.

rew_mean = LinearMean(input_dim=S,
                      trainable=rew_trainable_mean,
                      dtype=dtype)

rew_cov = EQ(log_coeff=rew_log_coeff,
             log_scales=rew_log_scales,
             trainable=rew_trainable_cov,
             dtype=dtype)

rewards_model = VFEGP(mean=rew_mean,
                      cov=rew_cov,
                      input_dim=S,
                      trainable_inducing=rew_trainable_inducing,
                      log_noise=rew_log_noise,
                      trainable_noise=rew_trainable_noise,
                      dtype=dtype,
                      x_ind=tf.zeros((1, S), dtype=dtype),
                      num_ind=None)

hidden_sizes = [64, 64]
trainable_policy = True

policy = FCNPolicy(hidden_sizes=hidden_sizes,
                   state_space=env.state_space,
                   action_space=env.action_space,
                   trainable=trainable_policy,
                   dtype=dtype)

# Initial distribution parameters
init_mean = tf.zeros(shape=(S,), dtype=dtype)
init_scales = tf.ones(shape=(S,), dtype=dtype)
init_trainable = False

initial_distribution = IndependentGaussian(state_space=env.state_space,
                                           mean=init_mean,
                                           scales=init_scales,
                                           trainable=init_trainable,
                                           dtype=dtype)

update_params = {
    "num_steps_dyn": 10,
    "learn_rate_dyn": 1e-3,
    "num_steps_rew": 10,
    "learn_rate_rew": 1e-3,
    "num_rollouts": 20,
    "num_features": 200,
    "num_steps_policy": 100,
    "learn_rate_policy": 1e-3,
    "num_ind_dyn": 2,
    "num_ind_rew": 2
}

# Set up agent
agent_rng = next(rng_seq)
if args.agent == "GPPSRL":
    agent = GPPSRLAgent(action_space=env.action_space,
                        horizon=env.horizon,
                        gamma=args.gamma,
                        initial_distribution=initial_distribution,
                        dynamics_model=dynamics_model,
                        rewards_model=rewards_model,
                        policy=policy,
                        update_params=update_params,
                        dtype=dtype)
elif args.agent == "Random":
    agent = RandomAgent(action_space=env.action_space, rng=agent_rng)
else:
    raise ValueError(f"Invalid agent: {args.agent}")

# =============================================================================
# Training loop
# =============================================================================

# For each episode
for i in range(args.num_episodes):
    # Play episode
    cumulative_reward, episode = play_episode(agent=agent, environment=env)

    # Observe episode
    agent.observe(episode)

    # Train agent models and/or policy
    agent.update()

    print(f'Episode {i} | Return: {cumulative_reward:.3f}')

    # Save episode
    with open(os.path.join(args.data_dir, exp_name + f"_ep-{i}.pkl"), mode="wb") as f:
        pickle.dump({"Episode": i, "Transitions": episode}, f)

    # Save aggregated results of the episode
    with open(os.path.join(args.results_dir, exp_name + ".txt"), mode="a") as f:
        f.write(json.dumps({"Episode": i, "Return": cumulative_reward}))
        f.write("\n")

# =============================================================================
# Storing agents
# =============================================================================

# TODO: implement model saving/loading
