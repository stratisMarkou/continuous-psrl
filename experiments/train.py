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
from cpsrl.train_utils import play_episode, eval_models
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
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
parser.add_argument("--horizon", type=int, default=None, help="Env horizon.")

# Dynamics model parameters
parser.add_argument("--dyn_trainable_mean",
                    dest="dyn_trainable_mean",
                    action="store_true",
                    help="Optimize mean of the dynamics model")

parser.add_argument("--dyn_trainable_cov",
                    dest="dyn_trainable_cov",
                    action="store_true",
                    help="Optimize covariance of the dynamics model")

parser.add_argument("--dyn_trainable_inducing",
                    dest="dyn_trainable_inducing",
                    action="store_true",
                    help="Optimize inducing points of the dynamics model")

parser.add_argument("--dyn_trainable_noise",
                    dest="dyn_trainable_noise",
                    action="store_true",
                    help="Optimize noise of the dynamics model")

parser.add_argument("--dyn_log_coeff",
                    type=float,
                    default=-1.0,
                    help="Log coefficients for dynamics model.")

parser.add_argument("--dyn_log_scale",
                    type=float,
                    default=-1.0,
                    help="Log scale for dynamics model.")

parser.add_argument("--dyn_log_noise",
                    type=float,
                    default=-2.0,
                    help="Log noise for dynamics model.")

# Reward model parameters
parser.add_argument("--rew_trainable_mean",
                    dest="rew_trainable_mean",
                    action="store_true",
                    help="Optimize mean of the rewards model")

parser.add_argument("--rew_trainable_cov",
                    dest="rew_trainable_cov",
                    action="store_true",
                    help="Optimize covariance of the rewards model")

parser.add_argument("--rew_trainable_inducing",
                    dest="rew_trainable_inducing",
                    action="store_true",
                    help="Optimize inducing points of the rewards model")

parser.add_argument("--rew_trainable_noise",
                    dest="rew_trainable_noise",
                    action="store_true",
                    help="Optimize noise of the rewards model")

parser.add_argument("--rew_log_coeff",
                    type=float,
                    default=-1.0,
                    help="Log coefficients for rewards model.")

parser.add_argument("--rew_log_scale",
                    type=float,
                    default=-1.0,
                    help="Log scale for rewards model.")

parser.add_argument("--rew_log_noise",
                    type=float,
                    default=-2.0,
                    help="Log noise for rewards model.")

# Initial distribution parameters
parser.add_argument("--init_trainable",
                    dest="init_trainable",
                    action="store_true",
                    help="Optimize initial distribution.")

parser.add_argument("--init_mean",
                    type=float,
                    default=0.0,
                    help="Mean for initial distribution.")

parser.add_argument("--init_scale",
                    type=float,
                    default=1.0,
                    help="Scale for initial distribution.")

# Policy parameters
parser.add_argument("--hidden_size",
                    type=int,
                    default=64,
                    help="Hidden size for policy network.")

parser.add_argument("--trainable_policy",
                    dest="trainable_policy",
                    action="store_true",
                    help="Optimize policy.")

# Update/optimization parameters
parser.add_argument("--num_steps_dyn",
                    type=int,
                    default=1000,
                    help="Number of optimization steps for dynamics model.")

parser.add_argument("--learn_rate_dyn",
                    type=float,
                    default=1e-2,
                    help="Learning rate for optimizing dynamics model.")

parser.add_argument("--num_ind_dyn",
                    type=int,
                    default=None,
                    help="Number of inducing points for dynamics model. "
                         "Determined automatically if set to None.")

parser.add_argument("--num_steps_rew",
                    type=int,
                    default=1000,
                    help="Number of optimization steps for rewards model.")

parser.add_argument("--learn_rate_rew",
                    type=float,
                    default=1e-2,
                    help="Learning rate for optimizing rewards model.")

parser.add_argument("--num_ind_rew",
                    type=int,
                    default=None,
                    help="Number of inducing points for rewards model. "
                         "Determined automatically if set to None.")

parser.add_argument("--num_rollouts",
                    type=int,
                    default=300,
                    help="Number of rollouts to simulate.")

parser.add_argument("--num_features",
                    type=int,
                    default=200,
                    help="Number of Fourier features for posterior samples.")

parser.add_argument("--num_steps_policy",
                    type=int,
                    default=100,
                    help="Number of optimization steps for policy.")

parser.add_argument("--learn_rate_policy",
                    type=float,
                    default=3e-4,
                    help="Learning rate for optimizing policy.")

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
    env = MountainCar(rng=env_rng, horizon=args.horizon)
elif args.env == "CartPole":
    env = CartPole(rng=env_rng)
else:
    raise ValueError(f"Invalid environment: {args.env}")

# Set up models
S, A = len(env.state_space), len(env.action_space)

# 1. Dynamics models
dyn_means = [LinearMean(input_dim=S + A,
                        trainable=args.dyn_trainable_mean,
                        dtype=dtype)
             for i in range(S)]

dyn_covs = [EQ(log_coeff=args.dyn_log_coeff,
               log_scales=(S + A) * [args.dyn_log_scale],
               trainable=args.dyn_trainable_cov,
               dtype=dtype)
            for _ in range(S)]

dyn_vfe_gps = [VFEGP(mean=dyn_means[i],
                     cov=dyn_covs[i],
                     input_dim=S + A,
                     trainable_inducing=args.dyn_trainable_inducing,
                     log_noise=args.dyn_log_noise,
                     trainable_noise=args.dyn_trainable_noise,
                     dtype=dtype,
                     x_ind=tf.zeros((1, S + A), dtype=dtype),
                     num_ind=None)
               for i in range(S)]

dynamics_model = VFEGPStack(vfe_gps=dyn_vfe_gps, dtype=dtype)

# 2. Reward model
rew_mean = LinearMean(input_dim=S,
                      trainable=args.rew_trainable_mean,
                      dtype=dtype)

rew_cov = EQ(log_coeff=args.rew_log_coeff,
             log_scales=S * [args.rew_log_scale],
             trainable=args.rew_trainable_cov,
             dtype=dtype)

rewards_model = VFEGP(mean=rew_mean,
                      cov=rew_cov,
                      input_dim=S,
                      trainable_inducing=args.rew_trainable_inducing,
                      log_noise=args.rew_log_noise,
                      trainable_noise=args.rew_trainable_noise,
                      dtype=dtype,
                      x_ind=tf.zeros((1, S), dtype=dtype),
                      num_ind=None)

policy = FCNPolicy(hidden_sizes=[args.hidden_size] * 2,
                   state_space=env.state_space,
                   action_space=env.action_space,
                   trainable=args.trainable_policy,
                   dtype=dtype)

# 3. Initial distribution
init_means = args.init_mean * tf.ones(shape=(S,), dtype=dtype)
init_scales = args.init_scale * tf.ones(shape=(S,), dtype=dtype)

initial_distribution = IndependentGaussian(state_space=env.state_space,
                                           mean=init_means,
                                           scales=init_scales,
                                           trainable=args.init_trainable,
                                           dtype=dtype)

# TODO: choose number of inducing points dynamically
num_ind_dyn = args.num_ind_dyn or 100
num_ind_rew = args.num_ind_rew or 100

update_params = {
    "num_steps_dyn": args.num_steps_dyn,
    "learn_rate_dyn": args.learn_rate_dyn,
    "num_ind_dyn": num_ind_dyn,
    "num_steps_rew": args.num_steps_rew,
    "learn_rate_rew": args.learn_rate_rew,
    "num_ind_rew": num_ind_rew,
    "num_rollouts": args.num_rollouts,
    "num_features": args.num_features,
    "num_steps_policy": args.num_steps_policy,
    "learn_rate_policy": args.learn_rate_policy,
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

    if isinstance(agent, GPPSRLAgent):
        print()
        for on_policy in [True, False]:
            print(f"Evaluating models... (on_policy={on_policy})")
            eval_models(agent=agent,
                        environment=env,
                        dtype=dtype,
                        num_episodes=10,
                        on_policy=on_policy)

    cumulative_reward = cumulative_reward
    print()
    print(f'Episode {i} | Return: {cumulative_reward:.3f}')
    print()

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
