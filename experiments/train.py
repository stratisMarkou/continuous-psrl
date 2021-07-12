import os
import json
import pickle
import argparse
import sys

import numpy as np
import tensorflow as tf

from cpsrl.agents import RandomAgent, GroundTruthModelAgent, GPPSRLAgent
from cpsrl.models.mean import ConstantMean, LinearMean
from cpsrl.models.covariance import EQ
from cpsrl.models.gp import VFEGP, VFEGPStack
from cpsrl.models.initial_distributions import IndependentGaussianMAPMean
from cpsrl.policies.policies import FCNPolicy
from cpsrl.environments import MountainCar
from cpsrl.train_utils import play_episode, eval_models, record_episode
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
                    choices=["Random", "GroundTruth", "GPPSRL"],
                    help="Agent name.")

parser.add_argument("num_episodes",
                    type=int,
                    help="Number of episodes to play for.")

parser.add_argument("--sub_sampling_factor",
                    type=int,
                    default=6,
                    help="Sub-sampling factor of the environment.")

parser.add_argument("--seed",
                    type=int,
                    default=0,
                    help="Random seed.")

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
parser.add_argument("--gamma",
                    type=float,
                    default=0.99,
                    help="Discount factor.")

parser.add_argument("--horizon",
                    type=int,
                    default=100,
                    help="Environment horizon.")

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
                    default=-2.0,
                    help="Log coefficients for dynamics model.")

parser.add_argument("--dyn_log_scale",
                    type=float,
                    default=-1.0,
                    help="Log scale for dynamics model.")

parser.add_argument("--dyn_log_noise",
                    type=float,
                    default=-3.0,
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
                    default=-3.0,
                    help="Log noise for rewards model.")

# Initial distribution parameters
parser.add_argument("--init_mu0",
                    type=float,
                    default=0.0,
                    help="Mean for initial distribution.")

parser.add_argument("--init_alpha0",
                    type=float,
                    default=1.0,
                    help="Mean for initial distribution.")

parser.add_argument("--init_beta0",
                    type=float,
                    default=0.001,
                    help="Mean for initial distribution.")

# Policy parameters
parser.add_argument("--hidden_size",
                    type=int,
                    default=64,
                    help="Hidden size for policy network.")

# Update/optimization parameters
parser.add_argument("--num_steps_dyn",
                    type=int,
                    default=1000,
                    help="Number of optimization steps for dynamics model.")

parser.add_argument("--learn_rate_dyn",
                    type=float,
                    default=1e-2,
                    help="Learning rate for optimizing dynamics model.")

parser.add_argument("--num_steps_rew",
                    type=int,
                    default=500,
                    help="Number of optimization steps for rewards model.")

parser.add_argument("--learn_rate_rew",
                    type=float,
                    default=1e-2,
                    help="Learning rate for optimizing rewards model.")

parser.add_argument("--num_rollouts",
                    type=int,
                    default=200,
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
                    default=1e-3,
                    help="Learning rate for optimizing policy.")

parser.add_argument("--max_ind",
                    type=int,
                    default=None,
                    help="Maximum number of inducing points.")

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
    env = MountainCar(dtype=dtype,
                      horizon=args.horizon,
                      sub_sampling_factor=args.sub_sampling_factor)

else:
    raise ValueError(f"Invalid environment: {args.env}")

# Set up models
S = len(env.state_space)
A = len(env.action_space)

# Set up agent
agent_rng = next(rng_seq)

if args.agent == "Random":
    agent = RandomAgent(action_space=env.action_space, dtype=dtype)

elif args.agent == "GroundTruth":

    dynamics_model, rewards_model = env.ground_truth_models()

    policy = FCNPolicy(hidden_sizes=[args.hidden_size] * 2,
                       state_space=env.state_space,
                       action_space=env.action_space,
                       trainable=True,
                       dtype=dtype)

    # Initial distribution
    init_mu0 = args.init_mu0 * tf.ones(shape=(S,), dtype=dtype)
    init_alpha0 = args.init_alpha0 * tf.ones(shape=(S,), dtype=dtype)
    init_beta0 = args.init_beta0 * tf.ones(shape=(S,), dtype=dtype)

    initial_distribution = IndependentGaussianMAPMean(
        state_space=env.state_space,
        mu0=init_mu0,
        alpha0=init_alpha0,
        beta0=init_beta0,
        trainable=True,
        dtype=dtype)

    update_params = {
        "num_rollouts": args.num_rollouts,
        "num_steps_policy": args.num_steps_policy,
        "learn_rate_policy": args.learn_rate_policy,
    }

    agent = GroundTruthModelAgent(action_space=env.action_space,
                                  horizon=env.horizon,
                                  gamma=args.gamma,
                                  dtype=dtype,
                                  policy=policy,
                                  dynamics_model=dynamics_model,
                                  rewards_model=rewards_model,
                                  initial_distribution=initial_distribution,
                                  update_params=update_params)

elif args.agent == "GPPSRL":

    # Dynamics models
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
                         x_ind=tf.random.uniform(shape=(1, S + A), dtype=dtype),
                         num_ind=None)
                   for i in range(S)]

    dynamics_model = VFEGPStack(vfe_gps=dyn_vfe_gps, dtype=dtype)

    # Reward model
    rew_mean = ConstantMean(input_dim=S,
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
                          x_ind=tf.random.uniform(shape=(1, S), dtype=dtype),
                          num_ind=None)

    # Policy
    policy = FCNPolicy(hidden_sizes=[args.hidden_size] * 2,
                       state_space=env.state_space,
                       action_space=env.action_space,
                       trainable=True,
                       dtype=dtype)

    # Initial distribution
    init_mu0 = args.init_mu0 * tf.ones(shape=(S,), dtype=dtype)
    init_alpha0 = args.init_alpha0 * tf.ones(shape=(S,), dtype=dtype)
    init_beta0 = args.init_beta0 * tf.ones(shape=(S,), dtype=dtype)

    initial_distribution = IndependentGaussianMAPMean(
        state_space=env.state_space,
        mu0=init_mu0,
        alpha0=init_alpha0,
        beta0=init_beta0,
        trainable=True,
        dtype=dtype)

    update_params = {
        "num_steps_dyn": args.num_steps_dyn,
        "learn_rate_dyn": args.learn_rate_dyn,
        "num_steps_rew": args.num_steps_rew,
        "learn_rate_rew": args.learn_rate_rew,
        "num_rollouts": args.num_rollouts,
        "num_features": args.num_features,
        "num_steps_policy": args.num_steps_policy,
        "learn_rate_policy": args.learn_rate_policy,
    }

    agent = GPPSRLAgent(action_space=env.action_space,
                        horizon=env.horizon,
                        gamma=args.gamma,
                        initial_distribution=initial_distribution,
                        dynamics_model=dynamics_model,
                        rewards_model=rewards_model,
                        policy=policy,
                        update_params=update_params,
                        max_ind=args.max_ind,
                        dtype=dtype)


# =============================================================================
# Training loop
# =============================================================================

plot_dir = os.path.join(args.results_dir, exp_name)
os.makedirs(plot_dir, exist_ok=True)

# For each episode
for i in range(args.num_episodes):

    # Play episode
    cumulative_reward, episode = play_episode(agent=agent, environment=env)

    print(f'\nEpisode {i} | Return: {cumulative_reward:.3f}\n')

    # Observe episode
    agent.observe(episode)

    # Train agent models and/or policy
    info_dict = agent.update()

    if args.agent in ["GroundTruth", "GPPSRL"]:

        # Analyze optimization
        rollouts = info_dict["rollout"]
        plot_freq = np.maximum(1, len(rollouts) // 10)
        opt_steps = np.arange(len(rollouts))

        for t, rollout in zip(opt_steps[::plot_freq], rollouts[::plot_freq]):
            plot_file = f"opt_ep-{i}_iter-{t}.svg"
            save_dir = os.path.join(plot_dir, plot_file)
            env.plot_trajectories(rollout, save_dir=save_dir)

        # Evaluate model performance
        for on_policy in [True, False]:

            if args.agent == "GPPSRL":

                print(f"Evaluating models... (on_policy={on_policy})")

                eval_models(agent=agent,
                            environment=env,
                            num_episodes=10,
                            on_policy=on_policy)

    _, ground_truth = play_episode(agent=agent, environment=env)

    video_file = f"exp_ep-{i}.mp4"
    record_episode(env_name=args.env,
                   episode=ground_truth,
                   save_dir=os.path.join(plot_dir, video_file))

    plot_file = f"exp_ep-{i}.svg"
    env.plot_trajectories([episode],
                          save_dir=os.path.join(plot_dir, plot_file),
                          ground_truth=ground_truth)

    # Save episode
    with open(os.path.join(args.data_dir, f"{exp_name}_ep-{i}.pkl"),
              mode="wb") as f:
        pickle.dump({"Episode": i, "Transitions": episode}, f)

    # Save aggregated results of the episode
    with open(os.path.join(args.results_dir, f"{exp_name}.txt"), mode="a") as f:
        f.write(json.dumps({"Episode": i, "Return": cumulative_reward}))
        f.write("\n")

# =============================================================================
# Storing agents
# =============================================================================

# TODO: implement model saving/loading
