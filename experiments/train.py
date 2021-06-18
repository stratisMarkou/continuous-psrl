import os
import json
import pickle
import argparse
import sys

from cpsrl.agents import RandomAgent, GPPSRLAgent
from cpsrl.environments import MountainCar, CartPole
from cpsrl.train_utils import play_episode
from cpsrl.helpers import set_seed, Logger

parser = argparse.ArgumentParser()

# Base arguments
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

# # Environment parameters (for dynamics and rewards)
# parser.add_argument("--env_params")
#
# # Agent parameters
# parser.add_argument("--agent_params")


# =============================================================================
# Setup
# =============================================================================

args = parser.parse_args()
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

# Set up agent
agent_rng = next(rng_seq)
if args.agent == "GPPSRL":
    agent = GPPSRLAgent(action_space=env.action_space)
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
