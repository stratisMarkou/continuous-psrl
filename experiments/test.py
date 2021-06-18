import argparse

from cpsrl.agents import Agent, RandomAgent
from cpsrl.environments import MountainCar, CartPole
from cpsrl.helpers import set_seed
from cpsrl.train_utils import play_episode

parser = argparse.ArgumentParser()

# Base arguments
parser.add_argument("model_path",
                    type=str,
                    help="Path to saved model.")
parser.add_argument("env",
                    type=str,
                    choices=["MountainCar", "CartPole"],
                    default="MountainCar",
                    help="Environment name")

parser.add_argument("num_episodes", type=int, help="Number of episodes to play for.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")


def load_model(model_path: str) -> Agent:
    raise NotImplementedError  # TODO: implement


if __name__ == "__main__":
    args = parser.parse_args()
    rng_seq = set_seed(args.seed)

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
    #  agent = load_model(args.model_path)
    agent = RandomAgent(action_space=env.action_space, rng=agent_rng)

    # For each episode
    trajectories = []
    for i in range(args.num_episodes):
        # Play episode
        cumulative_reward, episode = play_episode(agent=agent, environment=env)

        # Observe episode
        agent.observe(episode)

        # Train agent models and/or policy
        agent.update()

        trajectories.append(episode)
        print(f'Episode {i} | Return: {cumulative_reward:.3f}')

    env.plot_trajectories(trajectories, save_dir="test.jpg")
