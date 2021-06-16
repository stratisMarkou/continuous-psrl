import argparse

from cpsrl.agents.agent import RandomAgent
from cpsrl.environments.custom.continuous_mountaincar import MountainCar

parser = argparse.ArgumentParser()

# Environment kind
parser.add_argument("env_name",type=str, default="MountainCar", help="Environment name")

# Agent kind
parser.add_argument("agent", type=str, default="GPPSRL", help="Agent name.")

# Number of episodes to observe
parser.add_argument("num_episodes", type=int, help="Number of episodes to play for.")

# # Environment parameters (for dynamics and rewards)
# parser.add_argument("--env_params")
#
# # Agent parameters
# parser.add_argument("--agent_params")

args = parser.parse_args()

# =============================================================================
# Helper for training one episode
# =============================================================================

def play_episode(agent, environment):
    """Plays an episode with the current policy."""
    state = environment.reset()
    cumulative_reward = 0
    episode = []

    while True:
        action = agent.act(state)
        next_state, reward = environment.step(action)

        cumulative_reward += reward
        episode.append((state, action, next_state, reward))
        state = next_state

        if environment.done: break

    return cumulative_reward, episode


# =============================================================================
# Training loop
# =============================================================================

env = MountainCar()
agent = RandomAgent(action_space=env.action_space)

# For each episode
for i in range(args.num_episodes):

    # Play episode
    cumulative_reward, episode = play_episode(agent=agent, environment=env)

    # Observe episode
    agent.observe(episode)

    # Train agent models and/or policy
    agent.update()

    print(f'Episode {i}: cum. reward {cumulative_reward:.3f}')

# =============================================================================
# Storing agents
# =============================================================================
