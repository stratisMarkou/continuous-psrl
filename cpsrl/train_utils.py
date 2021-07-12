from __future__ import annotations

from typing import Tuple, List, TYPE_CHECKING

import numpy as np
import tensorflow as tf

from cpsrl.helpers import Transition, convert_episode_to_tensors

if TYPE_CHECKING:  # avoid circular imports
    from cpsrl.agents import Agent, GPPSRLAgent
    from cpsrl.environments import Environment

# =============================================================================
# Helper for rollouts
# =============================================================================


def play_episode(agent: Agent, environment: Environment) \
        -> Tuple[float, List[Transition]]:
    """Plays an episode with the current policy."""

    state = environment.reset()
    cumulative_reward = 0.0
    episode = []

    while True:
        action = agent.act(state)
        next_state, reward = environment.step(action)

        cumulative_reward += reward.numpy().item()
        episode.append(Transition(state, action, reward, next_state))
        state = next_state

        if environment.done: break

    return cumulative_reward, episode


def play_random_episode(environment: Environment) \
        -> Tuple[float, List[Transition]]:
    """Plays an episode with random states and actions."""

    def get_random_sample(space, dtype):
        return tf.convert_to_tensor([np.random.uniform(low, high)
                                     for low, high in space],
                                    dtype=dtype)

    environment.reset()
    state = environment.state = get_random_sample(environment.state_space,
                                                  environment.dtype)

    cumulative_reward = 0.0
    episode = []

    while True:
        action = get_random_sample(environment.action_space,
                                   environment.dtype)
        next_state, reward = environment.step(action)

        cumulative_reward += reward.numpy().item()
        episode.append(Transition(state, action, reward, next_state))
        state = environment.state = get_random_sample(environment.state_space,
                                                      environment.dtype)

        if environment.done: break

    return cumulative_reward, episode


def eval_models(agent: GPPSRLAgent,
                environment: Environment,
                num_episodes: int,
                on_policy: bool):

    episodes = []
    for _ in range(num_episodes):
        if on_policy:  # collect data by following current agent
            _, episode = play_episode(agent=agent, environment=environment)
        else:  # randomly sample data from MDP
            _, episode = play_random_episode(environment=environment)

        episodes.append(episode)

    dynamics_errors, rewards_errors = [], []
    dynamics_lls, rewards_lls = [], []

    for episode in episodes:

        s, sa, s_, _, r = convert_episode_to_tensors(episode)

        ds = s_ - s
        dynamics_error = agent.dynamics_model.smse(sa, ds)
        dynamics_ll = agent.dynamics_model.pred_logprob(sa, ds)
        dynamics_errors.append(dynamics_error.numpy())
        dynamics_lls.append(dynamics_ll.numpy())

        rewards_error = agent.rewards_model.smse(s_, r)
        rewards_ll = agent.rewards_model.pred_logprob(s_, r)
        rewards_errors.append(rewards_error.numpy())
        rewards_lls.append(rewards_ll.numpy())

    dyn_err_mean = np.mean(dynamics_errors, axis=0).round(decimals=4)
    rew_err_mean = np.mean(rewards_errors, axis=0).round(decimals=4)
    dyn_ll_mean = np.mean(dynamics_lls, axis=0).round(decimals=4)
    rew_ll_mean = np.mean(rewards_lls, axis=0).round(decimals=4)

    print(f"Dynamics > SMSE: {dyn_err_mean}, LL: {dyn_ll_mean}")
    print(f"Rewards  > SMSE: {rew_err_mean}, LL: {rew_ll_mean}")


def ground_truth_trajectory(agent: GPPSRLAgent, environment: Environment) \
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

    _, episode = play_episode(agent=agent, environment=environment)

    s, sa, s_, _, r = convert_episode_to_tensors(episode)

    return s, sa, s_, r
