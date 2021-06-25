from __future__ import annotations

from typing import Tuple, List, Optional, TYPE_CHECKING

import numpy as np
import tensorflow as tf

from cpsrl.helpers import Transition, convert_episode_to_tensors

if TYPE_CHECKING:  # avoid circular imports
    from cpsrl.agents import Agent
    from cpsrl.models import VFEGP, VFEGPStack
    from cpsrl.environments import Environment

# =============================================================================
# Helper for rollouts
# =============================================================================


def play_episodes(agent: Agent,
                  environment: Environment,
                  num_episodes: Optional[int] = None,
                  init_states: Optional[List[np.ndarray]] = None) \
        -> List[List[Transition]]:

        if ((num_episodes is None and init_states is None)
                or (num_episodes is not None and init_states is not None)):
            raise ValueError("One of {num_trajectories, init_states}"
                             " should not be None.")

        if num_episodes is not None:
            init_states = [environment.reset() for _ in range(num_episodes)]

        episodes = []
        for init_state in init_states:
            _, episode = play_episode(agent=agent,
                                      environment=environment,
                                      init_state=init_state)
            episodes.append(episode)

        return episodes


def play_episode(agent: Agent,
                 environment: Environment,
                 init_state: Optional[np.ndarray] = None) \
        -> Tuple[float, List[Transition]]:
    """Plays an episode with the current policy."""

    environment.reset()
    if init_state is not None:
        environment.state = init_state

    state = environment.state
    cumulative_reward = 0.0
    episode = []

    while True:
        action = agent.act(state)
        next_state, reward = environment.step(action)

        cumulative_reward += reward.item()
        episode.append(Transition(state, action, reward, next_state))
        state = next_state

        if environment.done: break

    return cumulative_reward, episode


def eval_models(dynamics_model: VFEGPStack,
                rewards_model: VFEGP,
                agent: Agent,
                environment: Environment,
                num_features: int,
                dtype: tf.DType,
                num_trajectories: Optional[int] = None,
                init_states: Optional[List[np.ndarray]] = None):

    dynamics_sample = dynamics_model.sample_posterior(num_features)
    rewards_sample = rewards_model.sample_posterior(num_features)

    episodes = play_episodes(agent, environment, num_trajectories, init_states)

    dynamics_errors, rewards_errors = [], []
    for episode in episodes:
        s, sa, s_, sas_, r = convert_episode_to_tensors(episode, dtype=dtype)

        ds = dynamics_sample(sa, add_noise=True)
        pred_s_ = s + ds
        pred_r = rewards_sample(s_, add_noise=True)

        dynamics_error = tf.keras.metrics.mean_squared_error(s_, pred_s_)
        dynamics_errors.append(dynamics_error.numpy().mean().item())
        rewards_error = tf.keras.metrics.mean_squared_error(r, pred_r)
        rewards_errors.append(rewards_error.numpy().mean().item())

    dyn_err_mean = np.mean(dynamics_errors)
    rew_err_mean = np.mean(rewards_errors)
    print(f"MSE dynamics: {dyn_err_mean:.4f}, MSE rewards: {rew_err_mean:.4f}")
