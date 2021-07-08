from typing import List, Tuple, Callable
from abc import ABC, abstractmethod

import tensorflow as tf

from cpsrl.agents import Agent
from cpsrl.train_utils import play_episode
from cpsrl.helpers import check_shape, Transition


# =============================================================================
# Base environment class
# =============================================================================


class Environment(ABC):

    def __init__(self,
                 horizon: int,
                 dtype: tf.DType,
                 sub_sampling_factor: int = 1):

        self.horizon = horizon
        self.dtype = dtype
        self.sub_sampling_factor = sub_sampling_factor

        self.timestep = 0
        self.state = None
        self.reset()

    @abstractmethod
    def reset(self) -> tf.Tensor:
        pass

    @property
    def state_space(self) -> List[Tuple[float, float]]:
        raise NotImplementedError

    @property
    def action_space(self) -> List[Tuple[float, float]]:
        raise NotImplementedError

    @property
    def done(self) -> bool:
        return self.timestep >= self.horizon

    @abstractmethod
    def ground_truth_models(self) -> Tuple[Callable, Callable]:
        pass

    @abstractmethod
    def step_dynamics(self,
                      state: tf.Tensor,
                      action: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def get_reward(self,
                   state: tf.Tensor,
                   action: tf.Tensor,
                   next_state: tf.Tensor) -> tf.Tensor:
        pass

    def step(self, action: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        state = self.state
        next_state = None

        for i in range(self.sub_sampling_factor):
            next_state = self.step_dynamics(state, action)
            state = next_state

        reward = self.get_reward(self.state, action, next_state)

        reward = tf.reshape(reward, (-1,))
        next_state = tf.reshape(next_state, (-1,))

        self.timestep += 1
        self.state = next_state

        return next_state, reward

    def plot(self,
             agent: Agent,
             num_episodes: int,
             save_dir: str,
             **plot_kwargs):

        trajectories = []

        for _ in range(num_episodes):

            _, trajectory = play_episode(agent=agent, environment=self)
            trajectories.append(trajectory)

        self.plot_trajectories(trajectories, save_dir, **plot_kwargs)

    @abstractmethod
    def plot_trajectories(self,
                          trajectories: List[List[Transition]],
                          save_dir: str,
                          **plot_kwargs):
        pass
