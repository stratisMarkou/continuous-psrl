from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

import numpy as np

from cpsrl.agents import Agent
from cpsrl.train_utils import play_episode
from cpsrl.helpers import check_shape, Transition


# =============================================================================
# Base environment class
# =============================================================================


class Environment(ABC):

    def __init__(self,
                 horizon: int,
                 rng: np.random.Generator,
                 sub_sampling_factor: int = 1):

        self.horizon = horizon
        self.rng = rng
        self.sub_sampling_factor = sub_sampling_factor

        self.timestep = 0
        self.state = None
        self.reset()

    @abstractmethod
    def reset(self) -> np.ndarray:
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
    def step_dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_reward(self,
                   state: np.ndarray,
                   action: np.ndarray,
                   next_state: np.ndarray) -> np.ndarray:
        pass

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        state = self.state
        next_state = None

        for i in range(self.sub_sampling_factor):
            next_state = self.step_dynamics(state, action)
            state = next_state

        reward = self.get_reward(self.state, action, next_state)
        check_shape(reward, (1,))

        self.timestep += 1
        self.state = next_state

        return next_state, reward

    def plot(self,
             agent: Agent,
             num_episodes: int,
             **plot_kwargs):

        trajectories = []
        for _ in range(num_episodes):
            _, trajectory = play_episode(agent=agent, environment=self)
            trajectories.append(trajectory)

        self.plot_trajectories(trajectories, **plot_kwargs)

    @abstractmethod
    def plot_trajectories(self, trajectories: List[List[Transition]], **plot_kwargs):
        pass
