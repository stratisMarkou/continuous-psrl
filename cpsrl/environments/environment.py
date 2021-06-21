from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

import numpy as np

from cpsrl.agents import Agent
from cpsrl.train_utils import play_episode, Transition

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

        self.timestep += 1
        self.state = next_state

        return next_state, reward

    def plot(self,
             agent: Agent,
             num_trajectories: Optional[int] = None,
             init_states: Optional[List[np.ndarray]] = None,
             **plot_kwargs):

        if ((num_trajectories is None and init_states is None)
                or (num_trajectories is not None and init_states is not None)):
            raise ValueError("One of {num_trajectories, init_states}"
                             " should not be None.")

        if num_trajectories is not None:
            init_states = [self.reset() for _ in range(num_trajectories)]

        trajectories = []
        for init_state in init_states:
            _, trajectory = play_episode(agent=agent,
                                         environment=self,
                                         init_state=init_state)
            trajectories.append(trajectory)

        self.plot_trajectories(trajectories, **plot_kwargs)

    @abstractmethod
    def plot_trajectories(self, trajectories: List[List[Transition]], **plot_kwargs):
        pass
