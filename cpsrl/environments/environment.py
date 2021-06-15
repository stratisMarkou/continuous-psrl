from typing import List, Tuple
from abc import ABC, abstractmethod

import numpy as np

# =============================================================================
# Base environment class
# =============================================================================


class Environment(ABC):

    def __init__(self, horizon: int, sub_sampling_factor: int = 1):
        self.horizon = horizon
        self.sub_sampling_factor = sub_sampling_factor

        self.timestep = 0

    @abstractmethod
    def reset(self) -> np.ndarray:
        pass

    @property
    def state(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def action_space(self) -> List[Tuple[float, float]]:
        raise NotImplementedError

    @property
    def done(self) -> bool:
        return self.timestep >= self.horizon

    @abstractmethod
    def step_dynamics(self, state: np.ndarray, action: np.ndaray) -> np.ndarray:
        pass

    @abstractmethod
    def get_reward(self,
                   state: np.ndarray,
                   action: np.ndaray,
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
        return next_state, reward
