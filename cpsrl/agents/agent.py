import numpy as np
from typing import List, Tuple

from abc import ABC, abstractmethod

# =============================================================================
# Base agent class
# =============================================================================


class Agent(ABC):

    def __init__(self, action_space: List[Tuple[float, float]]):

        self.action_space = action_space

    @abstractmethod
    def act(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def observe(self, episode: List[Tuple]):
        pass


# =============================================================================
# Random agent class
# =============================================================================

class RandomAgent(Agent):
    def __init__(self,
                 action_space: List[Tuple[float, float]],
                 rng: np.random.Generator):

        super().__init__(action_space)
        self.rng = rng

    def act(self, state: np.ndarray) -> np.ndarray:
        return np.array([self.rng.uniform(lo, hi) for lo, hi in self.action_space])

    def observe(self, episode: List[Tuple]):
        pass

    def update(self):
        pass
