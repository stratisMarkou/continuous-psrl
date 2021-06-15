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

    def act(self, state: np.ndarray) -> np.ndarray:

        action = np.array([np.random.uniform(low, high) \
                           for low, high in self.action_space])

        return action

    def observe(self, episode: List[Tuple]):
        pass

    def update(self):
        pass
