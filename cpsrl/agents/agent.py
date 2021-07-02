import numpy as np
from typing import List, Tuple, Optional

from abc import ABC, abstractmethod
from cpsrl.helpers import Transition


# =============================================================================
# Base agent class
# =============================================================================


class Agent(ABC):

    def __init__(self,
                 action_space: List[Tuple[float, float]],
                 gamma: Optional[float],
                 horizon: Optional[int]):

        self.action_space = action_space
        self.gamma = gamma
        self.horizon = horizon

    @abstractmethod
    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Method called when the agent interacts with its environment, which
        produces an *action* given a *state* passed to the agent.

        :param state: np.ndarray repre
        :return:
        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> Optional[dict]:
        """
        Method called after each episode and performs the updates required by
        the agent, such as retraining the models or updating the policy.
        """
        pass

    @abstractmethod
    def observe(self, episode: List[Transition]):
        """
        Method called after each episode, which adds the data contained in
        *episode* to the dataset held by the agent.

        :param episode: List of Transition each of length 4. Each Transition
        contains np.ndarrays representing the state s, action a, next state
        s_ and reward r of an single interaction, in the format

            episode = [(s, a, s_, r), ..., (s, a, s_, r)].

        :return:
        """
        pass


# =============================================================================
# Random agent class
# =============================================================================

class RandomAgent(Agent):
    def __init__(self,
                 action_space: List[Tuple[float, float]],
                 rng: np.random.Generator):

        super().__init__(action_space, gamma=None, horizon=None)
        self.rng = rng

    def act(self, state: np.ndarray) -> np.ndarray:
        return np.array([self.rng.uniform(lo, hi)
                         for lo, hi in self.action_space])

    def observe(self, episode: List[Transition]):
        pass

    def update(self) -> Optional[dict]:
        pass
