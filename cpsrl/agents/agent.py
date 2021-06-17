import numpy as np
from typing import List, Tuple

from abc import ABC, abstractmethod
from cpsrl.models.gp import VFEGPStack

# =============================================================================
# Base agent class
# =============================================================================


class Agent(ABC):

    def __init__(self, action_space: List[Tuple[float, float]]):
        self.action_space = action_space

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
    def update(self):
        """
        Method called after each episode and performs the updates required by
        the agent, such as retraining the models or updating the policy.
        """
        pass

    @abstractmethod
    def observe(self, episode: List[Tuple]):
        """
        Method called after each episode, which adds the data contained in
        *episode* to the dataset held by the agent.

        :param episode: List of tuples each of length 4. Each tuple contains
        np.ndarrays representing the state s, action a, next state s_ and
        reward r of an single interaction, in the format

            episode = [(s, a, s_, r), ..., (s, a, s_, r)].

        :return:
        """
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



# =============================================================================
# GPPSRL agent
# =============================================================================


class GPPSRLAgent(ABC):

    def __init__(self,
                 dynamics_model: VFEGPStack,
                 rewards_model: VFEGPStack,
                 policy: Policy,
                 action_space: List[Tuple[float, float]]):

        self.action_space = action_space


    def act(self):
        pass
