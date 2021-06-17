import numpy as np
from typing import List, Tuple

from abc import ABC, abstractmethod
from cpsrl.models.gp import VFEGPStack
from cpsrl.helpers import check_shape

import tensorflow as tf

# =============================================================================
# Base agent class
# =============================================================================


class Agent(ABC):

    def __init__(self, action_space: List[Tuple[float]]):
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
                 state_space: List[Tuple[float, ]],
                 action_space: List[Tuple[float]],
                 dtype: tf.dtype):

        self.dynamics_model = dynamics_model
        self.rewards_model = rewards_model
        self.policy = policy

        self.state_space = state_space
        self.S = len(state_space)
        self.action_space = action_space
        self.A = len(action_space)

        self.dtype = dtype

    def act(self, state: np.ndarray):

        # Check state shape is (S,)
        check_shape(state, (len(self.state_space),))

        return self.policy(state)

    def convert_episode_to_tensors(self, episode: List[Tuple]):
        episode = list(zip(episode))

        episode_sa = []
        episode_s_ = []
        episode_sas_ = []
        episode_r = []

        # Check shapes and append data to arrays
        for s, a, s_, r in episode:

            # Check the shape of the states, actions and rewards
            check_shape([s, a, s_, r], [(self.S,), (self.A), (self.S,), (1,)])

            # Append states, actions and rewards to lists of observed data
            episode_sa.append(np.concatenate([s, a]))
            episode_s_.append(s_)

            episode_sas_.append(np.concatenate([s, a, s_]))
            episode_r.append(r)

        episode_sa = tf.convert_to_tensor(episode_sa, dtype=self.dtype)
        episode_s_ = tf.convert_to_tensor(episode_s_, dtype=self.dtype)
        episode_sas_ = tf.convert_to_tensor(episode_sas_, dtype=self.dtype)
        episode_r = tf.convert_to_tensor(episode_r, dtype=self.dtype)

        return episode_sa, episode_s_, episode_sas_


    def observe(self, episode: List[Tuple]):

        sa, s_, sas_, r = self.convert_episode_to_tensors(episode)





    @abstractmethod
    def update(self):
        """
        Method called after each episode and performs the updates required by
        the agent, such as retraining the models or updating the policy.
        """
        pass

