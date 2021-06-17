from typing import List, Tuple, Callable
from abc import ABC

from cpsrl.models.gp import VFEGP, VFEGPStack
from cpsrl.helpers import *
from cpsrl.policies.policies import Policy

import tensorflow as tf


# =============================================================================
# GPPSRL agent
# =============================================================================

class GPPSRLAgent(ABC):

    def __init__(self,
                 dynamics_model: VFEGPStack,
                 rewards_model: VFEGP,
                 policy: Policy,
                 dtype: tf.DType):

        self.dynamics_model = dynamics_model
        self.rewards_model = rewards_model
        self.policy = policy

        self.dtype = dtype

    def act(self, state: ArrayOrTensor) -> tf.Tensor:

        state = tf.convert_to_tensor(state, dtype=self.dtype)

        return self.policy(state)

    def observe(self, episode: List[Tuple]):

        # Convert episode to tensors, to update the models' training data
        sa, s_, sas_, r = convert_episode_to_tensors(episode, dtype=self.dtype)

        # Update the models' training data
        self.dynamics_model.add_training_data(sa, s_)
        self.rewards_model.add_training_data(sas_, r)

    def update(self):
        """
        Method called after each episode and performs the following updates:
            - Updates the pseudopoints
            - Trains the dynamics and rewards models, if necessary
            - Optimises the policy
        """

        # Update pseudopoints of the GP models

        # Train the dynamics and reward models

        # Optimise the policy


    def rollout(self,
                dynamics_sample: Callable[[tf.Tensor], tf.Tensor],
                rewards_sample: Callable[[tf.Tensor], tf.Tensor],
                horizon: int,
                s0: tf.Tensor):
        """
        Performs Monte Carlo rollouts, using a posterior sample of the dynamics
        and a posterior sample of the rewards models, for a length of *horizon*,
        starting from initial states *s0*.

        :param dynamics_sample:
        :param rewards_sample:
        :param horizon:
        :param s0:
        :return:
        """

        check_shape(s0, ('R', 'S'))

        for i in range(horizon):
            pass
