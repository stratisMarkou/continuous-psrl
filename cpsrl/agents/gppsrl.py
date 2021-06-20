from typing import List, Tuple, Callable

from cpsrl.policies.policies import Policy
from cpsrl.agents.agent import Agent
from cpsrl.models.gp import VFEGP, VFEGPStack
from cpsrl.models.initial_distributions import InitialStateDistribution
from cpsrl.errors import AgentError
from cpsrl.helpers import *

import tensorflow as tf


# =============================================================================
# GPPSRL agent
# =============================================================================

class GPPSRLAgent(Agent):

    def __init__(self,
                 action_space: List[Tuple[float, float]],
                 horizon: int,
                 gamma: float,
                 initial_distribution: InitialStateDistribution,
                 dynamics_model: VFEGPStack,
                 rewards_model: VFEGP,
                 policy: Policy,
                 dtype: tf.DType):
        """
        PSRL agent using Gaussian Processes for the dynamics and rewards models.

        :param action_space: space of allowed actions
        :param horizon: episode horizon, used for optimising the policy
        :param gamma: discount factor
        :param dynamics_model: dynamics model, collection of independent GPs
        :param rewards_model: rewards model, single GP modelling the rewards
        :param policy: policy to use
        :param dtype: data type of the agent, tf.float32 or tf.float64
        """

        # Use superclass init
        super().__init__(action_space=action_space,
                         horizon=horizon,
                         gamma=gamma)

        # Set dynamics and rewards models and policy
        self.initial_distribution = initial_distribution
        self.dynamics_model = dynamics_model
        self.rewards_model = rewards_model
        self.policy = policy

        # Set dtype
        self.dtype = dtype

    def act(self, state: ArrayOrTensor) -> tf.Tensor:

        state = tf.convert_to_tensor(state, dtype=self.dtype)

        return self.policy(state)

    def observe(self, episode: List[Tuple]):

        # Convert episode to tensors, to update the models' training data
        s, sa, s_, sas_, r = convert_episode_to_tensors(episode,
                                                        dtype=self.dtype)

        # Update the models' training data
        self.initial_distribution.add_training_data(s)
        self.dynamics_model.add_training_data(sa, s_)
        self.rewards_model.add_training_data(sas_, r)

    def update(self, num_ind_dyn: int, num_ind_rew: int):
        """
        Method called after each episode and performs the following updates:
            - Updates the pseudopoints
            - Trains the dynamics and rewards models, if necessary
            - Optimises the policy
        """

        # Update pseudopoints of the GP models
        self.dynamics_model.reset_inducing(num_ind_dyn)
        self.rewards_model.reset_inducing(num_ind_rew)

        # Train the initial distribution, dynamics and reward models
        self.initial_distribution.train()
        # self.dynamics_model.train()
        # self.rewards_model.train()

        # Optimise the policy
        self.optimise_policy(num_rollouts=num_rollouts,
                             num_features=num_features,
                             num_steps=num_steps,
                             learn_rate=learn_rate)

    def optimise_policy(self,
                        num_rollouts: int,
                        num_features: int,
                        num_steps: int,
                        learn_rate: float):
        """
        Draws a posterior dynamics and rewards model and trains the policy
        using these samples.

        :param num_rollouts: number of rollouts to use
        :param num_features: number of RFF features to use in posterior samples
        :param num_steps: number of optimisation steps to run for
        :param learn_rate: policy optimisation learning rate
        :return:
        """

        # Draw dynamics and rewards samples
        dyn_sample = self.dynamics_model.sample_posterior(num_features)
        rew_sample = self.rewards_model.sample_posterior(num_features)

        # Initialise optimiser
        optimizer = tf.optimizers.Adam(learn_rate)

        for i in range(num_steps):
            with tf.GradientTape() as tape:

                # Ensure policy variables are being watched
                tape.watch(self.policy.trainable_variables)

                # Draw initial states s0
                s0 = self.initial_distribution.sample(num_rollouts)

                # Perform rollouts
                rollout = self.rollout(dynamics_sample=dyn_sample,
                                       rewards_sample=rew_sample,
                                       horizon=self.horizon,
                                       gamma=self.gamma,
                                       s0=s0)

                # Unpack rollout results
                cum_reward, states, actions, next_states, rewards = rollout

                # Loss is (-ve) mean discounted reward, normalised by horizon
                loss = - tf.reduce_mean(cum_reward) / self.horizon

            # Compute gradients wrt policy variables and apply gradient step
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            optimizer.apply_gradients(zip(gradients,
                                          self.policy.trainable_variables))

    def rollout(self,
                dynamics_sample: Callable[[tf.Tensor], tf.Tensor],
                rewards_sample: Callable[[tf.Tensor], tf.Tensor],
                horizon: int,
                s0: tf.Tensor,
                gamma: float) -> List[tf.Tensor]:
        """
        Performs Monte Carlo rollouts, using a posterior sample of the dynamics
        and a posterior sample of the rewards models, for a length of *horizon*,
        starting from initial states *s0*.

        :param dynamics_sample:
        :param rewards_sample:
        :param horizon:
        :param s0:
        :param gamma:
        :return:
        """

        # Check discount factor is valid
        if not 0. <= gamma <= 1.:
            raise AgentError(f"Agent expected gamma between 0 and 1, "
                             f"got {gamma}.")

        # Check shape of initial states
        R, S = s0.shape

        # Set state to initial state
        s = s0

        # Arrays for storing rollouts
        states = []
        actions = []
        next_states = []
        rewards = []

        cumulative_reward = tf.zeros(shape=(R,), dtype=self.dtype)

        for i in range(horizon):

            # Get action from the policy
            a = self.policy(s)

            # Check shape of action returned by the policy is correct
            check_shape([s, a], [(R, S), (R, 'A')])

            # Concatenate states and actions
            sa = tf.concat([s, a], axis=1)

            # Get next state and rewards from the model samples
            s_ = dynamics_sample(sa, add_noise=True)
            r = rewards_sample(s, add_noise=True)

            # Check shapes of next state and rewards
            check_shape([s, s_, r], [(R, S), (R, S), (R, 1)])

            # Remove last dimension from reward
            r = tf.squeeze(r, axis=1)

            # Store states, actions and rewards
            states.append(s)
            actions.append(a)
            next_states.append(s_)
            rewards.append(r)

            # Increment cumulative reward and update state
            cumulative_reward = cumulative_reward + (gamma ** i) * r
            s = s_

        states = tf.stack(states, axis=1)
        actions = tf.stack(actions, axis=1)
        next_states = tf.stack(next_states, axis=1)
        rewards = tf.stack(rewards, axis=1)

        return cumulative_reward, states, actions, next_states, rewards

