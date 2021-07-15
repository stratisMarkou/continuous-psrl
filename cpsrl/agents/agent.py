import numpy as np
from typing import List, Tuple, Optional, Callable

from abc import ABC, abstractmethod
from cpsrl.models.initial_distributions import InitialStateDistribution
from cpsrl.policies import FCNPolicy
from cpsrl.errors import AgentError
from cpsrl.helpers import (
    convert_episode_to_tensors,
    check_shape,
    Transition
)

import tensorflow as tf


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
    def act(self, state: tf.Tensor) -> tf.Tensor:
        """
        Method called when the agent interacts with its environment, which
        produces an *action* given a *state* passed to the agent.

        :param state:
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
        contains tf.Tensors representing the state s, action a, next state
        s_ and reward r of an single interaction, in the format

            episode = [(s, a, r, s_), ..., (s, a, r, s_)].

        :return:
        """
        pass


# =============================================================================
# Random agent class
# =============================================================================

class RandomAgent(Agent):
    def __init__(self,
                 action_space: List[Tuple[float, float]],
                 dtype: tf.DType):

        super().__init__(action_space, gamma=None, horizon=None)
        self.dtype = dtype

    def act(self, state: tf.Tensor) -> tf.Tensor:
        return tf.stack([tf.random.uniform(
            shape=(), minval=lo, maxval=hi, dtype=self.dtype)
            for lo, hi in self.action_space])

    def observe(self, episode: List[Transition]):
        pass

    def update(self) -> Optional[dict]:
        pass


# =============================================================================
# Ground truth model agent class
# =============================================================================

class GroundTruthModelAgent(Agent):

    def __init__(self,
                 action_space: List[Tuple[float, float]],
                 horizon: int,
                 gamma: float,
                 initial_distribution: InitialStateDistribution,
                 dynamics_model: Callable,
                 rewards_model: Callable,
                 policy: FCNPolicy,
                 update_params: dict,
                 dtype: tf.DType):

        super().__init__(action_space=action_space,
                         gamma=gamma,
                         horizon=horizon)

        # Set initial state distribution and ground truth models
        self.initial_distribution = initial_distribution
        self.dynamics_model = dynamics_model
        self.rewards_model = rewards_model

        # Set policy and update parameters
        self.policy = policy
        self.update_params = update_params

        # Set data type
        self.dtype = dtype

    def act(self, state: tf.Tensor) -> tf.Tensor:

        if state.ndim == 1:
            state = tf.expand_dims(state, axis=0)

        action = self.policy(state)
        action = tf.squeeze(action, axis=0)

        return action

    def observe(self, episode: List[Transition]):

        # Convert episode to tensors, to update the models' training data
        s, sa, s_, sas_, r = convert_episode_to_tensors(episode)
        self.initial_distribution.add_training_data(s[0:1])

    def update(self, **kwargs) -> Optional[dict]:

        # Train the initial distribution, dynamics and reward models
        self.initial_distribution.update()

        # Optimise the policy
        print("\nUpdating policy using ground truth models...")
        self.policy.reset()
        info_dict = self.optimise_policy(
            num_rollouts=self.update_params["num_rollouts"],
            num_steps=self.update_params["num_steps_policy"],
            learn_rate=self.update_params["learn_rate_policy"]
        )

        return info_dict

    def optimise_policy(self,
                        num_rollouts: int,
                        num_steps: int,
                        learn_rate: float) -> dict:

        # Draw dynamics and rewards samples
        initial_distribution = self.initial_distribution.sample_posterior()

        # Initialise optimiser
        optimizer = tf.optimizers.Adam(learn_rate)
        print_freq = np.maximum(1, num_steps // 10)

        info_dict = {"policy_loss": [], "rollout": []}

        for i in range(num_steps):
            with tf.GradientTape() as tape:

                # Ensure policy variables are being watched
                tape.watch(self.policy.trainable_variables)

                # Draw initial states s0
                s0 = initial_distribution.sample(num_rollouts)

                # Perform rollouts
                cum_reward, rollout = self.rollout(
                    dynamics_model=self.dynamics_model,
                    rewards_model=self.rewards_model,
                    horizon=self.horizon,
                    gamma=self.gamma,
                    s0=s0
                )

                # Loss is (-ve) mean discounted reward, normalised by horizon
                loss = - tf.reduce_mean(cum_reward) / self.horizon

                info_dict["policy_loss"].append(loss.numpy().item())
                info_dict["rollout"].append(rollout)

                if i % print_freq == 0 or i == num_steps - 1:
                    rew_mean = tf.reduce_mean(cum_reward)
                    rew_std = tf.math.reduce_std(cum_reward)
                    rew_min = tf.math.reduce_min(cum_reward)
                    rew_max = tf.math.reduce_max(cum_reward)
                    print(f"Step: {i}, Loss: {loss:.4f}, "
                          f"Mean reward: {rew_mean:.4f}, "
                          f"Std reward: {rew_std:.4f}, "
                          f"Min reward: {rew_min:.4f}. "
                          f"Max reward: {rew_max:.4f}")

            # Compute gradients wrt policy variables and apply gradient step
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            gradients = [tf.clip_by_norm(g, 2.0) for g in gradients]
            optimizer.apply_gradients(zip(gradients,
                                          self.policy.trainable_variables))

        return info_dict

    def rollout(self,
                dynamics_model: Callable,
                rewards_model: Callable,
                horizon: int,
                s0: tf.Tensor,
                gamma: float) -> Tuple[tf.Tensor, List[List[Transition]]]:

        # Check discount factor is valid
        if not 0. <= gamma <= 1.:
            raise AgentError(f"Agent expected gamma between 0 and 1, "
                             f"got {gamma}.")

        # Check shape of initial states
        R, S = s0.shape

        # Set state to initial state
        s = s0

        rollouts = []
        cumulative_reward = tf.zeros(shape=(R,), dtype=self.dtype)

        for i in range(horizon):

            # Get action from the policy
            a = self.policy(s)

            # Check shape of action returned by the policy is correct
            check_shape([s, a], [(R, S), (R, 'A')])

            # Concatenate states and actions
            sa = tf.concat([s, a], axis=1)

            # Get next state and rewards from the model samples
            ds = dynamics_model(sa, add_noise=True)

            # Check shapes of s and ds match
            check_shape([s, ds], [(R, S), (R, S)])

            # Compute next state and reward
            s_ = s + ds
            r = rewards_model(s_, add_noise=False)

            # Check shapes of next state and rewards
            check_shape([s, s_, r], [(R, S), (R, S), (R, 1)])

            # Remove last dimension from reward
            r = tf.squeeze(r, axis=1)

            # Store states, actions and rewards
            rollout = [Transition(*t) for t in zip(s, a, r, s_)]
            rollouts.append(rollout)

            # Increment cumulative reward and update state
            cumulative_reward = cumulative_reward + (gamma ** i) * r
            s = s_

        rollouts = list(zip(*rollouts))

        return cumulative_reward, rollouts
