from typing import Callable

from cpsrl.policies.policies import FCNPolicy
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
                 policy: FCNPolicy,
                 update_params: dict,
                 dtype: tf.DType):
        """
        PSRL agent using Gaussian Processes for the dynamics and rewards models.

        :param action_space: space of allowed actions
        :param horizon: episode horizon, used for optimising the policy
        :param gamma: discount factor
        :param dynamics_model: dynamics model, collection of independent GPs
        :param rewards_model: rewards model, single GP modelling the rewards
        :param policy: policy to use
        :param update_params: parameters required in the update step
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

        # Set dtype and training parameters for models
        self.dtype = dtype
        self.update_params = update_params

    def act(self, state: ArrayOrTensor) -> tf.Tensor:

        state = tf.convert_to_tensor(state, dtype=self.dtype)
        if state.ndim == 1:
            state = tf.expand_dims(state, axis=0)

        action = self.policy(state)
        action = tf.squeeze(action, axis=0)
        return action

    def observe(self, episode: List[Transition]):

        # Convert episode to tensors, to update the models' training data
        s, sa, s_, sas_, r = convert_episode_to_tensors(episode,
                                                        dtype=self.dtype)
        
        # Initial states for initial dist. and state differences for dynamics
        s0 = s[0:1]
        ds = s_ - s

        # Update the models' training data
        self.initial_distribution.add_training_data(s0)
        self.dynamics_model.add_training_data(sa, ds)
        self.rewards_model.add_training_data(s, r)

    def update(self):
        """
        Method called after each episode and performs the following updates:
            - Updates the pseudopoints
            - Trains the dynamics and rewards models, if necessary
            - Optimises the policy
        """

        # Update pseudopoints of the GP models
        params = self.update_params
        self.dynamics_model.reset_inducing(num_ind=params["num_ind_dyn"])
        self.rewards_model.reset_inducing(num_ind=params["num_ind_rew"])

        # Train the initial distribution, dynamics and reward models
        self.initial_distribution.update()

        print("Updating dynamics model...")
        self.train_model(self.dynamics_model,
                         num_steps=params["num_steps_dyn"],
                         learn_rate=params["learn_rate_dyn"])

        print("\nUpdating rewards model...")
        self.train_model(self.rewards_model,
                         num_steps=params["num_steps_rew"],
                         learn_rate=params["learn_rate_rew"])

        # Optimise the policy
        print("\nUpdating policy...")
        self.optimise_policy(num_rollouts=params["num_rollouts"],
                             num_features=params["num_features"],
                             num_steps=params["num_steps_policy"],
                             learn_rate=params["learn_rate_policy"])

    def train_model(self,
                    model: Union[VFEGP, VFEGPStack],
                    num_steps: int,
                    learn_rate: float) -> float:

        if not model.trainable_variables:
            raise AgentError("Attempted to train model with no trainable "
                             "parameters.")

        # Initialise optimiser
        optimizer = tf.optimizers.Adam(learn_rate)
        print_freq = np.maximum(1, num_steps // 10)

        for i in range(num_steps):
            with tf.GradientTape() as tape:
                
                # Ensure policy variables are being watched
                tape.watch(model.trainable_variables)

                loss = - model.free_energy()

            if i % print_freq == 0 or i == num_steps - 1:
                print(f"Step: {i}, Loss: {loss.numpy().item():.4f}")

            # Compute gradients wrt policy variables and apply gradient step
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss.numpy().item()

    def optimise_policy(self,
                        num_rollouts: int,
                        num_features: int,
                        num_steps: int,
                        learn_rate: float):
        """
        Draws a posterior dynamics and rewards model and trains the policy
        using these samples.

        :param num_rollouts: number of rollouts to use
        :param num_features: number of RFFs to use in posterior samples
        :param num_steps: number of optimisation steps to run for
        :param learn_rate: policy optimisation learning rate
        :return:
        """

        # Draw dynamics and rewards samples
        dyn_sample = self.dynamics_model.sample_posterior(num_features)
        rew_sample = self.rewards_model.sample_posterior(num_features)
        initial_distribution = self.initial_distribution.posterior_sample()

        # Initialise optimiser
        optimizer = tf.optimizers.Adam(learn_rate)
        print_freq = np.maximum(1, num_steps // 10)

        for i in range(num_steps):
            with tf.GradientTape() as tape:
                
                # Ensure policy variables are being watched
                tape.watch(self.policy.trainable_variables)

                # Draw initial states s0
                s0 = initial_distribution.sample(num_rollouts)

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
            optimizer.apply_gradients(zip(gradients,
                                          self.policy.trainable_variables))

    def rollout(self,
                dynamics_sample: Callable,
                rewards_sample: Callable,
                horizon: int,
                s0: tf.Tensor,
                gamma: float) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
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
        rewards = []
        next_states = []

        cumulative_reward = tf.zeros(shape=(R,), dtype=self.dtype)

        for i in range(horizon):
            # Get action from the policy
            a = self.policy(s)

            # Check shape of action returned by the policy is correct
            check_shape([s, a], [(R, S), (R, 'A')])

            # Concatenate states and actions
            sa = tf.concat([s, a], axis=1)

            # Get next state and rewards from the model samples
            ds = dynamics_sample(sa, add_noise=True)
            
            # Check shapes of s and ds match
            check_shape([s, ds], [(R, S), (R, S)])
            
            # Compute next state and reward
            s_ = s + ds
            r = rewards_sample(s_, add_noise=True)

            # Check shapes of next state and rewards
            check_shape([s, s_, r], [(R, S), (R, S), (R, 1)])

            # Remove last dimension from reward
            r = tf.squeeze(r, axis=1)

            # Store states, actions and rewards
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_)

            # Increment cumulative reward and update state
            cumulative_reward = cumulative_reward + (gamma ** i) * r
            s = s_

        states = tf.stack(states, axis=1)
        actions = tf.stack(actions, axis=1)
        rewards = tf.stack(rewards, axis=1)
        next_states = tf.stack(next_states, axis=1)

        return cumulative_reward, states, actions, rewards, next_states
