import warnings
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
                 dtype: tf.DType,
                 max_ind: int = None):
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
        :param max_ind: max # of inducing points, set to training data if None
        """

        # Use superclass init
        super().__init__(action_space=action_space,
                         horizon=horizon,
                         gamma=gamma)

        # Set dynamics and rewards models and policy
        self.max_ind = max_ind
        self.initial_distribution = initial_distribution
        self.dynamics_model = dynamics_model
        self.rewards_model = rewards_model
        self.policy = policy
        self.num_observations = 0

        # Set dtype and training parameters for models
        self.dtype = dtype
        self.update_params = update_params

    def act(self, state: tf.Tensor) -> tf.Tensor:

        if state.ndim == 1:
            state = tf.expand_dims(state, axis=0)

        action = self.policy(state)
        action = tf.squeeze(action, axis=0)

        return action

    def observe(self, episode: List[Transition]):

        # Convert episode to tensors, to update the models' training data
        s, sa, s_, sas_, r = convert_episode_to_tensors(episode)
        
        # Initial states for initial dist. and state differences for dynamics
        s0 = s[0:1]
        ds = s_ - s

        # Update the models' training data
        self.initial_distribution.add_training_data(s0)
        self.dynamics_model.add_training_data(sa, ds)
        self.rewards_model.add_training_data(s, r)

        # Increment number of observations
        self.num_observations = self.num_observations + s.shape[0]

    def update(self) -> Optional[dict]:
        """
        Method called after each episode and performs the following updates:
            - Updates the pseudopoints
            - Trains the dynamics and rewards models, if necessary
            - Optimises the policy
        """

        info_dict = {}

        # Train the initial distribution, dynamics and reward models
        self.initial_distribution.update()

        # Update inducing points
        max_ind = self.num_observations if self.max_ind is None else \
                  self.max_ind
        num_ind = min(self.num_observations, max_ind)

        self.dynamics_model.reset_inducing(num_ind=num_ind)
        self.rewards_model.reset_inducing(num_ind=num_ind)

        print(f"\nUsing {num_ind} inducing points.\n")

        print("Updating dynamics model...")
        self.dynamics_model.reset_inducing(num_ind=num_ind)
        self.dynamics_model.reset_parameters()
        dyn_dict = self.train_model(
            self.dynamics_model,
            num_steps=self.update_params["num_steps_dyn"],
            learn_rate=self.update_params["learn_rate_dyn"]
        )
        info_dict["dynamics"] = dyn_dict
        print(self.dynamics_model.parameter_summary())

        print("\nUpdating rewards model...")
        self.rewards_model.reset_inducing(num_ind=num_ind)
        self.rewards_model.reset_parameters()
        rew_dict = self.train_model(
            self.rewards_model,
            num_steps=self.update_params["num_steps_rew"],
            learn_rate=self.update_params["learn_rate_rew"]
        )
        info_dict["rewards"] = rew_dict
        print(self.rewards_model.parameter_summary())

        # Optimise the policy
        print("\nUpdating policy...")
        self.policy.reset()
        pol_dict = self.optimise_policy(
            num_rollouts=self.update_params["num_rollouts"],
            num_features=self.update_params["num_features"],
            num_steps=self.update_params["num_steps_policy"],
            learn_rate=self.update_params["learn_rate_policy"]
        )

        info_dict.update(pol_dict)

        return info_dict

    def train_model(self,
                    model: Union[VFEGP, VFEGPStack],
                    num_steps: int,
                    learn_rate: float) -> dict:

        info_dict = {"loss": []}
        if not model.trainable_variables:
            warnings.warn("Attempted to train model with no trainable "
                           "parameters. Skipping training...")

        # Initialise optimiser
        optimizer = tf.optimizers.Adam(learn_rate)
        print_freq = np.maximum(1, num_steps // 10)

        for i in range(num_steps):
            with tf.GradientTape() as tape:

                # Ensure policy variables are being watched
                tape.watch(model.trainable_variables)

                loss = - model.free_energy()
                info_dict["loss"].append(loss.numpy().item())

            if i % print_freq == 0 or i == num_steps - 1:
                print(f"Step: {i}, Loss: {loss.numpy().item():.4f}")

            # Compute gradients wrt policy variables and apply gradient step
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return info_dict

    def optimise_policy(self,
                        num_rollouts: int,
                        num_features: int,
                        num_steps: int,
                        learn_rate: float) -> dict:
        """
        Draws a posterior dynamics and rewards model and trains the policy
        using these samples.

        :param num_rollouts: number of rollouts to use
        :param num_features: number of RFFs to use in posterior samples
        :param num_steps: number of optimisation steps to run for
        :param learn_rate: policy optimisation learning rate
        :return: dictionary with auxiliary information
        """

        # Draw dynamics and rewards samples
        dyn_sample = self.dynamics_model.sample_posterior(num_features)
        rew_sample = self.rewards_model.sample_posterior(num_features)
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
                cum_reward, rollout = self.rollout(dynamics_sample=dyn_sample,
                                                   rewards_sample=rew_sample,
                                                   horizon=self.horizon,
                                                   gamma=self.gamma,
                                                   s0=s0)

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
            optimizer.apply_gradients(zip(gradients,
                                          self.policy.trainable_variables))
            
        return info_dict

    def rollout(self,
                dynamics_sample: Callable,
                rewards_sample: Callable,
                horizon: int,
                s0: tf.Tensor,
                gamma: float) -> Tuple[tf.Tensor, List[List[Transition]]]:
        """
        Performs Monte Carlo rollouts, using a posterior sample of the dynamics
        and a posterior sample of the rewards models, for a length of *horizon*,
        starting from initial states *s0*.

        :param dynamics_sample:
        :param rewards_sample:
        :param horizon:
        :param s0:
        :param gamma:
        :return: dictionary with auxiliary information
        """

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
            ds = dynamics_sample(sa, add_noise=True)

            # Check shapes of s and ds match
            check_shape([s, ds], [(R, S), (R, S)])

            # Compute next state and reward
            s_ = s + ds
            r = rewards_sample(s_, add_noise=False)

            # Check shapes of next state and rewards
            check_shape([s, s_, r], [(R, S), (R, S), (R, 1)])

            # Remove last dimension from reward
            r = tf.squeeze(r, axis=1)

            # Store states, actions and rewards
            rollout = [Transition(*t) for t in
                       zip(s.numpy(), a.numpy(), r.numpy(), s_.numpy())]
            rollouts.append(rollout)

            # Increment cumulative reward and update state
            cumulative_reward = cumulative_reward + (gamma ** i) * r
            s = s_

        rollouts = list(zip(*rollouts))

        return cumulative_reward, rollouts
