from typing import List, Tuple, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt

from cpsrl.environments import Environment
from cpsrl.errors import EnvironmentError
from cpsrl.helpers import check_shape, Transition
from cpsrl.plot_utils import color_defaults

import tensorflow as tf

# =============================================================================
# MountainCar environment
# =============================================================================


class MountainCar(Environment):
    """
    Continuous MountainCar environment with a 2D state space and 1D
    action space.
    """

    def __init__(self,
                 dtype: tf.DType,
                 horizon: Optional[int] = None,
                 sub_sampling_factor: int = 1,
                 goal_pos: float = 0.5,
                 goal_scale: float = 1.0):

        horizon = horizon or 100
        super().__init__(dtype=dtype,
                         horizon=horizon,
                         sub_sampling_factor=sub_sampling_factor)

        self.reward_loc = tf.constant([[goal_pos, 0.]],
                                      dtype=self.dtype)

        self.reward_scale = tf.constant([[goal_scale, goal_scale]],
                                        dtype=self.dtype)
        self.power = 0.0015

        # Set dimensions of state space and action space
        self.S = 2
        self.A = 1

    def reset(self) -> tf.Tensor:

        self.state = tf.constant([-0.5, 0.], dtype=self.dtype)
        self.timestep = 0

        return self.state

    @property
    def state_space(self) -> List[Tuple[float, float]]:
        return [(-3.0, 3.0), (-0.07, 0.07)]

    @property
    def action_space(self) -> List[Tuple[float, float]]:
        return [(-1., 1.)]

    def ground_truth_models(self) -> Tuple[Callable, Callable]:

        def dynamics_model(state_action, add_noise):

            state = state_action[:, :2]
            # print("State")
            # print(state)
            action = state_action[:, 2:]
            # print("Action")
            # print(action)
            next_state = self.step_dynamics(state, action)
            # print("Next state")
            # print(next_state)
            # input("")

            return next_state - state

        def rewards_model(state, add_noise):
            # print("State")
            # print(state)
            reward = self.get_reward(state, None, None)
            # print("Reward")
            # print(reward)
            # input('')
            return reward

        return dynamics_model, rewards_model

    def step_dynamics(self, state: tf.Tensor, action: tf.Tensor) -> tf.Tensor:

        # Reshape state and action into two-dimensional tensors
        state = tf.reshape(state, (-1, self.S))
        action = tf.reshape(action, (-1, self.A))

        # Check that state and action have compatible shapes
        check_shape([state, action], [('R', 'S'), ('R', 'A')])

        # Check if action is in permissible space
        valid_action = [(a1 <= a <= a2)
                        for (a1, a2), a in zip(self.action_space, action)]
        valid_action = all(valid_action)

        if not valid_action:
            raise EnvironmentError(f'Expected action in the space '
                                   f'{self.action_space} got action {action}.')

        # Unpack position and velocity
        position = state[:, :1]
        velocity = state[:, 1:]

        # Check that position, velocity, action have compatible shapes
        check_shape([position, velocity, action],
                    [('R', 1), ('R', 1), ('R', 1)])

        # Increment position by velocity
        position_ = position + velocity

        # Increment velocity by Euler rule and clip
        velocity_ = velocity + \
                    action * self.power - 0.0025 * tf.cos(3 * position)

        next_state = tf.concat([position_, velocity_], axis=1)

        # Clip state to permissible space
        next_state = [tf.clip_by_value(next_state[:, i:i+1], s1, s2)
                      for i, (s1, s2) in enumerate(self.state_space)]
        next_state = tf.concat(next_state, axis=1)

        return next_state

    def get_reward(self,
                   state: tf.Tensor,
                   action: tf.Tensor,
                   next_state: tf.Tensor) -> tf.Tensor:

        # Reshape state and action into two-dimensional tensors
        state = tf.reshape(state, (-1, self.S))

        # Check shapes are compatible
        check_shape([state, self.reward_loc, self.reward_scale],
                    [(-1, self.S,), (1, self.S,), (1, self.S,)])

        diff = (state - self.reward_loc) / self.reward_scale
        quad = tf.reduce_sum(diff**2, axis=1, keepdims=True)
        reward = tf.exp(-0.5 * quad)

        # Check shape of the  reward
        check_shape([reward, state], [('R', 1), ('R', self.S)])

        return reward

    def plot_trajectories(self,
                          trajectories: List[List[Transition]],
                          ground_truth: List[Transition] = None,
                          save_dir: Optional[str] = None,
                          **plot_kwargs):

        fig = plt.figure()

        # joint axis
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w',
                       top=False,
                       bottom=False,
                       left=False,
                       right=False)

        # position
        ax1 = fig.add_subplot(411)
        ax1.set_ylabel("Position")
        plt.setp(ax1.get_xticklabels(), visible=False)

        # velocity
        ax2 = fig.add_subplot(412, sharex=ax1)
        ax2.set_ylabel("Velocity")
        plt.setp(ax2.get_xticklabels(), visible=False)

        # action
        ax3 = fig.add_subplot(413, sharex=ax1)
        ax3.set_ylabel("Action")
        plt.setp(ax3.get_xticklabels(), visible=False)

        # reward
        ax4 = fig.add_subplot(414, sharex=ax1)
        ax4.set_ylabel("Reward")

        for i, trajectory in enumerate(trajectories):

            t = np.arange(len(trajectory))
            states, actions, rewards, _ = Transition(*zip(*trajectory))

            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)

            color = color_defaults[0] if i > 2 else color_defaults[1]
            alpha = min(1., 5. / len(trajectories)) if i > 2 else 1.
            zorder = 1 if i > 2 else 2

            ax1.plot(t, states[:, 0], c=color, alpha=alpha, zorder=zorder)
            ax1.set_ylim(-2., 2.)

            ax2.plot(t, states[:, 1], c=color, alpha=alpha, zorder=zorder)
            ax2.set_ylim(-0.1, 0.1)

            ax3.plot(t, actions, c=color, alpha=alpha, zorder=zorder)
            ax3.set_ylim(-1.5, 1.5)

            ax4.plot(t, rewards, c=color, alpha=alpha, zorder=zorder)
            ax4.set_ylim(-0.5, 1.5)

        if ground_truth:

            t = np.arange(len(ground_truth))
            states, actions, rewards, _ = Transition(*zip(*ground_truth))

            color = color_defaults[2]
            alpha = 1.0
            zorder = 3

            ax1.plot(t, states[:, 0], c=color, alpha=alpha, zorder=zorder)
            ax2.plot(t, states[:, 1], c=color, alpha=alpha, zorder=zorder)
            ax3.plot(t, actions, c=color, alpha=alpha, zorder=zorder)
            ax4.plot(t, rewards, c=color, alpha=alpha, zorder=zorder)

        ax.set_xlabel('Time')
        fig.align_ylabels([ax1, ax2, ax3, ax4])

        if save_dir is not None:
            plt.savefig(save_dir, bbox_inches="tight")

        plt.close()



