from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from cpsrl.environments import Environment
from cpsrl.errors import EnvironmentError
from cpsrl.helpers import check_shape, Transition
from cpsrl.plot_utils import color_defaults


# =============================================================================
# MountainCar environment
# =============================================================================


class MountainCar(Environment):
    """
    Continuous MountainCar environment with a 2D state space and 1D
    action space.
    """

    def __init__(self,
                 rng: np.random.Generator,
                 horizon: Optional[int] = None,
                 sub_sampling_factor: int = 1,
                 goal_pos: float = 0.5,
                 goal_scale: float = 1.0):

        horizon = horizon or 100
        super().__init__(horizon=horizon,
                         rng=rng,
                         sub_sampling_factor=sub_sampling_factor)

        self.reward_loc = np.array([goal_pos, 0.])
        self.reward_scale = np.array([goal_scale, goal_scale])

        self.power = 0.0015

    def reset(self) -> np.ndarray:
        self.state = np.array([-0.5, 0.])
        self.timestep = 0
        return self.state

    @property
    def state_space(self) -> List[Tuple[float, float]]:
        return [(-3.0, 3.0), (-0.07, 0.07)]

    @property
    def action_space(self) -> List[Tuple[float, float]]:
        return [(-1., 1.)]

    def step_dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:

        # Check action is of shape (A,)
        check_shape(action, (len(self.action_space),))

        # Check if action is in permissible space
        valid_action = [(a1 <= a <= a2)
                        for (a1, a2), a in zip(self.action_space, action)]
        valid_action = all(valid_action)

        if not valid_action:
            raise EnvironmentError(f'Expected action in the space '
                                   f'{self.action_space} got action {action}.')

        # Unpack position and velocity
        position, velocity = self.state

        # Increment position by velocity
        position_ = position + velocity

        # Increment velocity by Euler rule and clip
        velocity_ = velocity + action * self.power - 0.0025 * np.cos(3 * position)

        next_state = np.array([float(position_), float(velocity_)])

        # Clip state to permissible space
        next_state = [float(np.clip(s, s1, s2))
                      for (s1, s2), s in zip(self.state_space, next_state)]
        next_state = np.array(next_state)

        return next_state

    def get_reward(self,
                   state: np.ndarray,
                   action: np.ndarray,
                   next_state: np.ndarray) -> np.ndarray:

        check_shape(state, (len(self.state_space),))
        check_shape(action, (len(self.action_space),))

        diff = (state - self.reward_loc) / self.reward_scale
        reward = np.exp(-0.5 * np.sum(diff**2))
        reward = reward.reshape((1,))

        return reward

    def plot_trajectories(self,
                          trajectories: List[List[Transition]],
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

            color = color_defaults[0] if i > 3 else color_defaults[1]
            alpha = min(1., 5. / len(trajectories)) if i > 3 else 1.
            zorder = 1 if i > 3 else 2

            ax1.plot(t, states[:, 0], c=color, alpha=alpha, zorder=zorder)
            plt.ylim(-2., 2.)
            ax2.plot(t, states[:, 1], c=color, alpha=alpha, zorder=zorder)
            plt.ylim(-2., 2.)
            ax3.plot(t, actions, c=color, alpha=alpha, zorder=zorder)
            plt.ylim(-1.5, 1.5)
            ax4.plot(t, rewards, c=color, alpha=alpha, zorder=zorder)
            plt.ylim(-2., 2.)

        ax.set_xlabel('Time')
        fig.align_ylabels([ax1, ax2, ax3, ax4])

        if save_dir is not None:
            plt.savefig(save_dir, bbox_inches="tight")

        plt.close()



