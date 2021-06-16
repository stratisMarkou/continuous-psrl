from typing import List, Tuple

import numpy as np

from cpsrl.environments.environment import Environment
from cpsrl.errors import EnvironmentError
from cpsrl.helpers import check_shape

# =============================================================================
# MountainCar environment
# =============================================================================


class MountainCar(Environment):
    """
    Continuous MountainCar environment with a 2D state space and 1D
    action space.
    """

    def __init__(self,
                 sub_sampling_factor: int = 1,
                 goal_pos: float = 0.5,
                 goal_scale: float = 1.0):
        super().__init__(horizon=100, sub_sampling_factor=sub_sampling_factor)

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

        return reward
