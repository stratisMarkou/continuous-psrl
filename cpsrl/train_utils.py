from __future__ import annotations

from collections import namedtuple
from typing import Tuple, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid circular imports
    from cpsrl.agents import Agent
    from cpsrl.environments import Environment

Transition = namedtuple("Transition", ("obs", "action", "reward", "next_obs"))

# =============================================================================
# Helper for rollout out one episode
# =============================================================================


def play_episode(agent: Agent,
                 environment: Environment,
                 init_state: Optional[np.ndarray] = None) \
        -> Tuple[float, List[Transition]]:
    """Plays an episode with the current policy."""

    environment.reset()
    if init_state is not None:
        environment.state = init_state

    state = environment.state
    cumulative_reward = 0
    episode = []

    while True:
        action = agent.act(state)
        next_state, reward = environment.step(action)

        cumulative_reward += reward
        episode.append(Transition(state, action, next_state, reward))
        state = next_state

        if environment.done: break

    return cumulative_reward, episode
