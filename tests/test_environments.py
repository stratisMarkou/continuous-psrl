from cpsrl.environments import MountainCar
from cpsrl.helpers import check_shape, set_seed
from cpsrl.errors import EnvironmentError

import tensorflow as tf

DTYPE = tf.float64


# ==============================================================================
# Test MountainCar admissible actions
# ==============================================================================


def test_mountaincar_valid_action():
    """
    Test mountaincar action space. Environment should accept actions in the
    admissible space and raise an Environment error when the action is outside
    the admissible space. This test checks the environment
        - Accepts valid actions
        - Rejects actions smaller than the minimum allowed action
        - Rejects actions larger than the maximum allowed action
    :return:
    """

    # Set random seed
    seed = 0

    tf.random.set_seed(seed)
    rng_seq = set_seed(seed)

    # Data type to use
    dtype = tf.float64

    # Horizon
    horizon = 100

    env = MountainCar(horizon=horizon, dtype=dtype)

    min_action, max_action = env.action_space[0]

    _ = env.reset()

    for i in range(horizon):

        action = tf.random.uniform(minval=min_action,
                                   maxval=max_action,
                                   dtype=dtype,
                                   shape=(1,))

        env.step(action=action)

    try:
        # Sample an invalid action smaller than the min action
        action = tf.random.uniform(minval=-2*min_action,
                                   maxval=min_action,
                                   dtype=dtype,
                                   shape=(1,))

        # Try to step the environment with invalid action
        _ = env.step(action=action)

        raise RuntimeError

    except EnvironmentError:
        pass

    _ = env.reset()

    try:
        # Sample an invalid action larger than the max action
        action = tf.random.uniform(minval=max_action,
                                   maxval=2*max_action,
                                   dtype=dtype,
                                   shape=(1,))

        # Try to step the environment with invalid action
        _ = env.step(action=action)

        raise RuntimeError

    except EnvironmentError:
        pass

    _ = env.reset()
