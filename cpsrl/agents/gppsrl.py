from typing import Tuple, List

import tensorflow as tf

from cpsrl.agents import Agent


class GPPSRLAgent(Agent):
    def __init__(self, action_space: List[Tuple[float, float]]):
        super().__init__(action_space=action_space)

    def rollout(self, dynamics_sample, rewards_sample, horizon):
        
        initial_states = None
        states = [init_states]
        actions = []
        rewards = []
        
        r = 0.
        
        # Do rollout and compute rewards
        for i in range(horizon):
            
            a = self.policy(s)
            
            sa = tf.concat([s, a], axis=1)
            
            s_ = dynamics_sample.sample_next_states(sa)
            
            sas_ = tf.concat([sa, s_], axis=1)
            
            r = rewards_sample.sample_rewards(sas_)
            
            actions.append(a)
            rewards.append(r)

    def optimise_policy(self):
        pass
