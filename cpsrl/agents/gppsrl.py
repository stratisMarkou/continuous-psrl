import tensorflow as tf

class GPPSRLAgent:
    
    def __init__(self):
        pass
    
    
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