import numpy as np


class SkillActionCache(object):
    def __init__(self, n_env, action_dim, skill_dim):
        self.n_env = n_env
        self.action_dim = action_dim
        self.skill_dim = skill_dim
        self.previous_actions = np.zeros((n_env, skill_dim), dtype=np.float32)
        self.cache_ready = np.zeros(n_env, dtype=np.bool)

    def cache_action(self, indices, actions):
        self.previous_actions[indices] = actions[indices, :self.skill_dim]
        self.cache_ready[indices] = True

    def refresh_cache(self, indices):
        # Assume all the episodes may not terminate together
        # done is a numpy array storing  boolean variabels.
        self.cache_ready[indices] = False

    def mod_action(self, indices, actions):
        # they will not be modified if not ready
        indices = indices & self.cache_ready
        actions[indices, :self.skill_dim] = self.previous_actions[indices]
        return actions 
        
        
