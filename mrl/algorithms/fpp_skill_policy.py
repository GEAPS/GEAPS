import numpy as np

def map_to_action(phi, theta):
    # only support phi and theta as scalars temporally.
    z = np.cos(phi)
    scale = np.sin(phi)
    x = np.cos(theta) * scale
    y = np.sin(theta) * scale
    return np.stack([x, y, z], axis=1)

# this policy only controls the mocap part and doesn't control gripper.
class FppSkillPolicy:
    def __init__(self, n_latents = 8, n_envs = 7):
        assert n_latents == 8, "the number of skills is incorrect"
        self.phi_offset = np.array([i * np.pi/2. for i in range(2)])
        self.theta_offset = np.array([i * np.pi/2. for i in range(4)])
        self.skill_param_indices = np.array([[i, j] for i in range(2) for j in range(4)])
        self.action_dim = 3
    
    # # the number of how many times it will be called depends on the higher level policy.
    def __call__(self, observations, latents, deterministic=False):
        latent_indices = np.argmax(latents, axis=1)
        n_env = len(latent_indices)
        skill_param_indices = self.skill_param_indices[latent_indices]
        phi_offset = self.phi_offset[skill_param_indices[:, 0]]
        theta_offset = self.theta_offset[skill_param_indices[:, 1]]
        phi_val = np.random.uniform(0, np.pi/2., n_env) + phi_offset
        theta_val = np.random.uniform(0, np.pi/2, n_env) + theta_offset
        action = map_to_action(phi_val, theta_val)
        assert action.shape == (n_env, 3), "the shape of action is not correct"
        return action
    
    def get_skill_state(self, state):
        return state
    
    def mod_actions(sefl, action, skill_action, mod_indices, randoms):
        assert action.shape[1] == 4, "the action dimension is not compatible with our skill policy"
        action[mod_indices][:, :3] = skill_action[mod_indices]
        #action[mod_indices][:, 3:] = randoms[mod_indices][:, 3:]
        return action



