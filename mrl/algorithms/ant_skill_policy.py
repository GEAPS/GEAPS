import torch
import numpy as np
from torch import nn


class AntSkillPolicy(nn.Module):
    def __init__(self, env_name, params_list, device, n_latents=6):
        super(AntSkillPolicy, self).__init__()
        assert n_latents == 6, "the number of skills for ant robot should be 6"
        self.device = device
        self.mean_network = nn.Sequential(nn.Linear(195, 64),
                                          nn.Tanh(),
                                          nn.Linear(64, 64),
                                          nn.Tanh(),
                                          nn.Linear(64, 8),
                                          nn.Tanh())
        self.skill_dim = 8        
        with torch.no_grad():
            param_idx = 0
            for i in range(len(self.mean_network)):
                if type(self.mean_network[i]) == nn.Linear:
                    self.mean_network[i].weight = torch.nn.parameter.Parameter(torch.from_numpy(params_list[param_idx].T))
                    self.mean_network[i].bias = torch.nn.parameter.Parameter(torch.from_numpy(params_list[param_idx + 1]))
                    print("assign the %i and %i values of param list to %i th layer" % (param_idx, param_idx + 1, i))
                    param_idx += 2
        self.mean_network.to(device)
        self.std = torch.exp(torch.from_numpy(params_list[-1])).to(device)
        self.min_std = torch.ones_like(self.std) * -1e6
        self.env_name = env_name
    
    # # the number of how many times it will be called depends on the higher level policy.
    def forward(self, observations, latents, deterministic=False):
        observations = np.array(observations)  # needed to do the outer product for the bilinear
        # This part should not go wrong.
        extended_obs = np.concatenate([observations, latents,
                                        np.reshape(
                                            observations[:, :, np.newaxis] * latents[:, np.newaxis, :],
                                            (observations.shape[0], -1))],
                                        axis=1)
        extended_obs = torch.from_numpy(extended_obs).to(self.device)
        # make mean, log_std also depend on the latents (as observ.)
        mean = self.mean_network(extended_obs)

        if deterministic:
            actions = mean
            # log_std = self.min_log_std
        else:
            actions = torch.normal(mean, self.std)
            # log_std = self.log_std
        return actions.detach().cpu().numpy()
    
    def get_skill_state(self, state):
        # the goal will not be considered for the low-level skill
        if isinstance(state, dict):
            obs = state['observation']
            if "maze" in self.env_name.lower():
                obs = obs[..., 2:29]
            elif "push" in self.env_name.lower():
                assert obs.shape[-1] == 33, "The environment is not AntPush"
                obs = np.concatenate([obs[..., 2:3], obs[..., 6:-1]], axis=-1)
            else:
                raise ValueError("I haven't considered this situation")
            return obs
        raise ValueError("I haven't considered this situation")   

 
    def mod_actions(self, action, skill_action, mod_indices, randoms):
        action[mod_indices] = skill_action[mod_indices]
        return action


# # torch.save(torch_policy.state_dict(), "skill_params.pkl")
# # print(torch_policy.state_dict().keys())
# np.random.seed(100)
# for _ in range(10):
#     lat = np.zeros(6, dtype="float32")
#     lat[np.random.randint(0,6)] = 1
#     # policy.set_pre_fix_latent(lat)
#     random_obs = np.array([np.random.random()*2-1 for i in range(27)])
#     # a_old = policy.get_actions(random_obs[None])
#     a_new = torch_policy.get_actions(random_obs[None], lat[None])
#     # print((a_old[1]["mean"] - a_new[1]["mean"]) < 1e-10)
#     print(a_new[1]["mean"])
