import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta

CODE_SIZE = 16

def create_nn(input_size, output_size, hidden_size, num_layers, activation_fn=nn.ReLU, input_normalizer=None,
              final_activation_fn=None, hidden_init_fn=None, b_init_value=None, last_fc_init_w=None):
    # Optionally add a normalizer as the first layer
    if input_normalizer is None:
        input_normalizer = nn.Sequential()
    layers = [input_normalizer]

    # Create and initialize all layers except the last one
    for layer_idx in range(num_layers - 1):
        fc = nn.Linear(input_size if layer_idx == 0 else hidden_size, hidden_size)
        if hidden_init_fn is not None:
            hidden_init_fn(fc.weight)
        if b_init_value is not None:
            fc.bias.data.fill_(b_init_value)
        layers += [fc, activation_fn()]

    # Create and initialize  the last layer
    last_fc = nn.Linear(hidden_size, output_size)
    if last_fc_init_w is not None:
        last_fc.weight.data.uniform_(-last_fc_init_w, last_fc_init_w)
        last_fc.bias.data.uniform_(-last_fc_init_w, last_fc_init_w)
    layers += [last_fc]

    # Optionally add a final activation function
    if final_activation_fn is not None:
        layers += [final_activation_fn()]
    print(layers)
    return nn.Sequential(*layers)


class EDLAntSkillPolicy(nn.Module):
    def __init__(self, env_name, device):
        self.action_size = 8
        super().__init__()
        self.policy = create_nn(27 + CODE_SIZE, self.action_size * 2, hidden_size = 128, num_layers = 4,
                      input_normalizer=None, final_activation_fn=nn.Softplus)
        self.embedding = nn.Embedding(6, CODE_SIZE)
        self.load_model()
        self.env_name = env_name
        self.device = device
        self.policy.to(device)
        self.embedding.to(device)
    
    def load_model(self):
        policy_state_dict = torch.load("mrl/algorithms/models/edl/policy.pth.tar")
        self.policy.load_state_dict(policy_state_dict)
        embed_state_dict = torch.load("mrl/algorithms/models/edl/embed.pth.tar")
        self.embedding.load_state_dict(embed_state_dict)
        print("loading from the pre-trained model successfully.")
    

    # g is the corresponding skill embedding.
    def action_stats(self, s, g):
        x = torch.cat([s, g], dim=1) if g is not None else s
        action_stats = self.policy(x) + 1.05 #+ 1e-6
        return action_stats[:, :self.action_size], action_stats[:, self.action_size:]
    
    def scale_action(self, logit):
        # Scale to [-1, 1]
        logit = 2 * (logit - 0.5)
        # Scale to the action range
        return logit
    
    def action_mode(self, s, skill_code):
        c0, c1 = self.action_stats(s, skill_code)
        action_mode = (c0 - 1) / (c0 + c1 - 2)
        return self.scale_action(action_mode)

    def forward(self, s, latents, deterministic=False):
        """Produce an action"""
        skill_id = torch.from_numpy(np.array(np.argmax(latents, axis=1))).to(self.device)
        s = torch.from_numpy(s).to(self.device)
        skill_code = self.embedding(skill_id)
        c0, c1 = self.action_stats(s, skill_code)
        action_mode = (c0 - 1) / (c0 + c1 - 2)
        m = Beta(c0, c1)

        # Sample.
        if deterministic:
            action_logit = action_mode
        else:
            action_logit = m.sample()

        # n_ent = -m.entropy().mean()
        # lprobs = m.log_prob(action_logit)
        action = self.scale_action(action_logit)
        return action.detach().cpu().numpy() #, action_logit, lprobs, n_ent
    
    def get_skill_state(self, state):
        # the goal will not be considered for the low-level skill
        if isinstance(state, dict):
            obs = state['observation']
            if "maze" in self.env_name.lower():
                assert obs.shape[-1] == 30, "The environment is not AntMaze"
                obs = obs[..., 2:-1]
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