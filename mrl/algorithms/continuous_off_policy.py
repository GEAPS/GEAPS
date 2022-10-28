import mrl
from mrl.utils.misc import soft_update, flatten_state
from mrl.modules.model import PytorchModel
from mrl.algorithms.skill_policy import get_skill_policy

import numpy as np
import torch
import torch.nn.functional as F
import os

# This one will remain the same.
class ActorPolicy(mrl.Module):
  """Used for DDPG / TD3 and other deterministic policy variants"""
  def __init__(self):
    super().__init__(
        'policy',
        required_agent_modules=[
            'actor', 'action_noise', 'env', 'replay_buffer'
        ],
        locals=locals())
    # the skill will depend on the environment.

    self.skill_policy = None

  def _setup(self):
    self.use_actor_target = self.config.get('use_actor_target')
    if self.config.use_skills:
        env_name = self.config.other_args["env"]
        n_latents = self.config.n_latents
        self.skill_policy = get_skill_policy(env_name, n_latents, self.config.other_args["skill_type"])
  

  def __call__(self, state, latents=None, greedy=False):
    action_scale = self.env.max_action
    # initial exploration and intrinsic curiosity
    res = None
    if self.training:
      if self.config.get('initial_explore') and len(self.replay_buffer) < self.config.initial_explore:
        res = np.array([self.env.action_space.sample() for _ in range(self.env.num_envs)])
      elif hasattr(self, 'ag_curiosity'):
        state = self.ag_curiosity.relabel_state(state)

    original_state = state
    state = flatten_state(state)  # concat the obs and the goal
    if hasattr(self, 'state_normalizer'):
      state = self.state_normalizer(state, update=self.training)

    go_explore_idx = np.zeros(self.env.num_envs, dtype=np.bool)

    if res is not None:
      #if self.config.use_skills:
      #  go_explore_idx = np.ones(self.env.num_envs, dtype=np.bool)
      #  assert isinstance(latents, np.ndarray), "the data type of latents is incorrect."
      #  skill_state = self.skill_policy.get_skill_state(original_state)
      #  skill_action = self.skill_policy(skill_state, latents)
      #  skill_action = action_scale * skill_action
      #  res = self.skill_policy.mod_actions(res, skill_action, go_explore_idx)
      #  res = np.clip(res, -action_scale, action_scale)
      return res, go_explore_idx

    state = self.torch(state)

    if self.use_actor_target:
      action = self.numpy(self.actor_target(state))
    else:
      action = self.numpy(self.actor(state))



    # TODO(lisheng) Fix this part.
    if self.training and not greedy:
      action = self.action_noise(action)
      if self.config.get('eexplore'):
        eexplore = self.config.eexplore
        randoms = np.random.random(action.shape) * (2 * action_scale) - action_scale
        if hasattr(self, 'ag_curiosity'):
          # go explore idx indicates those terms without specific goals. 
          go_explore_idx = (self.ag_curiosity.go_explore > 0).flatten()
          if self.config.use_skills:
            assert isinstance(latents, np.ndarray), "the data type of latents is incorrect."
            # if we use skills, simply replace the actions by the skill action           
            if np.any(self.ag_curiosity.go_explore):
              # get skill state should be processed by the skill policy.
              # also the action replacement should also be conducted by the skill policy.
              skill_state = self.skill_policy.get_skill_state(original_state)
              skill_action = self.skill_policy(skill_state, latents)
              skill_action = action_scale * skill_action
              action = self.skill_policy.mod_actions(action, skill_action, go_explore_idx, randoms)
              # action[go_explore_idx] = skill_action[go_explore_idx]
              #eexplore = eexplore * 2
              # eexplore = np.minimum(self.ag_curiosity.go_explore * self.config.go_eexplore + eexplore, 0.3)
          else:
            # otherwise, increase the random probabilities.
            eexplore = self.ag_curiosity.go_explore * self.config.go_eexplore + eexplore
        mask = (np.random.random((action.shape[0], 1)) < eexplore).astype(np.float32)
        action = mask * randoms + (1 - mask) * action
    return np.clip(action, -action_scale, action_scale), go_explore_idx


class StochasticActorPolicy(mrl.Module):
  """Used for SAC / learned action noise"""
  def __init__(self):
    super().__init__(
        'policy',
        required_agent_modules=[
            'actor', 'env', 'replay_buffer'
        ],
        locals=locals())

  def _setup(self):
    self.use_actor_target = self.config.get('use_actor_target')

  def __call__(self, state, greedy=False):
    action_scale = self.env.max_action

    # initial exploration and intrinsic curiosity
    res = None
    if self.training:
      if self.config.get('initial_explore') and len(self.replay_buffer) < self.config.initial_explore:
          res = np.array([self.env.action_space.sample() for _ in range(self.env.num_envs)])
      elif hasattr(self, 'ag_curiosity'):
        state = self.ag_curiosity.relabel_state(state)
        
    state = flatten_state(state)  # flatten goal environments
    if hasattr(self, 'state_normalizer'):
      state = self.state_normalizer(state, update=self.training)
    
    if res is not None:
      return res

    state = self.torch(state)

    if self.use_actor_target:
      action, _ = self.actor_target(state)
    else:
      action, _ = self.actor(state)
    action = self.numpy(action)

    if self.training and not greedy and self.config.get('eexplore'):
      eexplore = self.config.eexplore
      if hasattr(self, 'ag_curiosity'):
        eexplore = self.ag_curiosity.go_explore * self.config.go_eexplore + eexplore
      mask = (np.random.random((action.shape[0], 1)) < eexplore).astype(np.float32)
      randoms = np.random.random(action.shape) * (2 * action_scale) - action_scale
      action = mask * randoms + (1 - mask) * action
    
    return np.clip(action, -action_scale, action_scale)

class BehaviorClone(mrl.Module):
  """This only requires actor and performs Behavior Cloning."""

  def __init__(self):
    super().__init__(
        'algorithm',
        required_agent_modules=['actor','replay_buffer', 'env'],
        locals=locals())

  def _setup(self):
    """Sets up actor/critic optimizers and creates target network modules"""

    # Actor setup
    actor_params = []
    self.actors = []
    for module in list(self.module_dict.values()):
      name = module.module_name
      if name.startswith('actor') and isinstance(module, PytorchModel):
        self.actors.append(module)
        actor_params += list(module.model.parameters())

    self.actor_opt = torch.optim.Adam(
        actor_params,
        lr=self.config.actor_lr,
        weight_decay=self.config.actor_weight_decay)
    
    self.actor_params = actor_params
    self.action_scale = self.env.max_action

  def save(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    torch.save({
      'actor_opt_state_dict': self.actor_opt.state_dict(),
    }, path)

  def load(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    checkpoint = torch.load(path)
    self.actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])

  def _optimize(self):
    if len(self.replay_buffer) > self.config.warm_up:\
      # rewards will not be used
      states, actions, rewards, next_states, gammas = self.replay_buffer.sample(
          self.config.batch_size)

      self.optimize_from_batch(states, actions, rewards, next_states, gammas)

  def optimize_from_batch(self, states, actions, rewards, next_states, gammas):
    # CrossEntropyLoss if discrete; MSE loss, if continuous.
    # But currently we only apply it to continuous environments.
    a = self.actor(states) / self.action_scale
    actions = actions / self.action_scale
    actor_loss = F.mse_loss(a, actions, reduction="mean")
      
    self.actor_opt.zero_grad()
    actor_loss.backward()
      
    # Grad clipping
    if self.config.grad_norm_clipping > 0.:	
      for p in self.actor_params:
        clip_coef = self.config.grad_norm_clipping / (1e-6 + torch.norm(p.grad.detach()))
        if clip_coef < 1:
          p.grad.detach().mul_(clip_coef)
    if self.config.grad_value_clipping > 0.:
      torch.nn.utils.clip_grad_value_(self.actor_params, self.config.grad_value_clipping)
      
    self.actor_opt.step()

class OffPolicyActorCritic(mrl.Module):
  """This is the standard DDPG"""

  def __init__(self):
    super().__init__(
        'algorithm',
        required_agent_modules=['actor','critic','replay_buffer', 'env'],
        locals=locals())

  def _setup(self):
    """Sets up actor/critic optimizers and creates target network modules"""

    self.targets_and_models = []

    # Actor setup
    actor_params = []
    self.actors = []
    for module in list(self.module_dict.values()):
      name = module.module_name
      if name.startswith('actor') and isinstance(module, PytorchModel):
        self.actors.append(module)
        actor_params += list(module.model.parameters())
        target = module.copy(name + '_target')
        target.model.load_state_dict(module.model.state_dict())
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in target.model.parameters():
          p.requires_grad = False
        self.agent.set_module(name + '_target', target)
        self.targets_and_models.append((target.model, module.model))

    self.actor_opt = torch.optim.Adam(
        actor_params,
        lr=self.config.actor_lr,
        weight_decay=self.config.actor_weight_decay)
    
    self.actor_params = actor_params

    # Critic setup
    critic_params = []
    self.critics = []
    for module in list(self.module_dict.values()):
      name = module.module_name
      if name.startswith('critic') and isinstance(module, PytorchModel):
        self.critics.append(module)
        critic_params += list(module.model.parameters())
        target = module.copy(name + '_target')
        target.model.load_state_dict(module.model.state_dict())
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in target.model.parameters():
          p.requires_grad = False
        self.agent.set_module(name + '_target', target)
        self.targets_and_models.append((target.model, module.model))

    self.critic_opt = torch.optim.Adam(
        critic_params,
        lr=self.config.critic_lr,
        weight_decay=self.config.critic_weight_decay)
    
    self.critic_params = critic_params

    self.action_scale = self.env.max_action

  def save(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    torch.save({
      'actor_opt_state_dict': self.actor_opt.state_dict(),
      'critic_opt_state_dict': self.critic_opt.state_dict()
    }, path)

  def load(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    checkpoint = torch.load(path)
    self.actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
    self.critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])

  def _optimize(self):
    if len(self.replay_buffer) > self.config.warm_up:
      states, actions, rewards, next_states, gammas = self.replay_buffer.sample(
          self.config.batch_size)

      self.optimize_from_batch(states, actions, rewards, next_states, gammas)
      
      if self.config.opt_steps % self.config.target_network_update_freq == 0:
        for target_model, model in self.targets_and_models:
          soft_update(target_model, model, self.config.target_network_update_frac)

  def optimize_from_batch(self, states, actions, rewards, next_states, gammas):
    raise NotImplementedError('Subclass this!')


class DDPG(OffPolicyActorCritic):

  def optimize_from_batch(self, states, actions, rewards, next_states, gammas):

    with torch.no_grad():
      q_next = self.critic_target(next_states, self.actor_target(next_states))
      target = (rewards + gammas * q_next)
      target = torch.clamp(target, *self.config.clip_target_range)

    if hasattr(self, 'logger') and self.config.opt_steps % 1000 == 0:
      self.logger.add_histogram('Optimize/Target_q', target)
 
    q = self.critic(states, actions)
    critic_loss = F.mse_loss(q, target)

    self.critic_opt.zero_grad()
    critic_loss.backward()
      
    # Grad clipping
    if self.config.grad_norm_clipping > 0.:	
      for p in self.critic_params:
        clip_coef = self.config.grad_norm_clipping / (1e-6 + torch.norm(p.grad.detach()))
        if clip_coef < 1:
          p.grad.detach().mul_(clip_coef)
    if self.config.grad_value_clipping > 0.:
      torch.nn.utils.clip_grad_value_(self.critic_params, self.config.grad_value_clipping)

    self.critic_opt.step()

    for p in self.critic_params:
      p.requires_grad = False

    a = self.actor(states)
    if self.config.get('policy_opt_noise'):
      noise = torch.randn_like(a) * (self.config.policy_opt_noise * self.action_scale)
      a = (a + noise).clamp(-self.action_scale, self.action_scale)
      
    actor_loss = -self.critic(states, a)[:,-1].mean()
    if self.config.action_l2_regularization:
      actor_loss += self.config.action_l2_regularization * F.mse_loss(a / self.action_scale, torch.zeros_like(a))

    self.actor_opt.zero_grad()
    actor_loss.backward()
      
    # Grad clipping
    if self.config.grad_norm_clipping > 0.:	
      for p in self.actor_params:
        clip_coef = self.config.grad_norm_clipping / (1e-6 + torch.norm(p.grad.detach()))
        if clip_coef < 1:
          p.grad.detach().mul_(clip_coef)
    if self.config.grad_value_clipping > 0.:
      torch.nn.utils.clip_grad_value_(self.actor_params, self.config.grad_value_clipping)
      
    self.actor_opt.step()

    for p in self.critic_params:
      p.requires_grad = True


class TD3(OffPolicyActorCritic):

  def optimize_from_batch(self, states, actions, rewards, next_states, gammas):
    config = self.config

    with torch.no_grad():
      a_next_max = self.actor(next_states)
      noise = torch.randn_like(a_next_max) * (config.td3_noise * self.action_scale)
      noise = noise.clamp(-config.td3_noise_clip * self.action_scale,
                          config.td3_noise_clip * self.action_scale)
      a_next_max = (a_next_max + noise).clamp(-self.action_scale, self.action_scale)

      q1, q2 = self.critic_target(next_states, a_next_max), self.critic2_target(
          next_states, a_next_max)
      
      target = (rewards + gammas * torch.min(q1, q2))
      target = torch.clamp(target, *self.config.clip_target_range)

    if hasattr(self, 'logger') and self.config.opt_steps % 1000 == 0:
      self.logger.add_histogram('Optimize/Target_q', target)

    q1, q2 = self.critic(states, actions), self.critic2(states, actions)
    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

    self.critic_opt.zero_grad()
    critic_loss.backward()
      
    # Grad clipping
    if self.config.grad_norm_clipping > 0.:	
      torch.nn.utils.clip_grad_norm_(self.critic_params, self.config.grad_norm_clipping)
    if self.config.grad_value_clipping > 0.:
      torch.nn.utils.clip_grad_value_(self.critic_params, self.config.grad_value_clipping)

    self.critic_opt.step()

    if config.opt_steps % config.td3_delay == 0:
      for p in self.critic_params:
        p.requires_grad = False

      a = self.actor(states)
      if self.config.get('policy_opt_noise'):
        noise = torch.randn_like(a) * (config.policy_opt_noise * self.action_scale)
        a = (a + noise).clamp(-self.action_scale, self.action_scale)
      actor_loss = -torch.min(self.critic(states, a)[:,-1], self.critic2(states, a)[:,-1]).mean()
      if self.config.action_l2_regularization:
        actor_loss += self.config.action_l2_regularization * F.mse_loss(a / self.action_scale, torch.zeros_like(a))

      self.actor_opt.zero_grad()
      actor_loss.backward()
      
      # Grad clipping
      if self.config.grad_norm_clipping > 0.:	
        torch.nn.utils.clip_grad_norm_(self.actor_params, self.config.grad_norm_clipping)
      if self.config.grad_value_clipping > 0.:
        torch.nn.utils.clip_grad_value_(self.actor_params, self.config.grad_value_clipping)

      self.actor_opt.step()

      for p in self.critic_params:
        p.requires_grad = True


class SAC(OffPolicyActorCritic):

  def optimize_from_batch(self, states, actions, rewards, next_states, gammas):
    config = self.config

    with torch.no_grad():
      # Target actions come from *current* policy
      a_next, logp_next = self.actor(next_states)
      q1 = self.critic_target(next_states, a_next)
      q2 = self.critic2_target(next_states, a_next)
      target = rewards + gammas * (torch.min(q1, q2) - config.entropy_coef * logp_next)
      target = torch.clamp(target, *self.config.clip_target_range)

    if hasattr(self, 'logger') and self.config.opt_steps % 1000 == 0:
      self.logger.add_histogram('Optimize/Target_q', target)

    q1, q2 = self.critic(states, actions), self.critic2(states, actions)
    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

    self.critic_opt.zero_grad()
    critic_loss.backward()
      
    # Grad clipping
    if self.config.grad_norm_clipping > 0.:	
      torch.nn.utils.clip_grad_norm_(self.critic_params, self.config.grad_norm_clipping)
    if self.config.grad_value_clipping > 0.:
      torch.nn.utils.clip_grad_value_(self.critic_params, self.config.grad_value_clipping)

    self.critic_opt.step()

    for p in self.critic_params:
      p.requires_grad = False

    a, logp = self.actor(states)
    q = torch.min(self.critic(states, a), self.critic2(states, a))

    actor_loss = (config.entropy_coef * logp - q).mean()

    if self.config.action_l2_regularization:
      actor_loss += self.config.action_l2_regularization * F.mse_loss(a / self.action_scale, torch.zeros_like(a))

    self.actor_opt.zero_grad()
    actor_loss.backward()
      
    # Grad clipping
    if self.config.grad_norm_clipping > 0.:	
      torch.nn.utils.clip_grad_norm_(self.actor_params, self.config.grad_norm_clipping)
    if self.config.grad_value_clipping > 0.:
      torch.nn.utils.clip_grad_value_(self.actor_params, self.config.grad_value_clipping)

    self.actor_opt.step()

    for p in self.critic_params:
      p.requires_grad = True
