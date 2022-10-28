import mrl
from mrl.utils.misc import AttrDict
import numpy as np
from copy import deepcopy
import time
from mrl.algorithms.cache_action import SkillActionCache

class StandardTrain(mrl.Module):
  def __init__(self):
    super().__init__('train', required_agent_modules = ['env', 'policy', 'optimize'], locals=locals())

  def _setup(self):
    assert hasattr(self.config, 'optimize_every')
    self.optimize_every = self.config.optimize_every
    self.env_steps = 0
    self.reset_idxs = []
    self.skill_action_cache  = None
    self.num_latents = self.config.n_latents
    # initialize the skills(latents) used in exploration.
    if self.config.use_skills:
      init_latent_idx = np.random.randint(0, self.num_latents, self.env.num_envs)
      self.latents = np.zeros((self.env.num_envs, self.num_latents))
      self.latents[np.arange(self.env.num_envs), init_latent_idx] = 1
      self.go_explore_count = np.zeros(self.env.num_envs)
      if self.config.use_cache:
        skill_dim = 3 #self.policy.skill_policy.skill_dim
        # not yet used
       
        self.skill_action_cache = SkillActionCache(self.env.num_envs, self.env.action_dim, skill_dim)
    else:
      self.latents = None
      self.go_explore_count = None
    

  def reset_latent(self, indices):
    assert isinstance(self.latents, np.ndarray), "The data type of the latents is not correct."
    num2reset = int(np.sum(indices))
    latent_indices = np.random.randint(0, self.num_latents, num2reset)
    self.latents[indices] = 0
    self.latents[indices, latent_indices] = 1
    if self.config.use_cache: 
      self.skill_action_cache.refresh_cache(indices) 


  # it doesn't support latent switches during exploration.
  def reset_latent_count(self, indices):
    if not self.config.use_skills:
      return
    self.reset_latent(indices)
    self.go_explore_count[indices] = 0

  def __call__(self, num_steps : int, render=False, dont_optimize=False, dont_train=False):
    """
    Runs num_steps steps in the environment, saves collected experiences,
    and trains at every step
    """
    if not dont_train:
      self.agent.train_mode()
    env = self.env
    state = env.state 
    for _ in range(num_steps // env.num_envs):
      action, go_explore_idx = self.policy(state, latents=self.latents)
      if self.config.use_cache:
          self.skill_action_cache.mod_action(go_explore_idx, action)
          self.skill_action_cache.cache_action(go_explore_idx, action)
     
      next_state, reward, done, info = env.step(action)

      if self.reset_idxs:
        env.reset(self.reset_idxs)
        for i in self.reset_idxs:
          done[i] = True
          if not 'done_observation' in info[i]:
            if isinstance(next_state, np.ndarray):
              info[i].done_observation = next_state[i]
            else:
              for key in next_state:
                info[i].done_observation = {k: next_state[k][i] for k in next_state}
        next_state = env.state
        self.reset_idxs = []
      
      if self.config.use_skills:
        self.go_explore_count[go_explore_idx] += 1
        # let #skill_last skills last until the endi
        skill_reset_indices = (self.go_explore_count % self.config.skill_horizon == 0) & \
                              (self.go_explore_count > 0)
        if np.any(skill_reset_indices): 
            self.reset_latent(skill_reset_indices)
        if np.any(done):
            self.reset_latent_count(done)

      state, experience = debug_vectorized_experience(state, action, next_state, reward, done, info, go_explore_idx)
      self.process_experience(experience)

      if render:
        time.sleep(0.02)
        env.render()
      
      for _ in range(env.num_envs):
        self.env_steps += 1
        if self.env_steps % self.optimize_every == 0 and not dont_optimize:
          self.optimize()
    
    # If using MEP prioritized replay, fit the density model
    if self.config.prioritized_mode == 'mep':
      self.prioritized_replay.fit_density_model()
      self.prioritized_replay.update_priority()
  
  def reset_next(self, idxs):
    """Resets specified envs on next step"""
    self.reset_idxs = idxs

  def save(self, save_folder):
    self._save_props(['env_steps'], save_folder)

  def load(self, save_folder):
    self._load_props(['env_steps'], save_folder)

def debug_vectorized_experience(state, action, next_state, reward, done, info, go_explore_idx):
  """Gym returns an ambiguous "done" signal. VecEnv doesn't 
  let you fix it until now. See ReturnAndObsWrapper in env.py for where
  these info attributes are coming from."""
  experience = AttrDict(
    state = state,
    action = action,
    reward = reward,
    info = info,
    go_explore_idx=go_explore_idx
  )
  next_copy = deepcopy(next_state) # deepcopy handles dict states

  for idx in np.argwhere(done):
    i = idx[0]
    if isinstance(next_copy, np.ndarray):
      next_copy[i] = info[i].done_observation
    else:
      assert isinstance(next_copy, dict)
      for key in next_copy:
        next_copy[key][i] = info[i].done_observation[key]
  
  experience.next_state = next_copy
  experience.trajectory_over = done # done or the timelimie is reached
  # the episode terminates
  experience.done = np.array([info[i].terminal_state for i in range(len(done))], dtype=np.float32)
  experience.reset_state = next_state
  
  return next_state, experience
