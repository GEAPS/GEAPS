import mrl
import gym
from mrl.replays.core.shared_buffer import SharedMemoryTrajectoryBuffer as Buffer
import numpy as np
import pickle
import os
import time
from mrl.utils.misc import batch_block_diag

class OnlineHERBuffer(mrl.Module):

  def __init__(
      self,
      module_name='replay_buffer'
    ):
    """
    Buffer that does online hindsight relabeling.
    Replaces the old combo of ReplayBuffer + HERBuffer.
    """

    super().__init__(module_name, required_agent_modules=['env'], locals=locals())

    self.size = None
    self.goal_space = None
    self.buffer = None
    self.save_buffer = None
    
  def _setup(self):
    self.size = self.config.replay_size

    env = self.env
    if type(env.observation_space) == gym.spaces.Dict:
      observation_space = env.observation_space.spaces["observation"]
      self.goal_space = env.observation_space.spaces["desired_goal"]
    else:
      observation_space = env.observation_space

    items = [("state", observation_space.shape),
             ("action", env.action_space.shape), ("reward", (1,)),
             ("next_state", observation_space.shape), ("done", (1,))]

    if self.goal_space is not None:
      items += [("previous_ag", self.goal_space.shape), # for reward shaping
                ("ag", self.goal_space.shape), # achieved goal
                ("bg", self.goal_space.shape), # behavioral goal (i.e., intrinsic if curious agent)
                ("dg", self.goal_space.shape), # desired goal (even if ignored behaviorally)
                ("go_explore", (1,))] # whether under go_explore status

    self.buffer = Buffer(self.size, items)
    self._subbuffers = [[] for _ in range(self.env.num_envs)]
    self._skill_horizon = np.zeros((self.env.num_envs, 1), dtype=np.float)
    self._skill_start = np.zeros((self.env.num_envs,), dtype=np.int32)
    self.n_envs = self.env.num_envs

    # HER mode can differ if demo or normal replay buffer
    if 'demo' in self.module_name:
      self.rel, self.fut, self.pst, self.act, self.ach, self.beh, self.clo = parse_hindsight_mode(self.config.demo_her)
    else:
      self.rel, self.fut, self.pst, self.act, self.ach, self.beh, self.clo = parse_hindsight_mode(self.config.her)

  def _process_experience(self, exp):
    if getattr(self, 'logger'):
      self.logger.add_tabular('Replay buffer size', len(self.buffer))
    done = np.expand_dims(exp.done, 1)  # format for replay buffer
    reward = np.expand_dims(exp.reward, 1)  # format for replay buffer
    action = exp.action
    go_explore_idx = np.expand_dims(exp.go_explore_idx.astype(float), 1)
    # self._skill_horizon += go_explore_idx

    if self.goal_space:
      state = exp.state['observation']
      next_state = exp.next_state['observation']
      previous_achieved = exp.state['achieved_goal']
      achieved = exp.next_state['achieved_goal']
      desired = exp.state['desired_goal']
      if hasattr(self, 'ag_curiosity') and self.ag_curiosity.current_goals is not None:
        behavioral = self.ag_curiosity.current_goals
        # recompute online reward
        reward = self.env.compute_reward(achieved, behavioral, {'s':state, 'ns':next_state}).reshape(-1, 1)
      else:
        behavioral = desired
      for i in range(self.n_envs):
        self._subbuffers[i].append([
            state[i], action[i], reward[i], next_state[i], done[i], previous_achieved[i], achieved[i],
            behavioral[i], desired[i], go_explore_idx[i]
        ])
        # if self._skill_horizon[i] == 1:
        #   self._skill_start[i] = len(self._subbuffers[i]) - 1

        # if self._skill_horizon[i] > 0 and ( exp.trajectory_over[i] or\
        #   self._skill_horizon[i] % self.config.skill_horizon == 0):
        #   self._skill_horizon[i] = 0
        #   for j in range(self._skill_start[i], len(self._subbuffers[i])):
        #     self._subbuffers[i][j][-2] = achieved[i] # modify the desired goal
    else:
      state = exp.state
      next_state = exp.next_state
      for i in range(self.n_envs):
        self._subbuffers[i].append(
            [state[i], action[i], reward[i], next_state[i], done[i]])

    for i in range(self.n_envs):
      if exp.trajectory_over[i]:
        trajectory = [np.stack(a) for a in zip(*self._subbuffers[i])]
        self.buffer.add_trajectory(*trajectory)
        self._subbuffers[i] = []
        self._skill_horizon[i] = 0

  def sample(self, batch_size, to_torch=True):
    if hasattr(self, 'prioritized_replay'):
      batch_idxs = self.prioritized_replay(batch_size)
    else:
      batch_idxs = np.random.randint(self.buffer.size, size=batch_size)

    if self.goal_space:
      if "demo" in self.module_name:
        has_config_her = self.config.get('demo_her')
      else:
        has_config_her = self.config.get('her')
      
      if has_config_her:

        if self.config.env_steps > self.config.future_warm_up:
          fut_batch_size, pst_batch_size, act_batch_size, ach_batch_size, beh_batch_size, clo_batch_size, real_batch_size = np.random.multinomial(
              batch_size, [self.fut, self.pst, self.act, self.ach, self.beh, self.clo, self.rel])
        else:
          fut_batch_size, pst_batch_size, act_batch_size, ach_batch_size, beh_batch_size, clo_batch_size, real_batch_size  = batch_size, 0, 0, 0, 0, 0, 0

        fut_idxs, pst_idxs, act_idxs, ach_idxs, beh_idxs, clo_idxs, real_idxs = np.array_split(batch_idxs, 
          np.cumsum([fut_batch_size, pst_batch_size, act_batch_size, ach_batch_size, beh_batch_size, clo_batch_size]))


        # # Sample the real batch (i.e., goals = behavioral goals)
        if self.config.use_skills and real_batch_size > 0:
          # avoid those go explore data.
          all_idx = np.arange(len(self.buffer))
          non_go_explore_idx = (1.0 - self.buffer.BUFF.buffer_go_explore.data[:len(self.buffer)].copy()).astype(bool)
          candidates = all_idx[non_go_explore_idx.flatten()]
          real_idxs = np.random.choice(candidates, real_batch_size)
          
        states, actions, rewards, next_states, dones, previous_ags, ags, goals, _, _ =\
            self.buffer.sample(real_batch_size, batch_idxs=real_idxs)

        states_fut, actions_fut, _, next_states_fut, dones_fut, previous_ags_fut, ags_fut, _, _, _, goals_fut =\
            self.buffer.sample_future(fut_batch_size, batch_idxs=fut_idxs)

        if pst_batch_size > 0: # this would be set to zero by default.
          all_idx = np.arange(len(self.buffer))
          go_explore_idx = self.buffer.BUFF.buffer_go_explore.data[:len(self.buffer)].copy().astype(bool)
          candidates = all_idx[go_explore_idx.flatten()]
          pst_idxs = np.random.choice(candidates, pst_batch_size)
        states_pst, actions_pst, _, next_states_pst, dones_pst, previous_ags_pst, ags_pst, _, _, _, goals_pst =\
            self.buffer.sample_past(pst_batch_size, batch_idxs=pst_idxs)

        # Sample the actual batch
        states_act, actions_act, _, next_states_act, dones_act, previous_ags_act, ags_act, _, _, _, goals_act =\
          self.buffer.sample_from_goal_buffer('dg', act_batch_size, batch_idxs=act_idxs)

        # Sample the achieved batch
        states_ach, actions_ach, _, next_states_ach, dones_ach, previous_ags_ach, ags_ach, _, _, _, goals_ach =\
          self.buffer.sample_from_goal_buffer('ag', ach_batch_size, batch_idxs=ach_idxs)

        # Sample the behavioral batch
        states_beh, actions_beh, _, next_states_beh, dones_beh, previous_ags_beh, ags_beh, _, _, _, goals_beh =\
          self.buffer.sample_from_goal_buffer('bg', beh_batch_size, batch_idxs=beh_idxs)

        # ignore the allocated batch size and try to find sparse ones. 
        states_clo, actions_clo, _, next_states_clo, dones_clo, previous_ags_clo, ags_clo, _, _, _, goals_clo =\
          self.buffer.sample_close_to_sparse(clo_batch_size, self.ag_kde)
        # select the goals of lowest densities.
        # randomly select the within radius of 10% steps - 50 for 500 and 5 for 50.
        # take up to 10%.

        # Concatenate the five
        states = np.concatenate([states, states_fut, states_pst, states_act, states_ach, states_beh, states_clo], 0)
        actions = np.concatenate([actions, actions_fut, actions_pst, actions_act, actions_ach, actions_beh, actions_clo], 0)
        ags = np.concatenate([ags, ags_fut, ags_pst, ags_act, ags_ach, ags_beh, ags_clo], 0)
        goals = np.concatenate([goals, goals_fut, goals_pst, goals_act, goals_ach, goals_beh, goals_clo], 0)
        next_states = np.concatenate([next_states, next_states_fut, next_states_pst, next_states_act, next_states_ach,\
           next_states_beh, next_states_clo], 0)

        # Recompute reward online
        if hasattr(self, 'goal_reward'):
          rewards = self.goal_reward(ags, goals, {'s':states, 'ns':next_states}).reshape(-1, 1).astype(np.float32)
        else:
          rewards = self.env.compute_reward(ags, goals, {'s':states, 'ns':next_states}).reshape(-1, 1).astype(np.float32)

        if self.config.get('never_done'):
          dones = np.zeros_like(rewards, dtype=np.float32)
        elif self.config.get('first_visit_succ'):
          dones = np.round(rewards + 1.)
        else:
          raise ValueError("Never done or first visit succ must be set in goal environments to use HER.")
          # dones = np.concatenate([dones, dones_fut, dones_act, dones_ach, dones_beh], 0)

        if self.config.sparse_reward_shaping:
          previous_ags = np.concatenate([previous_ags, previous_ags_fut, previous_ags_pst, \
            previous_ags_act, previous_ags_ach, previous_ags_beh, previous_ags_clo], 0)
          previous_phi = -np.linalg.norm(previous_ags - goals, axis=1, keepdims=True)
          current_phi  = -np.linalg.norm(ags - goals, axis=1, keepdims=True)
          rewards_F = self.config.gamma * current_phi - previous_phi
          rewards += self.config.sparse_reward_shaping * rewards_F

      else:
        # Uses the original desired goals
        states, actions, rewards, next_states, dones, _ , _, _, goals =\
                                                    self.buffer.sample(batch_size, batch_idxs=batch_idxs)

      if self.config.slot_based_state:
        # TODO: For now, we flatten according to config.slot_state_dims
        I, J = self.config.slot_state_dims
        states = np.concatenate((states[:, I, J], goals), -1)
        next_states = np.concatenate((next_states[:, I, J], goals), -1)
      else:
        states = np.concatenate((states, goals), -1)
        next_states = np.concatenate((next_states, goals), -1)
      gammas = self.config.gamma * (1.-dones)

    elif self.config.get('n_step_returns') and self.config.n_step_returns > 1:
      states, actions, rewards, next_states, dones = self.buffer.sample_n_step_transitions(
        batch_size, self.config.n_step_returns, self.config.gamma, batch_idxs=batch_idxs
      )
      gammas = self.config.gamma**self.config.n_step_returns * (1.-dones)

    else:
      states, actions, rewards, next_states, dones = self.buffer.sample(
          batch_size, batch_idxs=batch_idxs)
      gammas = self.config.gamma * (1.-dones)

    if hasattr(self, 'state_normalizer'):
      states = self.state_normalizer(states, update=False).astype(np.float32)
      next_states = self.state_normalizer(
          next_states, update=False).astype(np.float32)
    
    if to_torch:
      return (self.torch(states), self.torch(actions),
            self.torch(rewards), self.torch(next_states),
            self.torch(gammas))
    else:
      return (states, actions, rewards, next_states, gammas)

  def __len__(self):
    return len(self.buffer)

  def save(self, save_folder):
    if self.config.save_replay_buf or self.save_buffer:
      state = self.buffer._get_state()
      with open(os.path.join(save_folder, "{}.pickle".format(self.module_name)), 'wb') as f:
        pickle.dump(state, f)

  def load(self, save_folder):
    load_path = os.path.join(save_folder, "{}.pickle".format(self.module_name))
    if os.path.exists(load_path):
      with open(load_path, 'rb') as f:
        state = pickle.load(f)
      self.buffer._set_state(state)
    else:
      self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='cyan')
      self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='red')
      self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='yellow')

def parse_hindsight_mode(hindsight_mode : str):
  if 'future_' in hindsight_mode:
    _, fut = hindsight_mode.split('_')
    rel = 1. / (1. + float(fut))
    fut = float(fut) / (1. + float(fut))
    pst = 0.
    act = 0.
    ach = 0.
    beh = 0.
    clo = 0.
  elif 'futureactual_' in hindsight_mode:
    _, fut, act = hindsight_mode.split('_')
    non_hindsight_frac = 1. / (1. + float(fut) + float(act))
    rel = non_hindsight_frac
    fut = float(fut) * non_hindsight_frac
    pst = 0.
    act = float(act) * non_hindsight_frac
    ach = 0.
    beh = 0.
    clo = 0
  elif 'futureachieved_' in hindsight_mode:
    _, fut, ach = hindsight_mode.split('_')
    non_hindsight_frac = 1. / (1. + float(fut) + float(ach))
    rel = non_hindsight_frac
    fut = float(fut) * non_hindsight_frac
    pst = 0.
    act = 0.
    ach = float(ach) * non_hindsight_frac
    beh = 0.
    clo = 0
  elif 'rfaa_' in hindsight_mode:
    _, real, fut, act, ach = hindsight_mode.split('_')
    denom = (float(real) + float(fut) + float(act) + float(ach))
    rel = float(real) / denom
    fut = float(fut) / denom
    pst = 0.
    act = float(act) / denom
    ach = float(ach) / denom
    beh = 0.
    clo = 0
  elif 'rfaab_' in hindsight_mode:
    _, real, fut, act, ach, beh = hindsight_mode.split('_')
    denom = (float(real) + float(fut) + float(act) + float(ach) + float(beh))
    rel = float(real) / denom
    fut = float(fut) / denom
    pst = 0.
    act = float(act) / denom
    ach = float(ach) / denom
    beh = float(beh) / denom
    clo = 0
  elif 'rfpaab_' in hindsight_mode:
    _, real, fut, pst, act, ach, beh = hindsight_mode.split('_')
    denom = (float(real) + float(fut) + float(pst) + float(act) + float(ach) + float(beh))
    rel = float(real) / denom
    fut = float(fut) / denom
    pst = float(pst) / denom
    act = float(act) / denom
    ach = float(ach) / denom
    beh = float(beh) / denom
    clo = 0
  elif 'rfaabc_' in hindsight_mode:
    _, real, fut, act, ach, beh, clo = hindsight_mode.split('_')
    denom = (float(real) + float(fut) + float(act) + float(ach) + float(beh) + float(clo))
    rel = float(real) / denom
    fut = float(fut) / denom
    pst = 0.
    act = float(act) / denom
    ach = float(ach) / denom
    beh = float(beh) / denom
    clo = float(clo) / denom
  else:
    rel = 1.
    fut = 0.
    pst = 0.
    act = 0.
    ach = 0.
    beh = 0.
    clo = 0.

  return rel, fut, pst, act, ach, beh, clo

