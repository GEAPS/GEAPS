# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

from envs.sibrivalry.ant_maze.create_maze_env import create_maze_env

import gym
import numpy as np
import torch
from gym.utils import seeding
from envs.sibrivalry.ant_maze import maze_env_utils

class AntMazeEnv(gym.GoalEnv):
  """Wraps the HIRO/SR Ant Environments in a gym goal env."""
  def __init__(self, variant='AntMaze-SR', env_max_steps=500, eval=False):

    self.done_env = False
    
    # is the final dimension useful?
    state_dims = 30
    self._process_goal = None
    self.goal_dim = 0
    if 'antmaze' in variant:
      mazeinfo = variant.split('-')
      goal_type = "sr"
      shape = "s"

      if len(mazeinfo) == 3:
        mazename, shape, goal_type = mazeinfo
      elif len(mazeinfo) == 2:
        mazename, shape = mazeinfo
      else:
        mazename = mazeinfo[0]

      # the name will be in the format antmaze-{shape}-{goal type}
      assert mazename == 'antmaze'
      self.goal_dims = [0, 1]
      self.eval_dims = [0, 1]
      # make it configurable.
      maze = "Maze_" + shape.upper()
      mazename = "Ant" + maze
      if goal_type == 'sr':
        goal_range = maze_env_utils.default_desired_goal_distribution[maze]
        self.sample_goal = lambda: self.np_random.uniform(goal_range[0], goal_range[1]).astype(np.float32)
        self.dist_threshold = 1.0
        if eval:
          self.done_env = True
      else: # HIRO VERSION
        self.sample_goal = lambda: maze_env_utils.default_fixed_desired_goal[maze]
    else:
      mazename = variant.split('-')[0]
      state_dims = 33
      if mazename == 'AntPush':
        #self.sample_goal = lambda: np.array([0., 19., 1.], dtype=np.float32)
        #self.sample_goal = lambda: self.np_random.uniform([-3.5, 12.5, 1], [3.5, 19.5, 1]).astype(np.float32)
        #self.sample_goal = lambda: self.np_random.uniform([-3.5, 16, 1], [3.5, 19.5, 1]).astype(np.float32)
        #self.sample_goal = lambda: np.array([0., 19., 2., 8.], dtype=np.float32)
        self.sample_goal = lambda: self.np_random.uniform([-3.5, 12.5], [3.5, 19.5]).astype(np.float32)
        self.goal_dims = [0, 1] # we need to distinguish the different situatios of box positions.
        self.eval_dims = [0, 1]
        #self.goal_dims = [0, 1, 3, 4] # we need to distinguish the different situatios of box positions.
        #self.eval_dims = [0, 1, 2]
        #self._process_goal = lambda x: np.array([x[0], x[1], 0 if x[3]>8 else 1])
        #self.goal_dim = 3
        if eval:
          self.done_env = True
          self.eval_dims = [0, 1]
      elif mazename == 'AntFall':
        self.sample_goal = lambda: np.array([0., 27., 8., 16.], dtype=np.float32)
        self.goal_dims = [0, 1, 3, 4]
        self.eval_dims = [0, 1, 2, 3]
        if eval:
          self.eval_dims = [0, 1]
      else:
        raise ValueError('Bad maze name!')


    self.maze = create_maze_env(mazename) # this returns a gym environment
    self.seed()
    self.max_steps = env_max_steps
    self.dist_threshold = 1.0


    self.action_space = self.maze.action_space
    observation_space = gym.spaces.Box(-np.inf, np.inf, (state_dims,))
    if self.goal_dim > 0:
        goal_space        = gym.spaces.Box(-np.inf, np.inf, (self.goal_dim,)) # first few coords of state
    else:
        goal_space        = gym.spaces.Box(-np.inf, np.inf, (len(self.eval_dims),)) # first few coords of state
    self.observation_space = gym.spaces.Dict({
        'observation': observation_space,
        'desired_goal': goal_space,
        'achieved_goal': goal_space
    })
    self.num_steps = 0

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    self.maze.wrapped_env.seed(seed)
    return [seed]



  def step(self, action):
    next_state, _, _, _ = self.maze.step(action)
    next_state = next_state.astype(np.float32)

    s_xy = next_state[self.goal_dims]
    if self._process_goal:
      s_xy = self._process_goal(s_xy)
    reward = self.compute_reward(s_xy, self.g_xy, None)
    info = {}
    self.num_steps += 1
    
    is_success = np.allclose(0., reward)
    done = is_success and self.done_env
    info['is_success'] = is_success
    if self.num_steps >= self.max_steps and not done:
      done = True
      info['TimeLimit.truncated'] = True

    obs = {
      'observation': next_state,
      'achieved_goal': s_xy,
      'desired_goal': self.g_xy,
    }
          
    return obs, reward, done, info

  def reset(self): 
    self.num_steps = 0

    ## Not exactly sure why we are reseeding here, but it's what the SR env does
    _ = self.maze.wrapped_env.seed(self.np_random.randint(np.iinfo(np.int32).max))
    s = self.maze.reset().astype(np.float32)
    _ = self.maze.wrapped_env.seed(self.np_random.randint(np.iinfo(np.int32).max))
    self.g_xy = self.sample_goal()
    s_xy = s[self.goal_dims]
    if self._process_goal:
      s_xy = self._process_goal(s_xy)
    return {
      'observation': s,
      'achieved_goal': s_xy,
      'desired_goal': self.g_xy,
    }

  def render(self):
    self.maze.render()

  def compute_reward(self, achieved_goal, desired_goal, info):
    if len(achieved_goal.shape) == 2:
      ag = achieved_goal[:,self.eval_dims]
      dg = desired_goal[:,self.eval_dims]
    else:
      ag = achieved_goal[self.eval_dims]
      dg = desired_goal[self.eval_dims]
    d = np.linalg.norm(ag - dg, axis=-1)
    return -(d >= self.dist_threshold).astype(np.float32)
