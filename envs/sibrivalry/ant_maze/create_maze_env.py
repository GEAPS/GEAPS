# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

from envs.sibrivalry.ant_maze.ant_maze_env import AntMazeEnv


def create_maze_env(env_name=None, top_down_view=False):
  n_bins = 0
  # hard coded
  sense_wall = sense_dropoff = sense_block = False
  sensor_range = 2

  manual_collision = False
  if env_name.startswith('Ego'):
    n_bins = 8
    env_name = env_name[3:]
  if env_name.startswith('Sense'):
    n_bins = 4
    sense_wall = True
    env_name = env_name[5:]
  if env_name.startswith('Ant'):
    cls = AntMazeEnv
    env_name = env_name[3:]
    maze_size_scaling = 8
  else:
    assert False, 'unknown env %s' % env_name

  observe_blocks = False
  put_spin_near_agent = False
  if env_name == 'Maze_U':
    maze_id = 'Maze_U'
  elif env_name == 'Maze_S':
    maze_id = 'Maze_S'
  elif env_name == 'Push':
    maze_id = 'Push'
    observe_blocks = True
  elif env_name == 'Fall':
    maze_id = 'Fall'
    observe_blocks = True
  elif env_name == 'Block':
    maze_id = 'Block'
    put_spin_near_agent = True
    observe_blocks = True
  elif env_name == 'BlockMaze':
    maze_id = 'BlockMaze'
    put_spin_near_agent = True
    observe_blocks = True
  else:
    raise ValueError('Unknown maze environment %s' % env_name)

  gym_mujoco_kwargs = {
      'maze_id': maze_id,
      'n_bins': n_bins,
      'sensor_range': sensor_range,
      'observe_blocks': observe_blocks,
      'sense_wall': sense_wall,
      'sense_dropoff': sense_dropoff,
      'sense_block': sense_block,
      'put_spin_near_agent': put_spin_near_agent,
      'top_down_view': top_down_view,
      'manual_collision': manual_collision,
      'maze_size_scaling': maze_size_scaling
  }
  gym_env = cls(**gym_mujoco_kwargs)
  gym_env.reset()
  return gym_env