# GEAPS

This code base imeplements the algorithm `Goal Exploration Augmentation via Pre-trained Skills` (GEAPS) and is established on modular RL ([mrl](\url{https://github.com/spitis/mrl})) code base. For the implementation of Skew-Fit, we use part of codes from [rlkit](https://github.com/rail-berkeley/rlkit).

## Installation

There is a `requirements.txt` that was works with venv:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then `pip install` the appropriate version of `Pytorch` by following the instructions here: https://pytorch.org/get-started/locally/.

To run `Mujoco` environments you need to have the Mujoco binaries and a license key. Follow the instructions [here](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key).

To test run:

```
pytest tests
PYTHONPATH=./ python experiments/mega/train_mega.py --env FetchReach-v1 --layers 256 256 --max_steps 5000
```

The first command should have 3/3 success.
The second command should solve the environment in <1 minute (better than -5 avg test reward). 

## Pre-trained Skills
There are three different skill_types: `maxent`(ours), `snn4hrl` and `edl`.

We use the code base [rllab](https://github.com/rll/rllab) to train `maxent` and `snn4hrl` skills for both *AntMaze* and *PointMaze*. `edl` skills are trained via the code base [edl](https://github.com/victorcampos7/edl). Those pre-trained skills are put under the folder `mrl/algorithms/models`.

## Reproduction
To reproduce the results, pleas refer to the commands listed in `experiments/mega/commands_geaps.txt`.

