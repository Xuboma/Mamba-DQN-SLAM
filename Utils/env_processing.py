import gym
from gym import spaces
from gym.wrappers.time_limit import TimeLimit
import numpy as np
from typing import Union
from maze_env import Maze, height, width

try:
    from gym_gridverse.gym import GymEnvironment
    from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
    from gym_gridverse.outer_env import OuterEnv
    from gym_gridverse.representations.observation_representations import (
        make_observation_representation,
    )
    from gym_gridverse.representations.state_representations import (
        make_state_representation,
    )
except ImportError:
    print(
        f"WARNING: ``gym_gridverse`` is not installed. This means you cannot run an experiment with the `gv_*` domains."
    )
    GymEnvironment = None
# from envs.gv_wrapper import GridVerseWrapper
import os
from enum import Enum
from typing import Tuple

from Utils.random import RNG


class ObsType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1
    IMAGE = 2


def get_env_obs_type(env: Maze) -> int:
    # 从你的Maze环境中获取观察空间的大小
    obs_space = env.height*env.width
    sample_obs = env.reset()

    # 如果你的环境的观察是一个256维的向量，我们可以将其视为图片
    if isinstance(sample_obs, np.ndarray) and len(sample_obs) == 4096:
        print("输入类型是图片")
        return ObsType.IMAGE.value

    # 如果你的环境的观察空间是一个离散值，我们可以将其视为离散类型
    elif isinstance(obs_space, int):
        return ObsType.DISCRETE.value

    # 否则，我们将其视为连续类型
    else:
        return ObsType.CONTINUOUS.value


def get_env_obs_length(env: Maze) -> int:
    """Gets the length of the observations in an environment"""
    if get_env_obs_type(env) == ObsType.IMAGE.value:
        print("环境长度", env.reset().shape)
        return env.reset().shape
    elif isinstance(env.observation_space, gym.spaces.Discrete):
        return 1
    elif isinstance(env.observation_space, (gym.spaces.MultiDiscrete, gym.spaces.Box)):
        if len(env.observation_space.shape) != 1:
            raise NotImplementedError(f"We do not yet support 2D observation spaces")
        return env.observation_space.shape[0]
    elif isinstance(env.observation_space, spaces.MultiBinary):
        return env.observation_space.n
    else:
        raise NotImplementedError(f"We do not yet support {env.observation_space}")


def get_env_obs_mask(env: Maze) -> Union[int, np.ndarray]:
    """Gets the number of observations possible (for discrete case).
    For continuous case, please edit the -5 to something lower than
    lowest possible observation (while still being finite) so the
    network knows it is padding.
    """
    # Check image first
    if get_env_obs_type(env) == ObsType.IMAGE.value:
        print("图像不用处理mask,返回值是0")
        return 0
    if isinstance(env.observation_space, gym.spaces.Discrete):
        return env.observation_space.n
    elif isinstance(env.observation_space, gym.spaces.MultiDiscrete):
        return max(env.observation_space.nvec) + 1
    elif isinstance(env.observation_space, gym.spaces.Box):
        # If you would like to use DTQN with a continuous action space, make sure this value is
        #       below the minimum possible observation. Otherwise it will appear as a real observation
        #       to the network which may cause issues. In our case, Car Flag has min of -1 so this is
        #       fine.
        return -5
    else:
        raise NotImplementedError(f"We do not yet support {env.observation_space}")


def get_env_max_steps(env: gym.Env) -> Union[int, None]:
    """Gets the maximum steps allowed in an episode before auto-terminating"""
    try:
        return env._max_episode_steps
    except AttributeError:
        try:
            return env.max_episode_steps
        except AttributeError:
            return None
