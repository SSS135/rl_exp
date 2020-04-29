import gym
import numpy as np
from gym import spaces
from gym.envs.registration import register

register(
    id='EchoEnvContinuous-v0',
    entry_point='mylib.ppo_pytorch.common.echo_env:EchoEnv',
    kwargs={'continuous': True},
)

register(
    id='EchoEnvDiscrete-v0',
    entry_point='mylib.ppo_pytorch.common.echo_env:EchoEnv',
    kwargs={'continuous': False},
)


D_ACT = 10


class EchoEnv(gym.Env):
    def __init__(self, continuous=False):
        super().__init__()
        self.continuous = continuous
        high = np.array([float('inf')])
        self.action_space = spaces.Box(-high, high) if continuous else spaces.Discrete(D_ACT)
        self.observation_space = spaces.Box(-high, high)
        self.reward_range = (-1, 1)

    def _reset(self):
        return np.array([0], dtype=np.float32)

    def _step(self, action):
        if np.isscalar(action):
            action = np.array([action], dtype=np.float32)
        return action, 1, False, {}
