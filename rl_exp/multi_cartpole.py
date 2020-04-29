import gym
import gym.spaces
import numpy as np
from gym.envs.registration import register

register(
    id='MultiCartPole-v1',
    entry_point='mylib.ppo_pytorch.common.multi_cartpole:MultiCartPole',
    kwargs={'num_envs': 3, 'base_env': 'CartPole-v1'},
    max_episode_steps=500,
    reward_threshold=475.0,
)


class MultiCartPole(gym.Env):
    def __init__(self, num_envs, base_env):
        self.envs = [gym.make(base_env) for _ in range(num_envs)]
        n = self.envs[0].observation_space.n
        high = np.tile(self.envs[0].action_space.high, num_envs)
        self.observation_space = gym.spaces.Box(-high, high)
        self.action_space = gym.spaces.MultiDiscrete([[0, n]] * num_envs)
        self.reward_range = (0, 1)

    def _step(self, action):
        s, r, d, _ = zip(*[e.step(a) for e, a in zip(self.envs, action)])
        d = any(d)
        return np.concatenate(s), 0 if d else 1, d, {}

    def _reset(self):
        return np.concatenate([e.reset() for e in self.envs])